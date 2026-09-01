"""Plot aligned baseline and CAR raw-logit distributions from prediction CSVs."""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


REQUIRED_FIELDS = ('generator', 'path', 'label', 'raw_logit', 'score')
SUPPORTED_FORMATS = ('svg', 'pdf', 'png')
REAL_COLOR = '#6f8fad'
FAKE_COLOR = '#d4935d'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Plot aligned C2P-CLIP baseline and CAR raw-logit '
            'distributions from prediction CSVs'
        ),
    )
    parser.add_argument('--baseline_csv', required=True)
    parser.add_argument('--car_csv', required=True)
    parser.add_argument('--protocol_label', required=True)
    parser.add_argument('--output_prefix', required=True)
    parser.add_argument('--bins', type=int, default=70)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument(
        '--plot_kind',
        choices=('histogram', 'ecdf'),
        default='histogram',
        help='density histogram or empirical cumulative distribution',
    )
    parser.add_argument(
        '--density_scale',
        choices=('linear', 'log'),
        default='linear',
        help='linear density or log-density y-axis; log preserves tail detail',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=SUPPORTED_FORMATS,
        default=list(SUPPORTED_FORMATS),
    )
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)

    if args.bins <= 0:
        parser.error('--bins must be a positive integer')
    if args.dpi <= 0:
        parser.error('--dpi must be a positive integer')
    if not math.isfinite(args.threshold):
        parser.error('--threshold must be finite')
    if len(args.formats) != len(set(args.formats)):
        parser.error('--formats cannot contain duplicates')
    if args.plot_kind == 'ecdf' and args.density_scale != 'linear':
        parser.error('--density_scale applies only to histogram plots')
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_csv(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'prediction CSV not found: {path}')

    records = []
    seen_paths = set()
    with path.open(newline='', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f'prediction CSV is missing fields: {sorted(missing)}')

        for row_index, row in enumerate(reader, start=2):
            missing_values = [
                field
                for field in REQUIRED_FIELDS
                if row.get(field) is None or not row[field].strip()
            ]
            if missing_values:
                raise ValueError(
                    f'missing values at CSV row {row_index}: '
                    f'{missing_values}')
            try:
                label = int(row['label'])
                raw_logit = float(row['raw_logit'])
                score = float(row['score'])
            except ValueError as error:
                raise ValueError(
                    f'invalid numeric value at CSV row {row_index}') from error
            if label not in (0, 1):
                raise ValueError(
                    f'label must be 0 or 1 at CSV row {row_index}')
            if not math.isfinite(raw_logit):
                raise ValueError(
                    f'raw_logit must be finite at CSV row {row_index}')
            if not math.isfinite(score):
                raise ValueError(
                    f'score must be finite at CSV row {row_index}')

            image_path = row['path'].strip()
            if image_path in seen_paths:
                raise ValueError(f'duplicate path in prediction CSV: {image_path}')
            seen_paths.add(image_path)
            records.append({
                'generator': row['generator'].strip(),
                'path': image_path,
                'label': label,
                'raw_logit': raw_logit,
                'score': score,
            })

    if not records:
        raise ValueError('prediction CSV contains no records')
    if {record['label'] for record in records} != {0, 1}:
        raise ValueError('prediction CSV must contain both real and fake labels')
    return path, records


def validate_alignment(baseline_records, car_records):
    identity_fields = ('generator', 'path', 'label')
    baseline_identities = [
        tuple(record[field] for field in identity_fields)
        for record in baseline_records
    ]
    car_identities = [
        tuple(record[field] for field in identity_fields)
        for record in car_records
    ]
    if set(baseline_identities) != set(car_identities):
        raise ValueError(
            'baseline and CAR prediction CSVs contain different image sets')
    if baseline_identities != car_identities:
        raise ValueError(
            'baseline and CAR prediction CSVs use different image order')


def split_logits(records):
    return {
        'real': np.asarray([
            record['raw_logit'] for record in records
            if record['label'] == 0
        ], dtype=np.float64),
        'fake': np.asarray([
            record['raw_logit'] for record in records
            if record['label'] == 1
        ], dtype=np.float64),
    }


def configure_matplotlib():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 7.0,
        'axes.labelsize': 8.0,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.0,
        'legend.fontsize': 7.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def decorate_panel(axis, title, threshold):
    axis.axvline(
        threshold,
        color='#777777',
        linestyle=(0, (3, 2)),
        linewidth=0.8,
    )
    axis.annotate(
        'Default threshold',
        xy=(threshold, 0.97),
        xycoords=('data', 'axes fraction'),
        xytext=(3, 0),
        textcoords='offset points',
        color='#666666',
        fontsize=7.0,
        rotation=90,
        ha='left',
        va='top',
    )
    axis.set_title(title, pad=5)
    axis.set_xlabel('Fake logit')
    axis.tick_params(width=0.7, length=3)
    axis.grid(axis='y', color='#dddddd', linewidth=0.5, alpha=0.55)
    axis.legend(frameon=False, loc='upper right')


def plot_histogram_panel(axis, distributions, bin_edges, title, threshold):
    for class_name, color in (('real', REAL_COLOR), ('fake', FAKE_COLOR)):
        values = distributions[class_name]
        display_name = class_name.capitalize()
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='stepfilled',
            alpha=0.34,
            color=color,
            edgecolor=color,
            linewidth=0.75,
            label=f'{display_name} (n={values.size:,})',
        )
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='step',
            color=color,
            linewidth=0.8,
        )

    axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    decorate_panel(axis, title, threshold)


def plot_ecdf_panel(axis, distributions, bin_edges, title, threshold):
    for class_name, color in (('real', REAL_COLOR), ('fake', FAKE_COLOR)):
        values = np.sort(distributions[class_name])
        cumulative = np.arange(1, values.size + 1, dtype=np.float64) / values.size
        display_name = class_name.capitalize()
        axis.step(
            values,
            cumulative,
            where='post',
            color=color,
            linewidth=1.15,
            label=f'{display_name} (n={values.size:,})',
        )
    axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    axis.set_ylim(0.0, 1.01)
    decorate_panel(axis, title, threshold)


def build_figure(
    baseline,
    car,
    bin_edges,
    threshold,
    protocol_label,
    density_scale,
    plot_kind,
):
    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.75),
        sharex=True,
        sharey=True,
    )
    panel_plotter = (
        plot_histogram_panel if plot_kind == 'histogram' else plot_ecdf_panel)
    panel_plotter(axes[0], baseline, bin_edges, 'C2P-CLIP', threshold)
    panel_plotter(axes[1], car, bin_edges, 'C2P-CLIP + CAR', threshold)
    if plot_kind == 'histogram':
        for axis in axes:
            axis.set_yscale(density_scale)
        y_label = (
            'Density' if density_scale == 'linear'
            else 'Density (log scale)')
    else:
        y_label = 'Cumulative probability'
    axes[0].set_ylabel(y_label)
    for label, axis in zip(('a', 'b'), axes):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9.0,
            fontweight='bold',
            ha='left',
            va='bottom',
        )
    figure.suptitle(protocol_label, fontsize=9.0, y=0.995)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.84,
        wspace=0.08,
    )
    return figure


def summarize_input(path, records):
    return {
        'path': str(path),
        'sha256': file_sha256(path),
        'samples': len(records),
        'real_samples': sum(record['label'] == 0 for record in records),
        'fake_samples': sum(record['label'] == 1 for record in records),
        'generators': len({record['generator'] for record in records}),
    }


def print_stats(label, stats, distributions):
    print(f'[{label}]')
    print(
        f"  Real: mean={stats['real_mean']:.6f} "
        f"std={stats['real_std']:.6f} n={distributions['real'].size}")
    print(
        f"  Fake: mean={stats['fake_mean']:.6f} "
        f"std={stats['fake_std']:.6f} n={distributions['fake'].size}")
    print(f"  Separation: {stats['separation']:.6f}")


def run(args):
    start_time = time.time()
    baseline_path, baseline_records = load_prediction_csv(args.baseline_csv)
    car_path, car_records = load_prediction_csv(args.car_csv)
    validate_alignment(baseline_records, car_records)

    baseline = split_logits(baseline_records)
    car = split_logits(car_records)
    distributions = [
        baseline['real'],
        baseline['fake'],
        car['real'],
        car['fake'],
    ]
    bin_edges = np.asarray(
        build_shared_bin_edges(distributions, bins=args.bins),
        dtype=np.float64,
    )
    baseline_stats = compute_logit_stats(
        baseline['real'], baseline['fake'])
    car_stats = compute_logit_stats(car['real'], car['fake'])
    print_stats('C2P-CLIP', baseline_stats, baseline)
    print_stats('C2P-CLIP + CAR', car_stats, car)

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(
        baseline,
        car,
        bin_edges,
        threshold=args.threshold,
        protocol_label=args.protocol_label,
        density_scale=args.density_scale,
        plot_kind=args.plot_kind,
    )
    outputs = {}
    try:
        for output_format in args.formats:
            output_path = Path(f'{output_prefix}.{output_format}')
            save_options = {
                'bbox_inches': 'tight',
                'facecolor': 'white',
            }
            if output_format == 'png':
                save_options['dpi'] = args.dpi
            figure.savefig(output_path, **save_options)
            outputs[output_format] = str(output_path)
            print(f'Saved {output_format.upper()}: {output_path}')
    finally:
        plt.close(figure)

    elapsed = time.time() - start_time
    summary = {
        'schema_version': 1,
        'protocol_label': args.protocol_label,
        'inputs': {
            'baseline': summarize_input(baseline_path, baseline_records),
            'car': summarize_input(car_path, car_records),
        },
        'alignment': {
            'identity_fields': ['generator', 'path', 'label'],
            'same_set_and_order': True,
            'samples': len(baseline_records),
        },
        'histogram': {
            'bins': args.bins,
            'shared_bin_edges': bin_edges.tolist(),
            'plot_kind': args.plot_kind,
            'density': args.plot_kind == 'histogram',
            'density_scale': args.density_scale,
            'threshold': args.threshold,
        },
        'statistics': {
            'baseline': baseline_stats,
            'car': car_stats,
        },
        'outputs': outputs,
        'metadata': {
            'script': str(Path(__file__).resolve()),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'runtime_seconds': elapsed,
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'matplotlib_version': matplotlib.__version__,
            'platform': platform.platform(),
        },
    }
    summary_path = Path(f'{output_prefix}.summary.json')
    summary['outputs']['summary'] = str(summary_path)
    with summary_path.open('w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write('\n')
    print(f'Saved summary: {summary_path}')
    return summary


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == '__main__':
    main()
