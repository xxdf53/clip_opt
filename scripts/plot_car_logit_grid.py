"""Plot a C2P-CLIP Figure-5-style baseline-versus-CAR logit grid.

The figure uses two method rows (C2P-CLIP and CAR) and explicit test-source
columns.  Prediction CSVs must contain the fields emitted by the unified
binary evaluator: generator, path, label, raw_logit, and score.

The script never selects sources from their measured performance.  Sources
are fixed by the command line, all matching observations are retained, and
baseline/CAR image identities and ordering must agree exactly.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
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
from matplotlib.lines import Line2D

from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


REQUIRED_FIELDS = ('generator', 'path', 'label', 'raw_logit', 'score')
SUPPORTED_FORMATS = ('svg', 'pdf', 'png')
REAL_COLOR = '#6F9FC7'
GENERATED_COLOR = '#E89A55'
DEFAULT_GAN_SOURCES = (
    'deepfake=Deepfakes',
    'crn=CRN',
)
DEFAULT_DIFFUSION_SOURCES = (
    'adm=ADM',
    'vqdm=VQDM',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Plot representative class-conditional raw-logit distributions '
            'for matched C2P-CLIP baseline and CAR predictions.'
        ),
    )
    parser.add_argument(
        '--gan_baseline_csv',
        nargs='+',
        required=True,
        help='one or more GAN-protocol baseline prediction CSVs',
    )
    parser.add_argument(
        '--gan_car_csv',
        nargs='+',
        required=True,
        help='one or more GAN-protocol CAR prediction CSVs',
    )
    parser.add_argument(
        '--diffusion_baseline_csv',
        nargs='+',
        required=True,
        help='one or more diffusion-protocol baseline prediction CSVs',
    )
    parser.add_argument(
        '--diffusion_car_csv',
        nargs='+',
        required=True,
        help='one or more diffusion-protocol CAR prediction CSVs',
    )
    parser.add_argument(
        '--gan_sources',
        nargs='+',
        default=list(DEFAULT_GAN_SOURCES),
        metavar='CSV_NAME=DISPLAY_NAME',
        help=(
            'fixed GAN source columns; default: '
            'deepfake=Deepfakes crn=CRN'
        ),
    )
    parser.add_argument(
        '--diffusion_sources',
        nargs='+',
        default=list(DEFAULT_DIFFUSION_SOURCES),
        metavar='CSV_NAME=DISPLAY_NAME',
        help=(
            'fixed diffusion source columns; default: adm=ADM vqdm=VQDM'
        ),
    )
    parser.add_argument('--output_prefix', required=True)
    parser.add_argument('--bins', type=int, default=70)
    parser.add_argument(
        '--gan_bins',
        type=int,
        default=None,
        help='GAN histogram bins; overrides --bins for the GAN protocol',
    )
    parser.add_argument(
        '--diffusion_bins',
        type=int,
        default=None,
        help=(
            'diffusion histogram bins; overrides --bins for the diffusion '
            'protocol'
        ),
    )
    parser.add_argument(
        '--gan_plot_kind',
        choices=('histogram', 'ecdf'),
        default='histogram',
    )
    parser.add_argument(
        '--diffusion_plot_kind',
        choices=('histogram', 'ecdf'),
        default='histogram',
    )
    parser.add_argument(
        '--gan_density_scale',
        choices=('linear', 'log'),
        default='linear',
    )
    parser.add_argument(
        '--diffusion_density_scale',
        choices=('linear', 'log'),
        default='linear',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=SUPPORTED_FORMATS,
        default=list(SUPPORTED_FORMATS),
    )
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--width', type=float, default=7.2, help='figure width in inches')
    parser.add_argument('--height', type=float, default=3.45, help='figure height in inches')
    args = parser.parse_args(argv)

    if args.bins <= 0:
        parser.error('--bins must be a positive integer')
    if args.gan_bins is not None and args.gan_bins <= 0:
        parser.error('--gan_bins must be a positive integer')
    if args.diffusion_bins is not None and args.diffusion_bins <= 0:
        parser.error('--diffusion_bins must be a positive integer')
    if args.dpi <= 0:
        parser.error('--dpi must be a positive integer')
    if args.width <= 0 or args.height <= 0:
        parser.error('--width and --height must be positive')
    if len(args.formats) != len(set(args.formats)):
        parser.error('--formats cannot contain duplicates')
    if args.gan_plot_kind == 'ecdf' and args.gan_density_scale != 'linear':
        parser.error('--gan_density_scale=log cannot be combined with ECDF')
    if (
        args.diffusion_plot_kind == 'ecdf'
        and args.diffusion_density_scale != 'linear'
    ):
        parser.error(
            '--diffusion_density_scale=log cannot be combined with ECDF')

    args.gan_sources = parse_source_specs(args.gan_sources)
    args.diffusion_sources = parse_source_specs(args.diffusion_sources)
    return args


def parse_source_specs(values):
    specs = []
    seen = set()
    for value in values:
        csv_name, separator, display_name = value.partition('=')
        csv_name = csv_name.strip()
        display_name = display_name.strip() if separator else csv_name
        if not csv_name or not display_name:
            raise ValueError(
                'source specifications must use CSV_NAME=DISPLAY_NAME')
        normalized = normalize_source_name(csv_name)
        if not normalized:
            raise ValueError(f'invalid source name: {value!r}')
        if normalized in seen:
            raise ValueError(f'duplicate source specification: {csv_name}')
        seen.add(normalized)
        specs.append({
            'csv_name': csv_name,
            'normalized_name': normalized,
            'display_name': display_name,
        })
    if not specs:
        raise ValueError('at least one source is required for each protocol')
    return specs


def normalize_source_name(value):
    return re.sub(r'[^a-z0-9]+', '', value.casefold())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_csvs(paths):
    resolved_paths = []
    records = []
    seen_identities = set()

    for input_path in paths:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'prediction CSV not found: {path}')
        resolved_paths.append(path)

        with path.open(newline='', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f'{path.name} is missing fields: {sorted(missing)}')

            for row_index, row in enumerate(reader, start=2):
                missing_values = [
                    field
                    for field in REQUIRED_FIELDS
                    if row.get(field) is None or not row[field].strip()
                ]
                if missing_values:
                    raise ValueError(
                        f'{path.name} row {row_index} has missing values: '
                        f'{missing_values}')
                try:
                    label = int(row['label'])
                    raw_logit = float(row['raw_logit'])
                    score = float(row['score'])
                except ValueError as error:
                    raise ValueError(
                        f'{path.name} row {row_index} has invalid numeric '
                        'values') from error
                if label not in (0, 1):
                    raise ValueError(
                        f'{path.name} row {row_index} label must be 0 or 1')
                if not math.isfinite(raw_logit) or not math.isfinite(score):
                    raise ValueError(
                        f'{path.name} row {row_index} contains a non-finite '
                        'raw_logit or score')

                record = {
                    'generator': row['generator'].strip(),
                    'path': row['path'].strip(),
                    'label': label,
                    'raw_logit': raw_logit,
                    'score': score,
                }
                identity = record_identity(record)
                if identity in seen_identities:
                    raise ValueError(
                        'duplicate (generator, path, label) across prediction '
                        f'CSVs: {identity}')
                seen_identities.add(identity)
                records.append(record)

    if not records:
        raise ValueError('prediction CSV inputs contain no records')
    return resolved_paths, records


def record_identity(record):
    return (record['generator'], record['path'], record['label'])


def validate_alignment(baseline_records, car_records, protocol_label):
    baseline_identities = [record_identity(record) for record in baseline_records]
    car_identities = [record_identity(record) for record in car_records]
    if set(baseline_identities) != set(car_identities):
        raise ValueError(
            f'{protocol_label} baseline and CAR CSVs contain different image '
            'sets')
    if baseline_identities != car_identities:
        raise ValueError(
            f'{protocol_label} baseline and CAR CSVs use different image order')


def available_sources(records):
    names = {}
    for record in records:
        normalized = normalize_source_name(record['generator'])
        names.setdefault(normalized, record['generator'])
    return names


def source_logits(records, source_spec):
    selected = [
        record
        for record in records
        if normalize_source_name(record['generator'])
        == source_spec['normalized_name']
    ]
    if not selected:
        known = sorted(set(record['generator'] for record in records))
        raise ValueError(
            f"source {source_spec['csv_name']!r} was not found; available "
            f'sources: {known}')
    labels = {record['label'] for record in selected}
    if labels != {0, 1}:
        raise ValueError(
            f"source {source_spec['csv_name']!r} must contain both real and "
            'generated samples')
    return {
        'real': np.asarray([
            record['raw_logit'] for record in selected
            if record['label'] == 0
        ], dtype=np.float64),
        'generated': np.asarray([
            record['raw_logit'] for record in selected
            if record['label'] == 1
        ], dtype=np.float64),
    }


def build_protocol_data(
    baseline_records,
    car_records,
    source_specs,
    bins,
):
    sources = []
    for source_spec in source_specs:
        baseline = source_logits(baseline_records, source_spec)
        car = source_logits(car_records, source_spec)
        distributions = [
            baseline['real'],
            baseline['generated'],
            car['real'],
            car['generated'],
        ]
        sources.append({
            **source_spec,
            'baseline': baseline,
            'car': car,
            'bin_edges': np.asarray(
                build_shared_bin_edges(distributions, bins=bins),
                dtype=np.float64,
            ),
            'statistics': {
                'baseline': compute_logit_stats(
                    baseline['real'], baseline['generated']),
                'car': compute_logit_stats(car['real'], car['generated']),
            },
        })
    return {'sources': sources}


def configure_matplotlib():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans',
            'sans-serif',
        ],
        'font.size': 6.3,
        'axes.labelsize': 6.3,
        'axes.titlesize': 7.1,
        'xtick.labelsize': 5.7,
        'ytick.labelsize': 5.7,
        'legend.fontsize': 5.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.65,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def require_matplotlib_panel_alignment(
    figure,
    axes,
    json_out,
    tolerance_pt=1.5,
):
    """Audit the regular 2-by-N grid in physical points before export."""
    figure.canvas.draw()
    width_pt = figure.get_figwidth() * 72.0
    height_pt = figure.get_figheight() * 72.0
    rectangles = []
    for row_index, row_axes in enumerate(axes):
        row = []
        for column_index, axis in enumerate(row_axes):
            position = axis.get_position()
            rectangle = {
                'panel': f'{row_index},{column_index}',
                'left_pt': position.x0 * width_pt,
                'right_pt': position.x1 * width_pt,
                'bottom_pt': position.y0 * height_pt,
                'top_pt': position.y1 * height_pt,
                'width_pt': position.width * width_pt,
                'height_pt': position.height * height_pt,
            }
            row.append(rectangle)
        rectangles.append(row)

    checks = []

    def add_check(name, values):
        deviation = float(max(values) - min(values))
        checks.append({
            'name': name,
            'deviation_pt': deviation,
            'tolerance_pt': tolerance_pt,
            'passed': bool(deviation <= tolerance_pt),
        })

    for row_index, row in enumerate(rectangles):
        add_check(f'row_{row_index}_tops', [item['top_pt'] for item in row])
        add_check(
            f'row_{row_index}_bottoms',
            [item['bottom_pt'] for item in row],
        )
        add_check(
            f'row_{row_index}_widths',
            [item['width_pt'] for item in row],
        )
        if len(row) > 2:
            gutters = [
                row[index + 1]['left_pt'] - row[index]['right_pt']
                for index in range(len(row) - 1)
            ]
            add_check(f'row_{row_index}_gutters', gutters)

    for column_index in range(axes.shape[1]):
        column = [row[column_index] for row in rectangles]
        add_check(
            f'column_{column_index}_lefts',
            [item['left_pt'] for item in column],
        )
        add_check(
            f'column_{column_index}_rights',
            [item['right_pt'] for item in column],
        )
        add_check(
            f'column_{column_index}_heights',
            [item['height_pt'] for item in column],
        )

    vertical_gutters = [
        rectangles[0][column_index]['bottom_pt']
        - rectangles[1][column_index]['top_pt']
        for column_index in range(axes.shape[1])
    ]
    add_check('vertical_gutters', vertical_gutters)
    passed = all(check['passed'] for check in checks)
    report = {
        'schema_version': 1,
        'verdict': 'PASS' if passed else 'FIX BEFORE DELIVERY',
        'tolerance_pt': tolerance_pt,
        'rectangles': rectangles,
        'checks': checks,
    }
    json_path = Path(json_out)
    with json_path.open('w', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write('\n')
    if not passed:
        failures = [check['name'] for check in checks if not check['passed']]
        raise RuntimeError(
            'panel alignment failed before export: ' + ', '.join(failures))
    return report


def plot_histogram(axis, distributions, bin_edges, density_scale):
    for key, label, color in (
        ('real', 'Real', REAL_COLOR),
        ('generated', 'Generated', GENERATED_COLOR),
    ):
        values = distributions[key]
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='stepfilled',
            color=color,
            alpha=0.42,
            edgecolor=color,
            linewidth=0.45,
            label=f'{label} (n={values.size:,})',
        )
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='step',
            color=color,
            linewidth=0.7,
        )
    axis.set_yscale(density_scale)


def plot_ecdf(axis, distributions):
    for key, label, color in (
        ('real', 'Real', REAL_COLOR),
        ('generated', 'Generated', GENERATED_COLOR),
    ):
        values = np.sort(distributions[key])
        cumulative = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            cumulative,
            where='post',
            color=color,
            linewidth=0.9,
            label=f'{label} (n={values.size:,})',
        )


def style_axis(axis, bin_edges):
    axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    axis.grid(axis='y', color='#D8D8D8', linewidth=0.4, alpha=0.55)
    axis.tick_params(length=2.5, pad=1.5)
    axis.legend(
        loc='upper right',
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.2,
        labelspacing=0.15,
    )


def equalize_source_axes(axes, bin_edges, plot_kind, density_scale):
    """Use identical axes for Baseline/CAR of one source only.

    Different sources may have very different density peaks and therefore use
    independent y ranges. This keeps the paired comparison fair without a
    high-density source flattening every other column in the protocol.
    """
    for axis in axes:
        axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    if plot_kind == 'ecdf':
        for axis in axes:
            axis.set_ylim(0.0, 1.01)
        return

    limits = [axis.get_ylim() for axis in axes]
    upper = max(limit[1] for limit in limits)
    if density_scale == 'linear':
        lower = 0.0
    else:
        positive_lowers = [limit[0] for limit in limits if limit[0] > 0]
        lower = min(positive_lowers) if positive_lowers else 1e-6
    for axis in axes:
        axis.set_ylim(lower, upper)


def build_figure(
    gan_data,
    diffusion_data,
    gan_plot_kind,
    diffusion_plot_kind,
    gan_density_scale,
    diffusion_density_scale,
    width,
    height,
):
    configure_matplotlib()
    columns = [
        *[('GAN', source) for source in gan_data['sources']],
        *[('Diffusion', source) for source in diffusion_data['sources']],
    ]
    figure, axes = plt.subplots(
        2,
        len(columns),
        figsize=(width, height),
        squeeze=False,
    )
    protocol_settings = {
        'GAN': {
            'data': gan_data,
            'plot_kind': gan_plot_kind,
            'density_scale': gan_density_scale,
        },
        'Diffusion': {
            'data': diffusion_data,
            'plot_kind': diffusion_plot_kind,
            'density_scale': diffusion_density_scale,
        },
    }

    for column_index, (protocol_name, source) in enumerate(columns):
        settings = protocol_settings[protocol_name]
        bin_edges = source['bin_edges']
        source_axes = []
        for row_index, method_key in enumerate(('baseline', 'car')):
            axis = axes[row_index, column_index]
            if settings['plot_kind'] == 'histogram':
                plot_histogram(
                    axis,
                    source[method_key],
                    bin_edges,
                    settings['density_scale'],
                )
            else:
                plot_ecdf(axis, source[method_key])
            style_axis(axis, bin_edges)
            source_axes.append(axis)
            if row_index == 1:
                axis.set_xlabel('Raw logit')

        equalize_source_axes(
            source_axes,
            bin_edges,
            settings['plot_kind'],
            settings['density_scale'],
        )

        panel_letter = chr(ord('a') + column_index)
        axes[0, column_index].set_title(
            f'({panel_letter}) {source["display_name"]}',
            pad=4.0,
            fontweight='bold',
        )

    gan_ylabel = (
        'Cumulative probability'
        if gan_plot_kind == 'ecdf'
        else ('Density' if gan_density_scale == 'linear'
              else 'Density (log scale)')
    )
    diffusion_ylabel = (
        'Cumulative probability'
        if diffusion_plot_kind == 'ecdf'
        else ('Density' if diffusion_density_scale == 'linear'
              else 'Density (log scale)')
    )
    axes[0, 0].set_ylabel(gan_ylabel)
    axes[1, 0].set_ylabel(gan_ylabel)
    diffusion_start = len(gan_data['sources'])
    if diffusion_ylabel != gan_ylabel:
        axes[0, diffusion_start].set_ylabel(diffusion_ylabel)
        axes[1, diffusion_start].set_ylabel(diffusion_ylabel)

    figure.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.15,
        top=0.79,
        wspace=0.18,
        hspace=0.26,
    )
    figure.canvas.draw()

    top_midpoint = sum(axes[0, 0].get_position().intervaly) / 2.0
    bottom_midpoint = sum(axes[1, 0].get_position().intervaly) / 2.0
    figure.text(
        0.018,
        top_midpoint,
        'C2P-CLIP',
        rotation=90,
        rotation_mode='anchor',
        ha='center',
        va='center',
        fontsize=7.2,
        fontweight='bold',
    )
    figure.text(
        0.018,
        bottom_midpoint,
        'CAR',
        rotation=90,
        rotation_mode='anchor',
        ha='center',
        va='center',
        fontsize=7.2,
        fontweight='bold',
    )

    gan_left = axes[0, 0].get_position().x0
    gan_right = axes[0, diffusion_start - 1].get_position().x1
    diffusion_left = axes[0, diffusion_start].get_position().x0
    diffusion_right = axes[0, -1].get_position().x1
    heading_y = 0.935
    figure.text(
        (gan_left + gan_right) / 2.0,
        heading_y,
        'GAN protocol',
        ha='center',
        va='center',
        fontsize=8.0,
        fontweight='bold',
    )
    figure.text(
        (diffusion_left + diffusion_right) / 2.0,
        heading_y,
        'Diffusion protocol',
        ha='center',
        va='center',
        fontsize=8.0,
        fontweight='bold',
    )
    separator_x = (gan_right + diffusion_left) / 2.0
    figure.add_artist(Line2D(
        [separator_x, separator_x],
        [0.12, 0.965],
        transform=figure.transFigure,
        color='#B8B8B8',
        linewidth=0.65,
        linestyle=(0, (3, 3)),
    ))
    return figure, axes


def summarize_inputs(paths):
    return [
        {
            'path': str(path),
            'sha256': file_sha256(path),
        }
        for path in paths
    ]


def json_safe_stats(stats):
    safe = {}
    for key, value in stats.items():
        safe[key] = float(value) if math.isfinite(value) else None
    return safe


def summarize_protocol(protocol_data):
    return {
        'sources': [
            {
                'csv_name': source['csv_name'],
                'display_name': source['display_name'],
                'samples': {
                    'real': int(source['baseline']['real'].size),
                    'generated': int(source['baseline']['generated'].size),
                },
                'shared_bin_edges': source['bin_edges'].tolist(),
                'statistics': {
                    'baseline': json_safe_stats(
                        source['statistics']['baseline']),
                    'car': json_safe_stats(source['statistics']['car']),
                },
            }
            for source in protocol_data['sources']
        ],
    }


def run(args):
    start_time = time.time()
    gan_baseline_paths, gan_baseline_records = load_prediction_csvs(
        args.gan_baseline_csv)
    gan_car_paths, gan_car_records = load_prediction_csvs(args.gan_car_csv)
    diffusion_baseline_paths, diffusion_baseline_records = load_prediction_csvs(
        args.diffusion_baseline_csv)
    diffusion_car_paths, diffusion_car_records = load_prediction_csvs(
        args.diffusion_car_csv)

    validate_alignment(gan_baseline_records, gan_car_records, 'GAN')
    validate_alignment(
        diffusion_baseline_records,
        diffusion_car_records,
        'Diffusion',
    )
    gan_data = build_protocol_data(
        gan_baseline_records,
        gan_car_records,
        args.gan_sources,
        args.gan_bins if args.gan_bins is not None else args.bins,
    )
    diffusion_data = build_protocol_data(
        diffusion_baseline_records,
        diffusion_car_records,
        args.diffusion_sources,
        (
            args.diffusion_bins
            if args.diffusion_bins is not None
            else args.bins
        ),
    )

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = build_figure(
        gan_data,
        diffusion_data,
        args.gan_plot_kind,
        args.diffusion_plot_kind,
        args.gan_density_scale,
        args.diffusion_density_scale,
        args.width,
        args.height,
    )
    alignment_path = Path(f'{output_prefix}.alignment.json')
    alignment_report = require_matplotlib_panel_alignment(
        figure,
        axes,
        json_out=alignment_path,
        tolerance_pt=1.5,
    )
    outputs = {}
    try:
        if 'svg' in args.formats:
            svg_path = Path(f'{output_prefix}.svg')
            figure.savefig(
                svg_path,
                bbox_inches='tight',
                facecolor='white',
            )
            outputs['svg'] = str(svg_path)
            print(f'Saved SVG: {svg_path}')
        if 'pdf' in args.formats:
            pdf_path = Path(f'{output_prefix}.pdf')
            figure.savefig(
                pdf_path,
                bbox_inches='tight',
                facecolor='white',
            )
            outputs['pdf'] = str(pdf_path)
            print(f'Saved PDF: {pdf_path}')
        if 'png' in args.formats:
            png_path = Path(f'{output_prefix}.png')
            figure.savefig(
                png_path,
                dpi=args.dpi,
                bbox_inches='tight',
                facecolor='white',
            )
            outputs['png'] = str(png_path)
            print(f'Saved PNG: {png_path}')
    finally:
        plt.close(figure)

    summary = {
        'schema_version': 1,
        'figure_contract': {
            'comparison': 'C2P-CLIP baseline versus CAR',
            'quantity': 'class-conditional raw-logit distributions',
            'selection': 'explicit representative sources; no automatic ranking',
            'all_matching_samples_included': True,
            'axis_comparability': (
                'Baseline and CAR share bins and axes within each source; '
                'different sources use independent ranges'
            ),
        },
        'inputs': {
            'gan_baseline': summarize_inputs(gan_baseline_paths),
            'gan_car': summarize_inputs(gan_car_paths),
            'diffusion_baseline': summarize_inputs(diffusion_baseline_paths),
            'diffusion_car': summarize_inputs(diffusion_car_paths),
        },
        'alignment': {
            'identity_fields': ['generator', 'path', 'label'],
            'gan_same_set_and_order': True,
            'diffusion_same_set_and_order': True,
            'panel_geometry_report': str(alignment_path),
            'panel_geometry_verdict': alignment_report['verdict'],
        },
        'plot': {
            'gan_bins': (
                args.gan_bins if args.gan_bins is not None else args.bins
            ),
            'diffusion_bins': (
                args.diffusion_bins
                if args.diffusion_bins is not None
                else args.bins
            ),
            'gan_plot_kind': args.gan_plot_kind,
            'diffusion_plot_kind': args.diffusion_plot_kind,
            'gan_density_scale': args.gan_density_scale,
            'diffusion_density_scale': args.diffusion_density_scale,
            'normalization': 'per-class probability density'
            if args.gan_plot_kind == args.diffusion_plot_kind == 'histogram'
            else 'protocol-specific; see plot kinds',
            'width_inches': args.width,
            'height_inches': args.height,
        },
        'protocols': {
            'gan': summarize_protocol(gan_data),
            'diffusion': summarize_protocol(diffusion_data),
        },
        'outputs': outputs,
        'metadata': {
            'script': str(Path(__file__).resolve()),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'runtime_seconds': time.time() - start_time,
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
