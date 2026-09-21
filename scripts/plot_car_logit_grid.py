"""Plot a C2P-CLIP Figure-5-style baseline-versus-CAR logit grid.

The mixed figure uses Real/Generated rows and overlays C2P-CLIP/CAR within
explicit test-source columns.  The diffusion-only figure uses C2P-CLIP/CAR
rows and plots Real/Generated distributions within each method panel.
Prediction CSVs must contain the fields emitted by the unified binary
evaluator: generator, path, label, raw_logit, and score.

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
from matplotlib.patches import Patch
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)

from utils.logit_distribution import build_shared_bin_edges, compute_logit_stats


REQUIRED_FIELDS = ('generator', 'path', 'label', 'raw_logit', 'score')
SUPPORTED_FORMATS = ('svg', 'pdf', 'png')
KDE_GRID_POINTS = 320
REAL_COLOR = '#6F9FC7'
GENERATED_COLOR = '#E89A55'
BASELINE_COLOR = '#737373'
DEFAULT_GAN_SOURCES = (
    'deepfake=Deepfakes',
    'crn=CRN',
)
DEFAULT_DIFFUSION_SOURCES = (
    'adm=ADM',
    'vqdm=VQDM',
)
DEFAULT_DIFFUSION_ONLY_SOURCES = (
    'adm=ADM',
    'glide=GLIDE',
    'sdv5=SDv5',
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
        '--layout',
        choices=('mixed', 'diffusion-only'),
        default='mixed',
        help=(
            'mixed plots GAN and diffusion sources; diffusion-only plots '
            'four diffusion sources and does not require GAN CSVs'
        ),
    )
    parser.add_argument(
        '--gan_baseline_csv',
        nargs='+',
        help='one or more GAN-protocol baseline prediction CSVs',
    )
    parser.add_argument(
        '--gan_car_csv',
        nargs='+',
        help='one or more GAN-protocol CAR prediction CSVs',
    )
    parser.add_argument(
        '--diffusion_baseline_csv',
        nargs='+',
        help='one or more diffusion-protocol baseline prediction CSVs',
    )
    parser.add_argument(
        '--diffusion_car_csv',
        nargs='+',
        help='one or more diffusion-protocol CAR prediction CSVs',
    )
    parser.add_argument(
        '--gan_sources',
        nargs='+',
        default=None,
        metavar='CSV_NAME=DISPLAY_NAME',
        help=(
            'fixed GAN source columns; default: '
            'deepfake=Deepfakes crn=CRN'
        ),
    )
    parser.add_argument(
        '--diffusion_sources',
        nargs='+',
        default=None,
        metavar='CSV_NAME=DISPLAY_NAME',
        help=(
            'fixed diffusion source columns; mixed default: adm=ADM '
            'vqdm=VQDM; diffusion-only default: adm=ADM glide=GLIDE '
            'sdv5=SDv5 vqdm=VQDM'
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
        choices=('histogram', 'kde', 'ecdf'),
        default=None,
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
    if args.diffusion_baseline_csv is None or args.diffusion_car_csv is None:
        parser.error(
            '--diffusion_baseline_csv and --diffusion_car_csv are required')
    if args.layout == 'mixed':
        if args.gan_baseline_csv is None or args.gan_car_csv is None:
            parser.error(
                '--gan_baseline_csv and --gan_car_csv are required for '
                '--layout mixed')
    elif args.gan_baseline_csv is not None or args.gan_car_csv is not None:
        parser.error(
            'GAN CSV arguments are not used with --layout diffusion-only')
    if args.diffusion_plot_kind is None:
        args.diffusion_plot_kind = (
            'kde' if args.layout == 'diffusion-only' else 'histogram'
        )
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

    gan_source_values = (
        args.gan_sources
        if args.gan_sources is not None
        else list(DEFAULT_GAN_SOURCES)
    )
    diffusion_source_values = args.diffusion_sources
    if diffusion_source_values is None:
        diffusion_source_values = list(
            DEFAULT_DIFFUSION_ONLY_SOURCES
            if args.layout == 'diffusion-only'
            else DEFAULT_DIFFUSION_SOURCES
        )
    args.gan_sources = parse_source_specs(gan_source_values)
    args.diffusion_sources = parse_source_specs(diffusion_source_values)
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
        'font.size': 6.6,
        'axes.labelsize': 6.8,
        'axes.titlesize': 7.6,
        'xtick.labelsize': 6.2,
        'ytick.labelsize': 7.2,
        'legend.fontsize': 6.0,
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


def plot_histogram_comparison(
    axis,
    source,
    class_key,
    bin_edges,
    density_scale,
):
    class_color = REAL_COLOR if class_key == 'real' else GENERATED_COLOR
    baseline_values = source['baseline'][class_key]
    car_values = source['car'][class_key]

    axis.hist(
        car_values,
        bins=bin_edges,
        density=True,
        histtype='stepfilled',
        color=class_color,
        alpha=0.22,
        edgecolor='none',
    )
    axis.hist(
        baseline_values,
        bins=bin_edges,
        density=True,
        histtype='step',
        color=BASELINE_COLOR,
        linewidth=0.95,
        linestyle=(0, (3.0, 2.0)),
    )
    axis.hist(
        car_values,
        bins=bin_edges,
        density=True,
        histtype='step',
        color=class_color,
        linewidth=1.15,
    )
    axis.set_yscale(density_scale)
    if density_scale == 'log':
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def plot_ecdf_comparison(axis, source, class_key):
    class_color = REAL_COLOR if class_key == 'real' else GENERATED_COLOR
    for method_key, color, linestyle, linewidth in (
        ('baseline', BASELINE_COLOR, (0, (3.0, 2.0)), 0.95),
        ('car', class_color, 'solid', 1.15),
    ):
        values = np.sort(source[method_key][class_key])
        cumulative = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            cumulative,
            where='post',
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def plot_method_histograms(
    axis,
    source,
    method_key,
    bin_edges,
    density_scale,
):
    """Plot Real and Generated distributions for one method row."""
    for class_key, class_color in (
        ('real', REAL_COLOR),
        ('generated', GENERATED_COLOR),
    ):
        values = source[method_key][class_key]
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='stepfilled',
            color=class_color,
            alpha=0.24,
            edgecolor='none',
        )
        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype='step',
            color=class_color,
            linewidth=1.05,
        )
    axis.set_yscale(density_scale)
    if density_scale == 'log':
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def plot_method_ecdfs(axis, source, method_key):
    """Plot Real and Generated ECDFs for one method row."""
    for class_key, class_color in (
        ('real', REAL_COLOR),
        ('generated', GENERATED_COLOR),
    ):
        values = np.sort(source[method_key][class_key])
        cumulative = np.arange(1, values.size + 1) / values.size
        axis.step(
            values,
            cumulative,
            where='post',
            color=class_color,
            linewidth=1.05,
        )


def gaussian_kde_curve(values, x_grid):
    """Return a deterministic Gaussian KDE using a robust Silverman rule."""
    values = np.asarray(values, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('KDE values must be a non-empty one-dimensional array')
    if x_grid.ndim != 1 or x_grid.size < 2:
        raise ValueError('KDE grid must contain at least two points')

    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    first_quartile, third_quartile = np.percentile(values, [25.0, 75.0])
    robust_scale = float((third_quartile - first_quartile) / 1.34)
    positive_scales = [
        scale
        for scale in (standard_deviation, robust_scale)
        if math.isfinite(scale) and scale > 0.0
    ]
    scale = min(positive_scales) if positive_scales else 0.0
    bandwidth = 0.9 * scale * values.size ** (-0.2) if scale > 0.0 else 0.0
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        grid_span = float(x_grid[-1] - x_grid[0])
        bandwidth = max(grid_span / 100.0, np.finfo(np.float64).eps)

    density = np.zeros_like(x_grid, dtype=np.float64)
    chunk_size = 2048
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        standardized = (x_grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * standardized * standardized).sum(axis=1)
    density /= values.size * bandwidth * math.sqrt(2.0 * math.pi)
    return density, bandwidth


def plot_method_kdes(axis, source, method_key, bin_edges, density_scale):
    """Plot smooth Real/Fake density curves without dropping observations."""
    x_grid = np.linspace(
        float(bin_edges[0]),
        float(bin_edges[-1]),
        KDE_GRID_POINTS,
    )
    for class_key, class_color in (
        ('real', REAL_COLOR),
        ('generated', GENERATED_COLOR),
    ):
        density, _ = gaussian_kde_curve(source[method_key][class_key], x_grid)
        if density_scale == 'linear':
            axis.fill_between(
                x_grid,
                0.0,
                density,
                color=class_color,
                alpha=0.22,
                linewidth=0.0,
            )
        axis.plot(x_grid, density, color=class_color, linewidth=1.15)
    axis.set_yscale(density_scale)
    if density_scale == 'log':
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def add_class_legend(axis):
    handles = [
        Patch(facecolor=REAL_COLOR, edgecolor='none', alpha=0.7, label='Real'),
        Patch(
            facecolor=GENERATED_COLOR,
            edgecolor='none',
            alpha=0.7,
            label='Fake',
        ),
    ]
    axis.legend(
        handles=handles,
        loc='upper left',
        ncol=2,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.2,
        labelspacing=0.2,
        fontsize=5.8,
    )


def style_axis(axis, bin_edges):
    axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    axis.grid(axis='y', color='#D8D8D8', linewidth=0.4, alpha=0.55)
    axis.tick_params(length=0.0, pad=4.0)
    axis.xaxis.labelpad = 6.0
    axis.yaxis.labelpad = 7.0


def align_source_class_axes(axes, bin_edges, plot_kind):
    """Use one raw-logit range for both class rows of a source.

    Baseline and CAR are overlaid within each axis. Histogram y ranges remain
    class-specific so an extreme density peak in one class cannot flatten the
    paired method comparison in the other class.
    """
    for axis in axes:
        axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    if plot_kind == 'ecdf':
        for axis in axes:
            axis.set_ylim(0.0, 1.01)
        return


def align_source_method_axes(
    axes,
    bin_edges,
    plot_kind,
    density_scale,
):
    """Share both x and y ranges across C2P-CLIP/CAR method rows."""
    for axis in axes:
        axis.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    if plot_kind == 'ecdf':
        for axis in axes:
            axis.set_ylim(0.0, 1.01)
        return

    y_limits = [axis.get_ylim() for axis in axes]
    if density_scale == 'log':
        shared_bottom = min(limit[0] for limit in y_limits)
        shared_top = max(limit[1] for limit in y_limits) * 1.8
    else:
        shared_bottom = 0.0
        shared_top = max(limit[1] for limit in y_limits) * 1.28
    for axis in axes:
        axis.set_ylim(shared_bottom, shared_top)


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
    columns = []
    if gan_data is not None:
        columns.extend(('GAN', source) for source in gan_data['sources'])
    columns.extend(
        ('Diffusion', source) for source in diffusion_data['sources'])
    figure, axes = plt.subplots(
        2,
        len(columns),
        figsize=(width, height),
        squeeze=False,
    )
    protocol_settings = {
        'Diffusion': {
            'data': diffusion_data,
            'plot_kind': diffusion_plot_kind,
            'density_scale': diffusion_density_scale,
        },
    }
    if gan_data is not None:
        protocol_settings['GAN'] = {
            'data': gan_data,
            'plot_kind': gan_plot_kind,
            'density_scale': gan_density_scale,
        }

    method_row_layout = gan_data is None
    for column_index, (protocol_name, source) in enumerate(columns):
        settings = protocol_settings[protocol_name]
        bin_edges = source['bin_edges']
        source_axes = []
        if method_row_layout:
            for row_index, method_key in enumerate(('baseline', 'car')):
                axis = axes[row_index, column_index]
                if settings['plot_kind'] == 'histogram':
                    plot_method_histograms(
                        axis,
                        source,
                        method_key,
                        bin_edges,
                        settings['density_scale'],
                    )
                elif settings['plot_kind'] == 'kde':
                    plot_method_kdes(
                        axis,
                        source,
                        method_key,
                        bin_edges,
                        settings['density_scale'],
                    )
                else:
                    plot_method_ecdfs(axis, source, method_key)
                style_axis(axis, bin_edges)
                add_class_legend(axis)
                source_axes.append(axis)
            align_source_method_axes(
                source_axes,
                bin_edges,
                settings['plot_kind'],
                settings['density_scale'],
            )
        else:
            for row_index, class_key in enumerate(('real', 'generated')):
                axis = axes[row_index, column_index]
                if settings['plot_kind'] == 'histogram':
                    plot_histogram_comparison(
                        axis,
                        source,
                        class_key,
                        bin_edges,
                        settings['density_scale'],
                    )
                else:
                    plot_ecdf_comparison(axis, source, class_key)
                style_axis(axis, bin_edges)
                source_axes.append(axis)
            align_source_class_axes(
                source_axes,
                bin_edges,
                settings['plot_kind'],
            )

        panel_letter = chr(ord('a') + column_index)
        if not method_row_layout:
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
    if gan_data is None:
        diffusion_start = 0
    else:
        axes[0, 0].set_ylabel(gan_ylabel)
        axes[1, 0].set_ylabel(gan_ylabel)
        diffusion_start = len(gan_data['sources'])
        if diffusion_ylabel != gan_ylabel:
            axes[0, diffusion_start].set_ylabel(diffusion_ylabel)
            axes[1, diffusion_start].set_ylabel(diffusion_ylabel)

    figure.subplots_adjust(
        left=0.105 if method_row_layout else 0.11,
        right=0.992,
        bottom=0.22 if method_row_layout else 0.17,
        top=0.95 if method_row_layout else 0.79,
        wspace=0.28,
        hspace=0.28 if method_row_layout else 0.26,
    )
    figure.canvas.draw()

    plot_left = axes[1, 0].get_position().x0
    plot_right = axes[1, -1].get_position().x1
    figure.text(
        (plot_left + plot_right) / 2.0,
        0.035 if method_row_layout else 0.055,
        'Raw logit',
        ha='center',
        va='center',
        fontsize=6.8,
    )

    if method_row_layout:
        row_specs = (
            ('C2P-CLIP', '#222222'),
            ('CAR', '#222222'),
        )
        legend_handles = []
    else:
        row_specs = (
            ('Real', REAL_COLOR),
            ('Fake', GENERATED_COLOR),
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=BASELINE_COLOR,
                linewidth=1.0,
                linestyle=(0, (3.0, 2.0)),
                label='C2P-CLIP',
            ),
            Line2D(
                [0],
                [0],
                color='#222222',
                linewidth=1.2,
                label='CAR',
            ),
        ]
    for row_index, (row_label, row_color) in enumerate(row_specs):
        row_axis_position = axes[row_index, 0].get_position()
        figure.text(
            0.018,
            (
                (row_axis_position.y0 + row_axis_position.y1) / 2.0
                if method_row_layout
                else row_axis_position.y1 + 0.012
            ),
            row_label,
            ha='center' if method_row_layout else 'left',
            va='center' if method_row_layout else 'bottom',
            fontsize=7.8,
            fontweight='bold',
            color=row_color,
            rotation=90 if method_row_layout else 0,
            rotation_mode='anchor',
        )

    if not method_row_layout:
        figure.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.875),
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )

    if method_row_layout:
        for column_index, source in enumerate(diffusion_data['sources']):
            axis_position = axes[1, column_index].get_position()
            panel_letter = chr(ord('a') + column_index)
            figure.text(
                (axis_position.x0 + axis_position.x1) / 2.0,
                0.105,
                f'({panel_letter}) {source["display_name"]}',
                ha='center',
                va='center',
                fontsize=7.6,
                fontweight='bold',
            )

    diffusion_left = axes[0, diffusion_start].get_position().x0
    diffusion_right = axes[0, -1].get_position().x1
    heading_y = 0.955 if method_row_layout else 0.935
    if not method_row_layout:
        figure.text(
            (diffusion_left + diffusion_right) / 2.0,
            heading_y,
            'Diffusion protocol',
            ha='center',
            va='center',
            fontsize=8.6,
            fontweight='bold',
        )
    if gan_data is not None:
        gan_left = axes[0, 0].get_position().x0
        gan_right = axes[0, diffusion_start - 1].get_position().x1
        figure.text(
            (gan_left + gan_right) / 2.0,
            heading_y,
            'GAN protocol',
            ha='center',
            va='center',
            fontsize=8.6,
            fontweight='bold',
        )
        separator_x = (gan_right + diffusion_left) / 2.0
        figure.add_artist(Line2D(
            [separator_x, separator_x],
            [0.12, 0.80],
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
    gan_baseline_paths = []
    gan_car_paths = []
    gan_data = None
    if args.layout == 'mixed':
        gan_baseline_paths, gan_baseline_records = load_prediction_csvs(
            args.gan_baseline_csv)
        gan_car_paths, gan_car_records = load_prediction_csvs(args.gan_car_csv)
        validate_alignment(gan_baseline_records, gan_car_records, 'GAN')
        gan_data = build_protocol_data(
            gan_baseline_records,
            gan_car_records,
            args.gan_sources,
            args.gan_bins if args.gan_bins is not None else args.bins,
        )
    diffusion_baseline_paths, diffusion_baseline_records = load_prediction_csvs(
        args.diffusion_baseline_csv)
    diffusion_car_paths, diffusion_car_records = load_prediction_csvs(
        args.diffusion_car_csv)

    validate_alignment(
        diffusion_baseline_records,
        diffusion_car_records,
        'Diffusion',
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
                'C2P-CLIP and CAR method rows use identical bins and shared '
                'x/y axes within each diffusion source'
                if args.layout == 'diffusion-only'
                else (
                    'Baseline and CAR are overlaid with identical bins and '
                    'axes within each source and class; class rows share the '
                    'source raw-logit range'
                )
            ),
            'layout': (
                'rows encode C2P-CLIP/CAR methods; color encodes '
                'Real/Generated classes'
                if args.layout == 'diffusion-only'
                else (
                    'rows encode Real/Generated classes; line style encodes '
                    'C2P-CLIP/CAR methods'
                )
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
            'gan_same_set_and_order': (
                True if args.layout == 'mixed' else None
            ),
            'diffusion_same_set_and_order': True,
            'panel_geometry_report': str(alignment_path),
            'panel_geometry_verdict': alignment_report['verdict'],
        },
        'plot': {
            'layout': args.layout,
            'gan_bins': (
                (args.gan_bins if args.gan_bins is not None else args.bins)
                if args.layout == 'mixed'
                else None
            ),
            'diffusion_bins': (
                args.diffusion_bins
                if args.diffusion_bins is not None
                else args.bins
            ),
            'gan_plot_kind': (
                args.gan_plot_kind if args.layout == 'mixed' else None
            ),
            'diffusion_plot_kind': args.diffusion_plot_kind,
            'gan_density_scale': (
                args.gan_density_scale if args.layout == 'mixed' else None
            ),
            'diffusion_density_scale': args.diffusion_density_scale,
            'normalization': (
                'per-class Gaussian kernel density estimate using a robust '
                'Silverman bandwidth and every observation'
                if args.diffusion_plot_kind == 'kde'
                else (
                    'per-class probability density'
                    if (
                        args.diffusion_plot_kind == 'histogram'
                        and (
                            args.layout == 'diffusion-only'
                            or args.gan_plot_kind == 'histogram'
                        )
                    )
                    else 'protocol-specific; see plot kinds'
                )
            ),
            'kde_grid_points': (
                KDE_GRID_POINTS if args.diffusion_plot_kind == 'kde' else None
            ),
            'diffusion_only_protocol_heading': (
                False if args.layout == 'diffusion-only' else None
            ),
            'diffusion_only_density_axis_title': (
                False if args.layout == 'diffusion-only' else None
            ),
            'diffusion_only_column_labels': (
                'panel letter and source name below each column'
                if args.layout == 'diffusion-only'
                else None
            ),
            'width_inches': args.width,
            'height_inches': args.height,
        },
        'protocols': {
            'gan': (
                summarize_protocol(gan_data)
                if gan_data is not None
                else None
            ),
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
