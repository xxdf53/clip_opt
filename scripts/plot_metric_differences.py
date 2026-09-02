"""Plot CAR-minus-baseline metric differences from registered results."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLOR = '#8d969d'
CAR_COLOR = '#4f8a78'
NEGATIVE_COLOR = '#b96868'
CONNECTOR_COLOR = '#c5c9c8'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Plot registered CAR-minus-baseline metric differences')
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--baseline', nargs='+', type=float, required=True)
    parser.add_argument('--car', nargs='+', type=float, required=True)
    parser.add_argument('--protocol_label', required=True)
    parser.add_argument('--output_prefix', required=True)
    parser.add_argument('--baseline_label', default='C2P-CLIP')
    parser.add_argument('--car_label', default='C2P-CLIP + CAR')
    args = parser.parse_args(argv)

    lengths = {len(args.metrics), len(args.baseline), len(args.car)}
    if len(lengths) != 1:
        parser.error('--metrics, --baseline, and --car must have equal lengths')
    if not args.metrics:
        parser.error('at least one metric is required')
    if len(set(args.metrics)) != len(args.metrics):
        parser.error('--metrics cannot contain duplicates')
    if any(not math.isfinite(value) for value in args.baseline + args.car):
        parser.error('--baseline and --car values must be finite')
    return args


def configure_matplotlib():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 7.0,
        'axes.labelsize': 8.0,
        'axes.titlesize': 9.0,
        'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.linewidth': 0.7,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def build_figure(
    metrics,
    baseline,
    car,
    protocol_label,
    baseline_label,
    car_label,
):
    configure_matplotlib()
    differences = car - baseline
    positions = np.arange(len(metrics), dtype=np.float64)

    height = max(3.0, 0.82 * len(metrics) + 1.35)
    figure, axis = plt.subplots(figsize=(7.1, height))
    left = np.minimum(baseline, car)
    right = np.maximum(baseline, car)
    axis.hlines(
        positions,
        left,
        right,
        color=CONNECTOR_COLOR,
        linewidth=2.4,
        zorder=1,
    )
    axis.scatter(
        baseline,
        positions,
        s=54,
        color=BASELINE_COLOR,
        edgecolor='white',
        linewidth=0.7,
        label=baseline_label,
        zorder=3,
    )
    axis.scatter(
        car,
        positions,
        s=54,
        color=CAR_COLOR,
        edgecolor='white',
        linewidth=0.7,
        label=car_label,
        zorder=4,
    )
    axis.set_yticks(positions, labels=metrics)
    axis.invert_yaxis()
    axis.set_xlabel('Score (%)')
    figure.suptitle(protocol_label, fontsize=9.0, y=0.965)
    axis.grid(axis='x', color='#dddddd', linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0)
    axis.legend(
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    score_min = float(min(np.min(baseline), np.min(car)))
    score_max = float(max(np.max(baseline), np.max(car)))
    score_range = max(score_max - score_min, 1.0)
    padding = max(score_range * 0.12, 0.8)
    x_min = score_min - padding
    x_max = score_max + padding
    axis.set_xlim(x_min, x_max)
    for index, (base_value, car_value, difference) in enumerate(zip(
        baseline,
        car,
        differences,
    )):
        sign = '+' if difference >= 0.0 else ''
        axis.text(
            base_value,
            index + 0.19,
            f'{base_value:.2f}',
            ha='center',
            va='top',
            fontsize=6.8,
            color='#60686e',
        )
        axis.text(
            car_value,
            index - 0.19,
            f'{car_value:.2f}',
            ha='center',
            va='bottom',
            fontsize=6.8,
            color='#356b5c',
        )
        delta_color = CAR_COLOR if difference >= 0.0 else NEGATIVE_COLOR
        axis.text(
            1.025,
            index,
            f'{sign}{difference:.2f} pp',
            transform=axis.get_yaxis_transform(),
            ha='left',
            va='center',
            fontsize=7.2,
            fontweight='bold',
            color=delta_color,
            clip_on=False,
        )

    axis.text(
        1.025,
        1.075,
        'Difference',
        transform=axis.transAxes,
        ha='left',
        va='bottom',
        fontsize=7.0,
        color='#555555',
        clip_on=False,
    )
    axis.set_ylim(len(metrics) - 0.45, -0.55)
    figure.subplots_adjust(left=0.18, right=0.84, bottom=0.18, top=0.73)
    return figure, differences


def run(args):
    baseline = np.asarray(args.baseline, dtype=np.float64)
    car = np.asarray(args.car, dtype=np.float64)
    figure, differences = build_figure(
        args.metrics,
        baseline,
        car,
        args.protocol_label,
        args.baseline_label,
        args.car_label,
    )

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(f'{output_prefix}.pdf')
    try:
        figure.savefig(
            output_path,
            bbox_inches='tight',
            facecolor='white',
        )
    finally:
        plt.close(figure)
    print(f'Saved PDF: {output_path}')

    summary = {
        'protocol_label': args.protocol_label,
        'baseline_label': args.baseline_label,
        'car_label': args.car_label,
        'output_pdf': str(output_path),
        'metrics': [
            {
                'name': name,
                'baseline': float(base_value),
                'car': float(car_value),
                'difference_pp': float(difference),
            }
            for name, base_value, car_value, difference in zip(
                args.metrics,
                baseline,
                car,
                differences,
            )
        ],
    }
    for metric in summary['metrics']:
        print(
            f"{metric['name']}: {metric['baseline']:.2f} -> "
            f"{metric['car']:.2f} "
            f"({metric['difference_pp']:+.2f} pp)")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == '__main__':
    main()
