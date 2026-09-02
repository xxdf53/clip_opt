"""Plot CAR-minus-baseline metric differences from registered results."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


POSITIVE_COLOR = '#4f8a78'
NEGATIVE_COLOR = '#b96868'
ZERO_COLOR = '#777777'


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


def difference_limits(differences):
    largest = max(float(np.max(np.abs(differences))), 0.5)
    padding = max(largest * 0.24, 0.35)
    return -largest - padding, largest + padding


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
    colors = [
        POSITIVE_COLOR if difference >= 0.0 else NEGATIVE_COLOR
        for difference in differences
    ]

    height = max(2.4, 0.52 * len(metrics) + 1.15)
    figure, axis = plt.subplots(figsize=(7.1, height))
    axis.barh(
        positions,
        differences,
        height=0.56,
        color=colors,
        edgecolor='white',
        linewidth=0.6,
    )
    axis.axvline(0.0, color=ZERO_COLOR, linewidth=0.8)
    axis.set_yticks(positions, labels=metrics)
    axis.invert_yaxis()
    axis.set_xlabel(f'{car_label} - {baseline_label} (percentage points)')
    axis.set_title(protocol_label, pad=9)
    axis.grid(axis='x', color='#dddddd', linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0)

    x_min, x_max = difference_limits(differences)
    axis.set_xlim(x_min, x_max)
    text_offset = (x_max - x_min) * 0.018
    for index, (base_value, car_value, difference) in enumerate(zip(
        baseline,
        car,
        differences,
    )):
        sign = '+' if difference >= 0.0 else ''
        text = (
            f'{sign}{difference:.2f} pp  '
            f'({base_value:.2f} -> {car_value:.2f})')
        if difference >= 0.0:
            x_position = difference + text_offset
            horizontal_alignment = 'left'
        else:
            x_position = difference - text_offset
            horizontal_alignment = 'right'
        axis.text(
            x_position,
            index,
            text,
            ha=horizontal_alignment,
            va='center',
            fontsize=7.0,
            color='#333333',
        )

    figure.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.86)
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
