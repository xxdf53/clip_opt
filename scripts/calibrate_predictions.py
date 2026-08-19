"""Fit validation-only calibration and apply frozen parameters to logits."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.binary_calibration import (
    calibrated_records,
    fit_calibration,
    load_calibration,
    load_prediction_csv,
    save_calibration,
    summarize_calibrated_records,
    write_calibrated_csv,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Fit threshold/temperature on validation logits or apply frozen '
            'parameters to a prediction CSV'))
    subparsers = parser.add_subparsers(dest='command', required=True)

    fit_parser = subparsers.add_parser(
        'fit', help='fit using labels from an independent validation CSV')
    fit_parser.add_argument('--validation_csv', required=True)
    fit_parser.add_argument('--output_json', required=True)
    fit_parser.add_argument('--calibrated_csv')
    fit_parser.add_argument('--min_temperature', type=float, default=0.01)
    fit_parser.add_argument('--max_temperature', type=float, default=100.0)

    apply_parser = subparsers.add_parser(
        'apply', help='apply frozen parameters without fitting')
    apply_parser.add_argument('--predictions_csv', required=True)
    apply_parser.add_argument('--calibration_json', required=True)
    apply_parser.add_argument('--output_csv', required=True)
    return parser


def format_metrics(name, metrics):
    return (
        f'{name:20s} n={metrics["n"]:>7d}  '
        f'ACC={metrics["acc"]:6.2f}%  '
        f'Real_ACC={metrics["real_acc"]:6.2f}%  '
        f'Fake_ACC={metrics["fake_acc"]:6.2f}%  '
        f'AP={metrics["ap"]:6.2f}%  '
        f'AUROC={metrics["roc_auc"]:6.2f}%  '
        f'ECE={metrics["ece"]:6.2f}%  '
        f'Brier={metrics["brier"]:.4f}'
    )


def print_summary(summary):
    for generator, metrics in summary['group_metrics'].items():
        print(format_metrics(generator, metrics))
    print('-' * 120)
    print(format_metrics('Macro mean', summary['macro_metrics']))
    print(format_metrics('Overall', summary['overall_metrics']))


def run_fit(args):
    validation_path, records = load_prediction_csv(args.validation_csv)
    parameters = fit_calibration(
        records,
        validation_csv=validation_path,
        temperature_bounds=(args.min_temperature, args.max_temperature),
    )
    output_json = save_calibration(parameters, args.output_json)
    print(f'Validation CSV: {validation_path}')
    print(f'Validation SHA256: {parameters["validation_csv_sha256"]}')
    print(f'Threshold tau: {parameters["threshold"]:.12g}')
    print(f'Temperature T: {parameters["temperature"]:.12g}')
    print(format_metrics(
        'Validation raw', parameters['validation_metrics_raw']))
    print(format_metrics(
        'Validation calibrated',
        parameters['validation_metrics_calibrated']))
    print(f'Calibration JSON: {output_json}')

    if args.calibrated_csv:
        output_records = calibrated_records(records, parameters)
        output_csv = write_calibrated_csv(
            output_records, args.calibrated_csv)
        print(f'Calibrated validation CSV: {output_csv}')


def run_apply(args):
    predictions_path, records = load_prediction_csv(args.predictions_csv)
    calibration_path, parameters = load_calibration(args.calibration_json)
    output_records = calibrated_records(records, parameters)
    output_path = write_calibrated_csv(output_records, args.output_csv)
    summary = summarize_calibrated_records(output_records)
    print(f'Predictions CSV: {predictions_path}')
    print(f'Calibration JSON: {calibration_path}')
    print(f'Threshold tau: {float(parameters["threshold"]):.12g}')
    print(f'Temperature T: {float(parameters["temperature"]):.12g}')
    print_summary(summary)
    print(f'Calibrated predictions: {output_path}')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'fit':
        if args.min_temperature <= 0:
            parser.error('--min_temperature must be positive')
        if args.max_temperature <= args.min_temperature:
            parser.error(
                '--max_temperature must be greater than --min_temperature')
        run_fit(args)
    else:
        run_apply(args)


if __name__ == '__main__':
    main()
