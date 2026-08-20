# D20 HFR Training Command Handoff

Task ID: D20
Objective: Restore the original global HFR objective, pre-register its paired protocol, and record the first fixed-manifest pilot.
Status: needs review

Inputs read: `AGENTS.md`, `PROJECT_CHARTER.md`, `CURRENT_STATE.md`, `TASK_BOARD.md`, `EXPERIMENT_PROTOCOL.md`, `EXPERIMENT_REGISTRY.csv`, and `HANDOFF_TEMPLATE.md`.
Files changed: `utils/training_objectives.py`, `networks/trainer.py`, `options/base_options.py`, `utils/retired_training.py`, `scripts/train.py`, HFR tests, and linked project records.
Experiment IDs created or updated: EXP-D20-006 through EXP-D20-011 (three matched baseline/HFR seed pairs)
Code revision: `19447ce92310f9b766d19b622e440151ef1f11ac`.
Commands or entry points used: `scripts/train.py`; the complete command is recorded in `EXPERIMENT_PROTOCOL.md`.
Data split and seed: `./sdv1.4`; fixed 12,800-sample manifest with `data_seed=271828` and SHA-256 `0e31e71b5ac10ced50e830c1021fa186554be2f1e851d0c5183099ea43e36d5a`; model seeds 123, 42, and 2024.
Artifacts and result locations: baseline checkpoint `/home/ac/data/xxxf/clip_opt_github/c2p_checkpoints/baseline_manifest_ds271828__20260819-092121__ds271828__ms123__r6a6d0.5__lr0.0002__c4.0/model_epoch_0_total_steps_200_testacc_99.3333.pth` with predictions `/home/ac/data/xxxf/clip_opt_github/baseline_manifest_ds271828_s123_predictions.csv`; HFR checkpoint `/home/ac/data/xxxf/clip_opt_github/c2p_checkpoints/hfr_manifest_ds271828__20260819-092529__ds271828__ms123__r6a6d0.5__lr0.0002__c4.0__hfr-w1.0-q0.25/model_epoch_0_total_steps_200_testacc_96.9833.pth` with predictions `/home/ac/data/xxxf/clip_opt_github/hfr_manifest_ds271828_s123_predictions.csv`.

Verified result: screenshot-reported matched seed-123 pilot on seven groups / 88,000 images. Baseline to HFR: Macro ACC 90.08% to 95.95%, Macro AP 98.90% to 99.18%, AUROC 99.16% to 99.37%, ECE 41.30% to 38.04%, and Brier 0.2122 to 0.2093. Full runtime unit tests previously passed in the `c2pclip` environment.

Per-generator screenshot report:

| Generator | Baseline ACC | HFR ACC | Delta ACC | Baseline AP | HFR AP | Baseline AUROC | HFR AUROC | Baseline ECE | HFR ECE | Baseline Brier | HFR Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adm | 72.35% | 92.22% | +19.87 pp | 96.38% | 97.23% | 97.14% | 97.77% | 38.55% | 28.66% | 0.2228 | 0.2191 |
| biggan | 83.02% | 96.58% | +13.56 pp | 98.39% | 98.78% | 98.93% | 99.16% | 42.62% | 30.28% | 0.2187 | 0.2160 |
| glide | 94.05% | 96.65% | +2.60 pp | 99.46% | 99.30% | 99.49% | 99.62% | 42.20% | 41.18% | 0.2129 | 0.2104 |
| midjourney | 94.81% | 96.37% | +1.56 pp | 99.48% | 99.56% | 99.53% | 99.60% | 41.50% | 40.39% | 0.2092 | 0.2074 |
| sdv5 | 99.09% | 96.70% | -2.39 pp | 99.94% | 99.97% | 99.95% | 99.97% | 41.06% | 44.00% | 0.2026 | 0.2003 |
| vqdm | 89.53% | 96.35% | +6.82 pp | 99.14% | 99.52% | 99.26% | 99.57% | 41.65% | 38.23% | 0.2129 | 0.2088 |
| wukong | 97.70% | 96.78% | -0.92 pp | 99.81% | 99.91% | 99.83% | 99.92% | 41.53% | 43.52% | 0.2060 | 0.2028 |

Interpretation: HFR improves ranking modestly and fixed-threshold detection strongly, especially fake recall (81.22% to 98.32%). The tradeoff is lower Real ACC (98.94% to 93.58%), consistent with a decision-boundary shift toward detecting fakes. `adm` remains the weakest group, but its ACC improves from 72.35% to 92.22%. The restored objective adds the normalized BCE of the globally lowest-logit fake fraction to the original classification loss.
Decision and rationale: The seed-123 pair began as a pilot; the later seed-42 and seed-2024 pairs now show consistent ACC/AP/AUROC/Brier gains. Keep the complete three-seed result as candidate evidence pending provenance and artifact verification. Later semantic-coverage, compensation, and bias-neutral variants remain retired.
Risks or unresolved issues: Raw training and evaluation logs are not yet archived in the project record. The checkpoint filenames include a training-side `testacc` value, but the split used for that value is not documented in the supplied artifacts; no training choice may be made from `diffusion_test_only` results.
## Three-Seed Summary

| Seed | Baseline ACC | HFR ACC | Delta ACC | Baseline AP | HFR AP | Delta AP | Delta AUROC | Delta ECE | Delta Brier | Delta Real ACC | Delta Fake ACC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 90.08% | 95.95% | +5.87 pp | 98.90% | 99.18% | +0.28 pp | +0.21 pp | -3.26 pp | -0.0029 | -5.36 pp | +17.10 pp |
| 42 | 87.65% | 95.76% | +8.11 pp | 98.82% | 99.12% | +0.30 pp | +0.21 pp | -0.86 pp | -0.0030 | -3.01 pp | +19.22 pp |
| 2024 | 87.22% | 94.92% | +7.70 pp | 99.33% | 99.47% | +0.14 pp | +0.11 pp | +0.93 pp | -0.0028 | -0.88 pp | +16.28 pp |
| Mean | 88.32% | 95.54% | +7.23 pp | 99.02% | 99.26% | +0.24 pp | +0.18 pp | -1.06 pp | -0.0029 | -3.08 pp | +17.53 pp |

Updated interpretation: HFR has a consistent three-seed effect on raw ranking and threshold metrics, with Macro ACC gains of 5.87-8.11 pp and AP gains of 0.14-0.30 pp. The dominant effect is recovering fake accuracy, while real accuracy falls in every seed. The completed validation-fixed analysis retains the AP/AUROC gain but does not retain an ACC, ECE, or Brier advantage. Raw artifact preservation, ablations, and fresh cross-domain evaluation remain before a paper claim.

Updated required next task: Preserve all prediction CSVs and training/evaluation logs, then run HFR-specific ablations and fresh cross-domain evaluation before promotion to T40.

## Calibration Tooling

Implementation: `scripts/calibrate_predictions.py` and
`utils/binary_calibration.py`. The `fit` command reads labels only from an
independent validation prediction CSV, selects `tau` by balanced accuracy,
fits positive `T` by binary NLL with `tau` fixed, and writes a JSON file that
includes the validation CSV SHA-256. The `apply` command reads only a
prediction CSV and the frozen JSON; it does not fit or modify parameters.

Verification: `conda run -n c2pclip python -m unittest discover -s tests -p
"test_*.py"` passed 85 tests.

## Validation Calibration Audit

EXP-D20-012 and EXP-D20-013 completed validation-only calibration for model
seeds 123, 42, and 2024. Training used the 12,800-entry manifest with SHA-256
`0e31e71b5ac10ced50e830c1021fa186554be2f1e851d0c5183099ea43e36d5a`.
Calibration used a separate 4,000-entry manifest with SHA-256
`21fe72c98ce3c5deeb6a1e3405618943662a16f43c91c08f47dd9494ca9cdd23`.
The split audit found zero missing targets and zero resolved real-file overlap.
Each same-name validation prediction CSV fitted its own `tau/T` JSON, which
was then frozen and applied to the corresponding test prediction CSV. This
remains image-only inference and does not use test labels for fitting.

After calibration, baseline versus HFR mean metrics were Macro ACC
85.97% versus 85.60%, AP 99.02% versus 99.26%, AUROC 99.25% versus 99.42%,
ECE 14.14% versus 14.80%, and Brier 0.1187 versus 0.1234. Thus HFR retains a
ranking gain but does not establish an advantage for calibrated accuracy or
calibration quality.

## Symmetric Hard-Example Ablation

Task ID: D20 / EXP-D20-014
Objective: Test whether HFR's fake-only asymmetry is necessary by adding a
coefficient- and selected-count-budget-matched hard-real branch.
Status: implementation completed; seed-123 server run planned

Code revision: `f571bf5d90b0b5e679e0e11704f5338c79381db9`.
Implementation: `hard_real_reweighting_loss` selects the highest-logit real
fraction from the global batch. New default-off options are
`--hard_real_loss_weight` and `--hard_real_fraction`; training logs report the
hard-real loss, selected count, total real count, and selected-logit mean.
The existing HFR defaults and image-only inference path are unchanged.

Protocol: seed 123, fixed `ds271828` manifest, hard-fake weight/fraction
0.5/0.25, and hard-real weight/fraction 0.5/0.25. The full command is stored
in `EXPERIMENT_PROTOCOL.md`. Expand to seeds 42 and 2024 only after the pilot
is registered and shows a better Real/Fake tradeoff without erasing HFR's
ranking gain.

Verification: 89 unit tests passed in the `c2pclip` environment. No training,
evaluation, checkpoint, or result has been produced locally.
