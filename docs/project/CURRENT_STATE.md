# Current State

Last reviewed: 2026-08-19

## Migration Status

This file is the new live state record. The initial snapshot below is migrated from the July 18 handoff documents and must be treated as `reported, pending re-verification` unless an experiment registry row says otherwise.

## Confirmed Implementation Facts

- Backbone: local CLIP ViT-L/14.
- The text encoder and projection layer are frozen in the C2P-CLIP setup.
- LoRA is applied to the visual attention Q/K/V projections.
- The baseline classifier is `Linear(768, 1)`.
- Training combines image-caption InfoNCE with weighted BCE classification loss.
- The repository contains a unified image-only binary evaluation path with per-generator metrics and CSV output.

## Reported Experimental Snapshot

On the currently observed 12-generator, 89,969-image subset of `CNN_synth_testset`, the historical handoff reports:

| Model | Macro ACC | Macro AP | Verification state |
|---|---:|---:|---|
| Official released model | 95.45% | 99.64% | reported, pending rerun |
| Self-trained baseline | about 83.96% | about 98.69% | reported, pending rerun |
| Current local-feature model | 85.13% | 98.29% | reported, pending rerun |

The benchmark dataset is treated as complete for future experiment records.

## Active Risks And Blockers

- The self-trained baseline trails the official released model materially; method claims should wait until the reproduction gap is explained.
- The current local-feature comparison changes classifier capacity, so its gain is confounded.
- The local-feature result improves fixed-threshold ACC but lowers Macro AP and degrades StyleGAN and StyleGAN2 in the historical record.
- `options/train_options.py` has user-owned uncommitted changes. Do not overwrite it.
- `paper_rewriting_output/`, `.claude/`, and `tmp/` are currently untracked. They are not automatically protected by Git.

## Migrated Track Evidence

The following task records have been entered in `EXPERIMENT_REGISTRY.csv`. They are reported historical evidence, not yet fully rerun or paper-ready.

- GAN / G10: a symmetric logit-anchor experiment reached 91.18% Macro ACC and 99.60% Macro AP on a four-generator development split. The exact checkpoint path is missing and the nine-generator follow-up was not found, so it cannot support a final GAN claim yet.
- Diffusion / D20: an ordinary no-manifest baseline reached 91.31% Macro ACC and 98.94% Macro AP on the seven-group `diffusion_test_only` benchmark. It has a checkpoint path but is not paired with HFR and must not be used to calculate an HFR gain.
- Diffusion / D20: on the older fixed `data_seed=314159` manifest, HFR reported Macro ACC of 93.68%, 91.92%, and 91.95% for model seeds 123, 42, and 2024. The matching historical baselines were 93.10%, 90.60%, and 90.72% respectively. The manifest hash and raw artifacts remain to be recovered, and the repeatedly observed benchmark cannot be presented as untouched final evidence.
- CPD and symmetric logit-anchor are retained for the GAN track. Historical diffusion records report that CPD-only and anchor-style training reduced fixed-threshold diffusion accuracy, so they are not active D20 candidates.

## Candidate Directions By Track

### GAN: G10

| Direction | State | Required evidence before adoption |
|---|---|---|
| Symmetric logit anchor | historical development evidence | Recover the planned nine-generator result, then use fresh or predeclared paired evidence before a paper claim |
| Counterfactual Prompt Direction (CPD) | active GAN-only candidate | Compare CPD-only and Anchor+CPD to a matched GAN baseline; retain image-only inference and report category/generator failure cases |
| Capacity-controlled local residual | active research candidate | Matched-head, multi-seed ablation with AP, calibration, and worst-generator gains; explicitly audit StyleGAN-family regressions |

### Diffusion: D20

| Direction | State | Required evidence before adoption |
|---|---|---|
| Hard-Fake Reweighting (HFR) | implementation restored; historical paired candidate evidence | Repeat the original global objective under a predeclared paired manifest; report AP/AUROC and avoid claims based only on repeatedly observed tests |
| DRCT-style reconstruction pairs | proposal | Training-only pilot versus equal-size ordinary augmentation, then cross-check unseen diffusion models, GANs, and degradation |
| IAPL-style image-adaptive prompt | proposal | Conditional-prompt pilot before test-time tuning; separately report per-image adaptation latency, memory, and state-reset behavior |

## Next Gate

Complete the baseline and dataset-completeness audit before implementing a full candidate method.

## Coordination Status

- 2026-08-18: T10 is prepared for server-only official/self-trained baseline reproduction. No local inference result is recorded.
- 2026-08-18: T20 records experiment results in `docs/project/T20_AUDIT_LEDGER.csv`; Macro ACC and Macro AP are primary, while other diagnostics are optional.
- 2026-08-18: The user resumed T00 coordination after the pause. EXP-D20-001 is now linked to a T20 ledger row and handoff summary for the ordinary seven-group diffusion baseline output: Macro ACC 91.31, Macro AP 98.94, AUROC 99.18, ECE 41.02, Brier 0.2120, with `adm` as the weakest fixed-threshold generator.
- G10 and D20 remain gated by T10/T20 for any new full candidate implementation because the current registered evidence is still historical or pending rerun.
- 2026-08-19: Fresh seed-123, fixed-manifest (`ds271828`) results form a matched HFR pilot. Relative to EXP-D20-007 baseline, EXP-D20-006 HFR improves Macro ACC from 90.08% to 95.95% (+5.87 pp), Macro AP from 98.90% to 99.18% (+0.28 pp), AUROC from 99.16% to 99.37%, ECE from 41.30% to 38.04%, and Brier from 0.2122 to 0.2093. The improvement trades Real ACC down from 98.94% to 93.58% for Fake ACC up from 81.22% to 98.32%; `adm` remains the weakest group but its ACC rises from 72.35% to 92.22%.
- 2026-08-19: The fresh `ds271828` HFR comparison now covers model seeds 123, 42, and 2024. Mean baseline to HFR changes are: Macro ACC 88.32% to 95.54% (+7.23 pp), Macro AP 99.02% to 99.26% (+0.24 pp), AUROC 99.25% to 99.42% (+0.18 pp), ECE 40.35% to 39.28% (-1.06 pp), Brier 0.2037 to 0.2008 (-0.0029), Real ACC 99.25% to 96.17% (-3.08 pp), and Fake ACC 77.38% to 94.92% (+17.53 pp). HFR improves ACC, AP, AUROC, and Brier in all three seeds; ECE worsens for seed 2024. Code revision is `19447ce92310f9b766d19b622e440151ef1f11ac`; manifest SHA-256 is `0e31e71b5ac10ced50e830c1021fa186554be2f1e851d0c5183099ea43e36d5a`. Raw training/evaluation logs still need durable archival before promotion beyond candidate evidence.
- 2026-08-19: Server-operation constraint: do not provide `wc -l` or other commands that may terminate, close, or interrupt the user's terminal. Use non-destructive, shell-preserving alternatives for counts and checks, and keep commands executable within the current shell session.
- 2026-08-19: Validation-only calibration has completed for the three matched baseline/HFR seeds. The training manifest has 12,800 entries and SHA-256 `0e31e71b5ac10ced50e830c1021fa186554be2f1e851d0c5183099ea43e36d5a`; the independent calibration manifest has 4,000 entries and SHA-256 `21fe72c98ce3c5deeb6a1e3405618943662a16f43c91c08f47dd9494ca9cdd23`. Resolved real-file overlap is zero, with no missing targets. Same-name validation CSVs fitted each model/seed `tau/T`; frozen JSONs were then applied to test CSVs. HFR retains mean AP/AUROC gains of +0.24/+0.18 pp, but has worse calibrated Macro ACC (-0.37 pp), ECE (+0.66 pp), and Brier (+0.0046) than baseline. The calibration protocol is valid; it does not establish HFR as a calibration or fixed-threshold improvement.
- 2026-08-20: Revision `f571bf5d90b0b5e679e0e11704f5338c79381db9` adds a default-off hard-real objective for EXP-D20-014. The pre-registered seed-123 ablation splits HFR's auxiliary coefficient equally between the lowest-logit 25% fake samples and highest-logit 25% real samples. This is a budget-matched diagnostic of class asymmetry, not a new active method; no result has been reported.
