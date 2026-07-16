# LoRA Logit Distribution Plot Design

## Goal

Make `scripts/plot_logit_dist.py` analyze checkpoints saved by `scripts/train.py`, for both baseline and local-feature LoRA classifiers, with optional same-figure comparison.

## Interface

The primary model uses `--checkpoint` and `--use_local_features`. An optional second model uses `--compare_checkpoint` and `--compare_use_local_features`. Both models share CLIP, LoRA, and local-feature dimension settings because the intended baseline/local experiments use matched training hyperparameters.

## Data Flow

Load the training checkpoint payload, validate its `model` state dictionary, remove only leading DataParallel `module.` prefixes, build the matching `CLIPModel_lora`, and load strictly. Run image-only classification with `model(img, None, None, cla=True)` and retain logits before sigmoid. Split logits by binary label.

## Visualization

Compute one bin array from every loaded real/fake distribution. Plot Real and Fake for each model on the same axes, draw the `logit=0` decision line, and print per-model mean, standard deviation, and d-prime-like separation. Reject empty classes and invalid checkpoint payloads with actionable errors.

## Verification

Pure unit tests cover checkpoint extraction, DataParallel prefix removal, statistics, and shared bins. CLI help and syntax checks run locally. Full CLIP checkpoint instantiation and GPU inference are left for the configured Linux environment and are not claimed as locally verified.
