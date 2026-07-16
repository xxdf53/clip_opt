# Keep Last Training Batch Design

## Goal

Allow two-GPU training to process all 144,024 selected images exactly once while preserving the existing default behavior for other experiments.

## Interface

Add a `--keep_last_batch` command-line flag. When omitted, training continues to discard an incomplete final batch. When enabled, the training DataLoader includes the final incomplete batch. Evaluation DataLoaders always keep their final batch.

## Two-GPU Configuration

Use a global batch size of 64 on two visible GPUs. The dataset produces 2,251 batches: 2,250 full batches of 64 images and one final batch of 24 images. The final batch divides evenly across two GPUs, so each replica receives 12 images.

Set `total_steps=2251` and `eval_freq=2251` to process one complete shuffled epoch and evaluate once at the end.

## Compatibility

The new flag is opt-in. Existing training commands retain `drop_last=True`, and test/inference loading remains unchanged.

## Verification

Unit tests cover default training behavior, opt-in final-batch retention, and evaluation behavior. Static compilation checks Python syntax. Git staging uses an explicit source-file list so datasets, checkpoints, model weights, images, and `paper_rewriting_output/` remain untracked and unpushed.
