# Evaluation Frequency Design

## Goal

Avoid evaluating all held-out test categories after every epoch. Evaluation should follow `eval_freq`, matching the intended step-based training configuration while still producing a final result.

## Behavior

- Train only on the categories selected by `--classes`.
- Evaluate all subdirectories under `<dataroot>/test` when the completed training step is divisible by `eval_freq`.
- Save a checkpoint whenever an evaluation is performed.
- Stop the outer epoch loop once `total_steps` is reached.
- If `total_steps` is not divisible by `eval_freq`, perform one final evaluation and checkpoint save after training.
- Do not repeat evaluation at an epoch boundary merely because an epoch ended.

## Structure

Extract the trigger decision into a small pure helper so its boundary cases can be tested without loading PyTorch models or datasets. Keep the existing `testmodel()` implementation and checkpoint naming format.

## Verification

Tests cover regular intervals, non-interval steps, disabled/invalid frequencies, and the final-evaluation fallback. Static compilation verifies Linux-compatible Python syntax.
