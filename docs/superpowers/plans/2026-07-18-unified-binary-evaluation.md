# Unified Binary Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make official and self-trained C2P-CLIP checkpoints use one deterministic, pure-image binary evaluation pipeline.

**Architecture:** Create a focused shared module for path-returning datasets, CLIP preprocessing, deterministic loaders, inference collection, aggregation, and optional CSV export. Keep model loading and forward signatures in their existing scripts, which become thin adapters around the shared evaluator.

**Tech Stack:** Python 3.10, PyTorch, torchvision, pathlib, NumPy, scikit-learn, argparse, unittest.

---

### Task 1: Shared dataset contract

**Files:**
- Create: `tests/test_binary_evaluation.py`
- Create: `utils/binary_evaluation.py`

- [ ] Write a failing test that creates direct and nested binary leaves with
  real image files and asserts deterministic `(image, label, path)` order.
- [ ] Run `python -m unittest tests.test_binary_evaluation -v` and confirm it
  fails because `utils.binary_evaluation` does not exist.
- [ ] Implement `build_transform()`, `PathImageFolder`, and
  `build_group_dataset()` using the exact CLIP preprocessing and strict
  `{'0_real': 0, '1_fake': 1}` mapping.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Shared inference and aggregation

**Files:**
- Modify: `tests/test_binary_evaluation.py`
- Modify: `utils/binary_evaluation.py`

- [ ] Add failing tests for a final partial batch, caller-provided logit
  function, raw-logit preservation, generator tagging, and macro/overall
  aggregation.
- [ ] Run the focused suite and confirm failures reference missing shared
  inference functions.
- [ ] Implement `evaluate_dataset()` and `evaluate_groups()` with
  `shuffle=False`, `drop_last=False`, score computation via sigmoid, and
  `compute_binary_metrics()`.
- [ ] Re-run focused and existing binary layout/metric tests.

### Task 3: Optional prediction CSV

**Files:**
- Modify: `tests/test_binary_evaluation.py`
- Modify: `utils/binary_evaluation.py`

- [ ] Add a failing test asserting the exact CSV header and one row per image.
- [ ] Implement `write_predictions_csv()` with explicit parent creation and
  UTF-8 newline-safe output.
- [ ] Re-run the focused test and confirm it passes.

### Task 4: Refactor both test scripts

**Files:**
- Modify: `scripts/test_airplane_official.py`
- Modify: `scripts/test_checkpoint.py`
- Modify: `README.md`

- [ ] Replace official-script-local dataset/inference code with imports from
  `utils.binary_evaluation`; retain strict official model loading.
- [ ] Replace LoRA `create_dataloader()` evaluation with the same shared
  module and `model(images, None, None, cla=True)`.
- [ ] Add `--predictions_csv` to both scripts and `--num_workers` to the
  LoRA script.
- [ ] Keep existing checkpoint architecture arguments and direct/nested roots.
- [ ] Update README commands for official, baseline, and local evaluation.

### Task 5: Verification

**Files:**
- Test: `tests/`

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q scripts utils tests`.
- [ ] Run both script `--help` commands.
- [ ] Assert the real local hierarchy has 13 groups and 90,329 samples.
- [ ] Run a CPU DataLoader smoke test and verify a `(64,3,224,224)` batch.
- [ ] Run `git diff --check` and inspect `git status` to ensure datasets,
  models, `.claude/`, and paper outputs remain unstaged.
