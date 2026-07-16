# LoRA Logit Distribution Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support train.py LoRA checkpoints and baseline/local comparison in the logit-distribution plotting script.

**Architecture:** Put checkpoint payload normalization in a dependency-free utility shared by testing and plotting. Keep inference, statistics, shared-bin construction, plotting, and CLI orchestration in the plotting script with pure calculation helpers covered by unit tests.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, argparse, unittest.

---

### Task 1: Checkpoint normalization

**Files:**
- Create: `utils/checkpoint_loading.py`
- Create: `tests/test_checkpoint_loading.py`
- Modify: `scripts/test_checkpoint.py`

- [ ] Write tests proving a wrapped `model` dictionary is required, `total_steps` is preserved, and only leading `module.` prefixes are removed.
- [ ] Run `python -m unittest tests.test_checkpoint_loading -v` and confirm the module-not-found RED failure.
- [ ] Implement `extract_training_state_dict(payload)` and switch `test_checkpoint.py` to it.
- [ ] Re-run the focused test and confirm GREEN.

### Task 2: Plot calculations and CLI

**Files:**
- Replace: `scripts/plot_logit_dist.py`
- Create: `tests/test_plot_logit_dist.py`

- [ ] Write tests for real/fake statistics, zero-variance separation, empty-class rejection, and shared bin edges across multiple models.
- [ ] Run the tests and confirm RED for missing helpers.
- [ ] Implement the requested CLI, strict LoRA model loading, raw-logit collection, unified histogram plotting, decision line, statistics, and optional comparison checkpoint.
- [ ] Re-run the plot helper tests and confirm GREEN.

### Task 3: Usage and verification

**Files:**
- Modify: `README.md`

- [ ] Add `my_first_test` baseline, local-feature, and comparison examples.
- [ ] Run all unit tests, `compileall`, and `plot_logit_dist.py --help` in the available environment.
- [ ] Inspect the final diff and explicitly stage only the files in this plan.
- [ ] Commit with `feat: support LoRA logit distribution comparison` and push the current branch.
