# Official CNN Synth Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate an official raw C2P-CLIP state dictionary across all direct and nested generators in CNN_synth_testset.

**Architecture:** Use a pure filesystem-discovery utility to map top-level generators to binary leaf directories. The official test script loads one model, constructs image-only concatenated datasets per group, computes per-group and aggregate metrics, and exposes explicit offline-friendly CLI paths.

**Tech Stack:** Python 3, pathlib, PyTorch, torchvision, NumPy, scikit-learn, argparse, unittest.

---

### Task 1: Recursive binary dataset discovery

**Files:**
- Create: `utils/binary_dataset_layout.py`
- Create: `tests/test_binary_dataset_layout.py`

- [ ] Write tests for direct roots, mixed direct/nested generators, sorted grouping, and roots without paired label directories.
- [ ] Run the focused tests and verify a module-not-found RED failure.
- [ ] Implement `discover_binary_groups(root)` returning an ordered mapping of group names to leaf paths.
- [ ] Re-run the focused tests and verify GREEN.

### Task 2: Official model test runner

**Files:**
- Replace: `scripts/test_airplane_official.py`

- [ ] Add positional compatibility and named arguments for dataroot, model_path, clip_path, batch_size, gpu, and num_workers.
- [ ] Load the official raw state dictionary strictly into one `C2P_CLIP` instance.
- [ ] Build pure-image datasets from discovered leaves and compute per-generator, macro, and overall metrics.
- [ ] Preserve deterministic CLIP-compatible preprocessing and validate class labels.

### Task 3: Usage and verification

**Files:**
- Modify: `README.md`

- [ ] Add a complete CNN_synth_testset official-model command.
- [ ] Run all unit tests, syntax compilation, CLI help, real hierarchy discovery, and a CPU image-loader smoke check.
- [ ] Explicitly stage only files in this plan, inspect the diff, commit, and push `main`.
