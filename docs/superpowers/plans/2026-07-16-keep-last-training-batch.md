# Keep Last Training Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in training flag that retains an incomplete final batch so two-GPU training can process all 144,024 images once.

**Architecture:** Keep DataLoader policy in a small pure helper for dependency-free unit testing. Expose `--keep_last_batch` through shared options and let `create_dataloader()` use the helper while preserving current defaults.

**Tech Stack:** Python 3, argparse, PyTorch DataLoader, standard-library unittest.

---

### Task 1: Final-batch policy

**Files:**
- Create: `utils/data_loading.py`
- Create: `tests/test_data_loading.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest

from utils.data_loading import should_drop_last_batch


class DropLastPolicyTests(unittest.TestCase):
    def test_training_drops_last_batch_by_default(self):
        self.assertTrue(should_drop_last_batch(True, False))

    def test_training_keeps_last_batch_when_requested(self):
        self.assertFalse(should_drop_last_batch(True, True))

    def test_evaluation_always_keeps_last_batch(self):
        self.assertFalse(should_drop_last_batch(False, False))
        self.assertFalse(should_drop_last_batch(False, True))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_data_loading -v`

Expected: import failure because `utils.data_loading` does not exist.

- [ ] **Step 3: Implement minimal policy**

```python
def should_drop_last_batch(is_train, keep_last_batch=False):
    return bool(is_train and not keep_last_batch)
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_data_loading -v`

Expected: three tests pass.

### Task 2: Command-line and DataLoader integration

**Files:**
- Modify: `options/base_options.py:29-39`
- Modify: `data/__init__.py:1-55`

- [ ] **Step 1: Add the opt-in argument**

Add to `BaseOptions.initialize()`:

```python
parser.add_argument(
    '--keep_last_batch',
    action='store_true',
    help='keep an incomplete final training batch instead of dropping it',
)
```

- [ ] **Step 2: Apply the policy in DataLoader creation**

Import the helper and replace the hard-coded `drop_last` expression with:

```python
drop_last=should_drop_last_batch(
    opt.isTrain,
    getattr(opt, 'keep_last_batch', False),
),
```

- [ ] **Step 3: Run all focused tests and compile**

```bash
python -m unittest tests.test_data_loading tests.test_evaluation_schedule -v
python -m compileall -q data options utils tests scripts
```

Expected: ten tests pass and compilation exits successfully.

### Task 3: Commit and push source-only changes

**Files:**
- Stage explicitly: `options/base_options.py`, `data/__init__.py`, `utils/data_loading.py`, `tests/test_data_loading.py`, and this plan.

- [ ] **Step 1: Verify staged scope**

```bash
git diff --cached --check
git diff --cached --name-only
git status --short
```

Expected: no dataset, checkpoint, model weight, image, archive, or `paper_rewriting_output/` file is staged.

- [ ] **Step 2: Commit implementation**

```bash
git commit -m "feat: optionally keep final training batch"
```

- [ ] **Step 3: Push main**

```bash
git push origin main
```

Expected: remote `main` advances to the implementation commit.
