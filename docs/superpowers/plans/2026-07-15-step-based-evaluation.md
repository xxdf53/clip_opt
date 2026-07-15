# Step-Based Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate held-out datasets only at configured training-step intervals and once at the end when needed.

**Architecture:** Add a pure scheduling helper that can be tested without importing the training stack. Update the existing training loop to call evaluation at `eval_freq` steps, stop both loops at `total_steps`, and avoid duplicate final evaluation.

**Tech Stack:** Python 3, standard-library `unittest`, existing PyTorch training code.

---

### Task 1: Evaluation scheduling helper

**Files:**
- Create: `utils/evaluation_schedule.py`
- Create: `tests/test_evaluation_schedule.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from utils.evaluation_schedule import should_evaluate


class EvaluationScheduleTests(unittest.TestCase):
    def test_evaluates_on_exact_interval(self):
        self.assertTrue(should_evaluate(step=800, eval_freq=800))

    def test_skips_non_interval_step(self):
        self.assertFalse(should_evaluate(step=799, eval_freq=800))

    def test_skips_zero_step(self):
        self.assertFalse(should_evaluate(step=0, eval_freq=800))

    def test_non_positive_frequency_disables_periodic_evaluation(self):
        self.assertFalse(should_evaluate(step=800, eval_freq=0))
        self.assertFalse(should_evaluate(step=800, eval_freq=-1))


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_evaluation_schedule -v`

Expected: failure because `utils.evaluation_schedule` does not exist.

- [ ] **Step 3: Implement the minimal helper**

```python
def should_evaluate(step, eval_freq):
    return step > 0 and eval_freq > 0 and step % eval_freq == 0
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `python -m unittest tests.test_evaluation_schedule -v`

Expected: four tests pass.

### Task 2: Integrate step-based evaluation

**Files:**
- Modify: `scripts/train.py:21-249`

- [ ] **Step 1: Import `should_evaluate` and add an evaluation/checkpoint helper**

The helper calls the existing `testmodel()`, saves using the current filename format, and returns the evaluated step.

- [ ] **Step 2: Replace epoch-end evaluation with periodic evaluation**

After each optimizer step, call the helper only when `should_evaluate(model.total_steps, opt.eval_freq)` is true.

- [ ] **Step 3: Stop the outer loop at `total_steps`**

Check the limit before processing a new batch and break the epoch loop once the configured number of optimizer steps has completed.

- [ ] **Step 4: Add final-evaluation fallback**

After training, evaluate once only when the last evaluated step differs from `model.total_steps`. This avoids duplicate evaluation when the final step is already an `eval_freq` boundary.

- [ ] **Step 5: Run verification**

Run:

```bash
python -m unittest tests.test_evaluation_schedule -v
python -m compileall -q scripts utils tests
```

Expected: all scheduling tests pass and compilation exits successfully.

### Task 3: Commit and push only source artifacts

**Files:**
- Stage explicitly: `scripts/train.py`, `utils/evaluation_schedule.py`, `tests/test_evaluation_schedule.py`, and this plan.

- [ ] **Step 1: Inspect repository state and staged file list**

Run:

```bash
git status --short
git diff --check
git diff --cached --name-only
```

Expected: no datasets, model weights, checkpoints, generated paper output, or image assets are staged.

- [ ] **Step 2: Commit implementation**

```bash
git commit -m "fix: evaluate training at configured step intervals"
```

- [ ] **Step 3: Push the current branch**

```bash
git push origin main
```

Expected: push succeeds and the remote `main` contains only the intended documentation, tests, helper, and training-loop changes.
