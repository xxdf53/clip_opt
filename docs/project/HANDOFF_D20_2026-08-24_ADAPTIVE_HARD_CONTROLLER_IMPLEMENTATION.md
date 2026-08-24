# D20 Adaptive Hard Controller Implementation Handoff

Task ID: D20
Objective: Implement and verify the default-off label-free adaptive HFR/HRR controller authorized by the implementation gate.
Status: completed

Inputs read: `AGENTS.md`, `docs/project/PROJECT_CHARTER.md`, `docs/project/CURRENT_STATE.md`, `docs/project/TASK_BOARD.md`, `docs/project/DECISIONS.md`, `docs/project/EXPERIMENT_PROTOCOL.md`, and `docs/project/HANDOFF_TEMPLATE.md`.
Files changed: `utils/training_objectives.py`, `networks/trainer.py`, `options/base_options.py`, `scripts/train.py`, `tests/test_hard_fake_reweighting.py`, `tests/test_training_options.py`, `tests/test_training_entrypoint.py`, `docs/project/CURRENT_STATE.md`, `docs/project/TASK_BOARD.md`, and this handoff. The user-owned `options/train_options.py` was not modified.
Experiment IDs created or updated: none.
Code revision: uncommitted working tree based on `4dbf41afa2029e447d10f0e5abe1eaaa30b26dcb`; no commit or push was performed.
Commands or entry points used: healthy `c2pclip` Python at `C:\Users\xxf\anaconda3\envs\c2pclip\python.exe`; focused and full `unittest` runs; `py_compile`; `scripts/train.py --help`; `git diff --check`.
Data split and seed: none; no dataset was read and no training or evaluation was run.
Artifacts and result locations: source, tests, and canonical records listed above; no checkpoint, prediction, metric, or experiment artifact was generated.

Verified result: The controller is disabled by default with `--adaptive_hard_loss_weight 0.0`. When enabled, it reuses existing hard-fake and hard-real selections, computes detached selected-sample mean BCE routing statistics, optionally applies EMA, uses equal `0.5/0.5` warmup shares when both sides are available, and allocates one fixed total coefficient through a temperature softmax. Optimization retains the existing selected BCE sum divided by global batch size. Missing or zero-selected classes produce finite deterministic shares. A fake-only or real-only batch receives the full `1.0` share on its available side even during warmup; both missing retains deterministic `0.5/0.5`. Softmax statistics are centered and the positive temperature is clamped to the routing dtype's finite lower bound, so an accepted temperature as small as `1e-45` remains finite and sums to one. Adaptive use rejects nonzero static hard-fake or hard-real weights. Compact experiment names and training logs record the controller configuration, shares, routing statistics, and selected counts. Focused tests pass 39/39 and the full suite passes 103/103; syntax compilation, training CLI help, and diff checks also pass.
Interpretation: This completes only the implementation gate. It establishes no performance result, family recognition behavior, mixed-family conclusion, or paper claim. Model inputs, outputs, inference, and data loading are unchanged.
Decision and rationale: Retain the controller as a default-off training-only option. Its detached router prevents gradients from optimizing the allocation statistics, while the fixed coefficient and existing budget losses keep adaptive and fixed-share controls comparable.
Risks or unresolved issues: The existing checkpoint format saves model parameters and `total_steps` but not auxiliary controller state. Exact EMA continuation is therefore unsupported; `--continue_train` emits a warning and starts the EMA from empty state. Resolving this would require a separately scoped checkpoint-system change. Mixed-family manifests, batch schedule, controls, seeds, and validation-only threshold protocol remain unregistered.
Required next task: G10 and D20 must jointly pre-register the mixed-family protocol and minimum controls named in `docs/project/EXPERIMENT_PROTOCOL.md` before any server training or experiment ID is created.
