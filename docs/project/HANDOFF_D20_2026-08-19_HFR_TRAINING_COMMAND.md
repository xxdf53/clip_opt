# D20 HFR Training Command Handoff

Task ID: D20
Objective: Restore the original global HFR objective and retain its pre-registered training command.
Status: needs review

Inputs read: `AGENTS.md`, `PROJECT_CHARTER.md`, `CURRENT_STATE.md`, `TASK_BOARD.md`, `EXPERIMENT_PROTOCOL.md`, `EXPERIMENT_REGISTRY.csv`, and `HANDOFF_TEMPLATE.md`.
Files changed: `utils/training_objectives.py`, `networks/trainer.py`, `options/base_options.py`, `utils/retired_training.py`, `scripts/train.py`, HFR tests, and linked project records.
Experiment IDs created or updated: EXP-D20-006 (planned only)
Code revision: active uncommitted worktree; the restored HFR implementation is copied from `28c26ee40dbdd840d4703f268b5c567995425027`.
Commands or entry points used: `scripts/train.py`; the complete command is recorded in `EXPERIMENT_PROTOCOL.md`.
Data split and seed: `./sdv1.4`; model seed 123; no data seed or fixed training manifest was supplied.
Artifacts and result locations: none; the command has not run.

Verified result: implementation restoration and syntax checks only; no training or evaluation has run.
Interpretation: The restored objective adds the normalized BCE of the globally lowest-logit fake fraction to the original classification loss. The command preserves two GPUs, global batch size 64, 200 steps, optimizer settings, preprocessing, LoRA configuration, and disabled in-training evaluation.
Decision and rationale: Restore only the global HFR objective associated with the historical candidate evidence. Later semantic-coverage, compensation, and bias-neutral variants remain retired. Without a common predeclared fixed manifest, a run cannot support the D20 strict-pair claim.
Risks or unresolved issues: The exact shared manifest and its SHA-256 are absent. Full runtime tests are blocked locally because the Python environment imports an invalid `numpy` module without `ndarray`.
Required next task: On user confirmation, create or identify one fixed 12,800-sample manifest, run the baseline and HFR commands against it, then register artifacts and evaluation results before updating any conclusion.
