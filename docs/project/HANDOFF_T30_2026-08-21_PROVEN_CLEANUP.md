# T30 Handoff: Proven Cleanup

Task ID: T30.1
Objective: Remove proven superseded code and rebuildable caches, then apply behavior-neutral hygiene.
Status: needs review

Inputs read: project charter; current state; task board; T30 audit; Git status and diffs; source references; cache target resolution
Files changed: `.gitignore`; `README.md`; `scripts/inference.py`; `scripts/validate.py`; `scripts/Word_Frequency_Analysis.py`; `docs/project/TASK_BOARD.md`; `docs/project/CURRENT_STATE.md`; `docs/project/T30_CODE_CLEANUP_AUDIT.md`; this handoff
Files deleted: `scripts/test_model.py`; `scripts/test_chameleon.py`; `.pytest_cache/`; `data/__pycache__/`; `networks/__pycache__/`; `options/__pycache__/`; `scripts/__pycache__/`; `tests/__pycache__/`; `utils/__pycache__/`
Experiment IDs created or updated: none
Code revision: `ff3f939ffa950e1f370b844ea3337df35d5f22bb`; pre-existing user and task changes preserved
Commands or entry points used: exact-path cache dry run and cleanup; syntax compilation; `unittest` discovery; official evaluator and inference CLI help checks
Data split and seed: not applicable
Artifacts and result locations: this handoff and `docs/project/T30_CODE_CLEANUP_AUDIT.md`

Verified result: All 94 tests pass under `C:\Users\xxf\anaconda3\envs\c2pclip\python.exe`; three modified Python files compile; deleted script references are absent from active code and README; replacement official and inference CLI help commands return successfully; no cache target remains.
Interpretation: T30.1 removed 201 tracked lines and added 6 focused replacement/configuration lines, for a net reduction of 195 lines, plus 851,817 bytes of rebuildable cache, without changing model or evaluation behavior.
Decision and rationale: Delete only the two scripts proven covered by the unified official evaluator and the exact audited cache paths. Preserve SD/VQDM layout adapters, research plots, temporary paper evidence, models, datasets, captions, and tool state.
Risks or unresolved issues: `scripts/draw_tsne_kmean.py --help` cannot load in the tested environment because `MulticoreTSNE` is absent. This is pre-existing and outside T30.1. T30.2 layout migration and T30.3 research-utility decisions require separate confirmation.
Required next task: Review T30.1. Do not start T30.2 or delete conditional artifacts without user confirmation.
