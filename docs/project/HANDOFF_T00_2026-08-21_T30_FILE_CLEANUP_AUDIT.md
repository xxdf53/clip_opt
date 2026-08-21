# T00 Handoff: T30 Non-Code File Audit

Task ID: T00
Objective: Extend T30 with a read-only audit of redundant non-code files.
Status: needs review

Inputs read: project charter; current state; task board; T30 audit; `.gitignore`; tracked, untracked, and ignored file inventories; selected temporary, paper, editor, cache, model, and generated-artifact directories
Files changed: `docs/project/T30_CODE_CLEANUP_AUDIT.md`; `docs/project/CURRENT_STATE.md`; `docs/project/TASK_BOARD.md`; `docs/project/DECISIONS.md`; this handoff
Experiment IDs created or updated: none
Code revision: audit performed at `72ea90038a4220efa1d3bc1519ca1f24051e0cda` with all pre-existing changes preserved
Commands or entry points used: read-only Git dry-run cleanup inventory, file-size inventory, reference search, and selected artifact hash comparison
Data split and seed: not applicable
Artifacts and result locations: `docs/project/T30_CODE_CLEANUP_AUDIT.md`

Verified result: `.pytest_cache/` and six source-tree `__pycache__/` directories are rebuildable. No duplicate selected PNG artifact hashes were found. Root logit plots and `tmp/pdfs` renders are unregistered but may still support research or writing.
Interpretation: Cache cleanup is low risk; generated research artifacts require path-by-path confirmation. Untracked does not mean disposable in this repository.
Decision and rationale: Protect models, datasets, captions, canonical records, historical paper evidence, task scripts, and user-tool configuration. Prohibit broad `git clean` and require exact deletion targets.
Risks or unresolved issues: The two IAPL page images have no source PDF beside them, and the root logit plots may be the only retained visual comparison even though they are not registered evidence.
Required next task: On user confirmation, T30.1 may delete the exact cache directories and separately confirmed generated artifacts together with the proven superseded code files.
