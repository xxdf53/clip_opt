# Decision Log

| Date | Decision | Status | Evidence | Consequence |
|---|---|---|---|---|
| 2026-08-18 | Use repository documents as shared task memory | accepted | Project tasks do not reliably receive full chat context | New work must read and update `docs/project/` |
| 2026-08-18 | Preserve historical task chats and handoff documents | accepted | User requested that no tasks be deleted | Migration is additive; archival requires a stored handoff |
| 2026-08-18 | Do not promote the current local mean/std branch as a contribution | provisional | Historical record reports AP and StyleGAN-family regressions plus capacity confounding | Require capacity-matched, multi-seed evidence |
| 2026-08-18 | Keep DRCT and IAPL as separate proposals | accepted | Both are external methods with distinct protocol and novelty risks | Run isolated pilots only after T10/T20 |
| 2026-08-18 | Define T10 and T20 as prerequisite audit contracts | accepted | `CURRENT_STATE.md` lists the baseline reproduction gap and dataset completeness mismatch as the next gate | T10/T20 must reach review or document blockers before full candidate implementation |
| 2026-08-18 | Keep CPD in the GAN track only | provisional | Historical D20 records report CPD-only performance declines across the tested diffusion seeds | Evaluate CPD-only and Anchor+CPD under G10; do not reuse it as an active D20 method without new evidence |
| 2026-08-18 | Treat historical HFR gains as candidate evidence, not a paper claim | accepted | EXP-D20-003 through EXP-D20-005 report positive paired gains, but the fixed manifest and benchmark were repeatedly observed | Recover artifacts or create a predeclared fresh paired protocol before promotion to T40/T50 |
| 2026-08-19 | Restore the original global HFR implementation only | accepted | User-directed restoration; EXP-D20-003 through EXP-D20-005 identify `28c26ee` as the historical candidate implementation | Keep later HFR variants retired; require a fresh matched run before any conclusion |
