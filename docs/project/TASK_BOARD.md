# Task Board

Only tasks marked `active` may modify their owned implementation area. Historical chats remain unchanged and are not represented as deleted work.

## Primary Research Tracks

| ID | Task | Status | Owner area | Protocol / contract | Completion condition |
|---|---|---|---|---|---|
| G10 | GAN image detection and generalization | active, awaiting user direction | Main thread `01a01357-e57b-75f2-aba2-359918fa00c7`; existing Anchor task is a historical subtask | `docs/project/EXPERIMENT_PROTOCOL.md` | Establish a reproducible GAN baseline and verify that any method improves AP, calibration, and worst-generator behavior beyond a matched control |
| D20 | Diffusion image detection and generalization | active, awaiting user direction | Main thread `01a01358-17f5-76c2-a81a-899c6595ef44`; existing HFR task is a historical subtask | `docs/project/EXPERIMENT_PROTOCOL.md` | Establish a strictly paired diffusion baseline and test candidates without threshold or test-set tuning |

## Shared Support Tasks

| ID | Task | Status | Owner area | Protocol / contract | Completion condition |
|---|---|---|---|---|---|
| T00 | Project control and evidence migration | active, phase confirmation required | `docs/project/` | `docs/project/PROJECT_CHARTER.md` | Historical facts are indexed and each live task has a defined contract |
| T10 | Baseline compatibility and reproduction check | queued for server execution | training configuration and checkpoints | `docs/project/T10_BASELINE_REPRO_AUDIT.md` | Run the official and self-trained baselines on the server, then explain the observed ACC/AP gap or record the precise blocker |
| T20 | Experiment result registry | active | evaluation commands and result evidence | `docs/project/T20_DATA_EVAL_AUDIT.md` | Keep matched experiments comparable through recorded Macro ACC and Macro AP |
| T40 | Cross-track analysis and statistics | blocked by confirmed evidence | results and analysis | `docs/project/EXPERIMENT_PROTOCOL.md` | Produce multi-seed, per-generator, calibration, robustness, and failure analysis before pooling conclusions |
| T50 | Paper evidence and writing | blocked by T40 | manuscript artifacts | `docs/project/EXPERIMENT_REGISTRY.csv` | Every claim states whether it concerns GANs, diffusion models, or both, and maps to a reproducible experiment record |

## Task Lifecycle

`queued` -> `active` -> `review` -> `completed` or `rejected`.

No task may be archived as a project record until its handoff is stored in the canonical documents. Archiving a chat is never deletion.

## Coordination Notes

- 2026-08-18: The dataset is treated as complete. T20 records experiment results, with Macro ACC and Macro AP as primary metrics; diagnostics are optional. T10 is queued for server-only official/self-trained reproduction.
- 2026-08-18: G10 and D20 are the two primary research tracks. Method proposals belong to one of them and are not promoted to cross-family claims without direct evidence.
- 2026-08-18: The T00 coordinating task is paused after the user requested confirmation at every reconstruction stage.
- 2026-08-18: The user resumed T00 coordination. EXP-D20-001's ordinary seven-group diffusion baseline output was linked into the T20 ledger and handoff summary; wait for user confirmation before the next coordination phase.
- 2026-08-19: EXP-D20-006 (HFR) and EXP-D20-007 (baseline) established the first matched fresh `ds271828` pair at seed 123. HFR improved Macro ACC by 5.87 pp and Macro AP by 0.28 pp, with a 5.36 pp Real ACC loss; seeds 42 and 2024 were subsequently completed in EXP-D20-008 through EXP-D20-011.
- 2026-08-19: D20 restored only the original global HFR objective from `28c26ee`; semantic-coverage, compensation, and bias-neutral variants remain retired. The current pair is pilot evidence only, not a paper-level HFR claim.
- 2026-08-19: EXP-D20-008 through EXP-D20-011 complete the same `ds271828` comparison for seeds 42 and 2024. Across three seeds HFR consistently improves Macro ACC/AP/AUROC/Brier, but lowers Real ACC and does not consistently improve ECE. Revision `19447ce92310f9b766d19b622e440151ef1f11ac` and manifest SHA-256 `0e31e71b5ac10ced50e830c1021fa186554be2f1e851d0c5183099ea43e36d5a` are recorded; preserve raw logs/CSVs before review.
- 2026-08-19: Operational instruction recorded: avoid `wc -l` and any command form that may close or exit the server terminal. Future data checks must preserve the active shell session.
- 2026-08-19: D20 validation calibration is complete and audited in EXP-D20-012/013. A 4,000-image calibration manifest (SHA-256 `21fe72c...cdd23`) has zero resolved real-file overlap with the 12,800-image training manifest (SHA-256 `0e31e71b...e36d5a`), and all six same-name validation CSV/JSON pairs completed. Frozen per-seed parameters were applied to test data only. HFR retains AP/AUROC gains but fails to improve calibrated Macro ACC, ECE, or Brier consistently.
- 2026-08-20: EXP-D20-014 is pre-registered as a seed-123, coefficient- and selected-count-budget-matched symmetric hard-example ablation at revision `f571bf5`. Run hard-fake/hard-real weights 0.5/0.5 and fractions 0.25/0.25 on the existing `ds271828` manifest; do not interpret or expand it before recording the pilot result.
- 2026-08-21: EXP-D20-014 completed seed 123 and improved validation-fixed Macro ACC over baseline from 86.08% to 88.74%, with AP/AUROC 99.15%/99.34%. Because it both halves hard-fake weight and adds hard-real, run pre-registered EXP-D20-015 (hard-fake weight 0.5 only) before expanding the symmetric variant or changing the main HFR decision.

## Existing Thread Mapping

| Existing task | Board mapping | Current use |
|---|---|---|
| `C2P-CLIP Anchor实验与后续优化` | G10, legacy subtask T11 | Preserve as the GAN-side logit-anchor record; do not use its development-set result as final evidence before the planned final-set evaluation is documented. |
| `C2P-CLIP 扩散泛化：HFR 严格配对实验` | D20, legacy subtask T12 | Active hard-fake reweighting pilot under a fixed diffusion manifest. It is distinct from DRCT reconstruction-pair research. |
| `C2P-CLIP CCF B/C 论文规划与撰写` | T50 | Writing and venue-planning record. It may summarize confirmed evidence but must not promote unverified experiment claims. |

## Main-Track Boundaries

### G10: GAN Image Detection And Generalization

- Data and evaluation: UniversalFakeDetect, CNN_synth, and related GAN or CNN-generated image benchmarks.
- Candidate methods: symmetric logit anchor, Counterfactual Prompt Direction (CPD), capacity-controlled local residuals, and other methods evaluated first against a GAN-matched baseline.
- CPD boundary: it may use paired captions and authenticity directions during training, but GAN-side deployment must remain image-only. Evaluate CPD-only and anchor-plus-CPD against the same GAN protocol.
- Required evidence: fixed protocol, per-generator metrics, calibration, and explicit analysis of StyleGAN-family regressions.

### D20: Diffusion Image Detection And Generalization

- Data and evaluation: SDv1.4 training, `diffusion_test_only`, GenImage-compatible settings, and other diffusion-generator benchmarks.
- Candidate methods: HFR, reconstruction-pair training, and image-adaptive prompts.
- Required evidence: identical train manifests for paired comparisons, fixed threshold, held-out evaluation discipline, and a separate assessment of any test-time adaptation cost.

### Shared Support

- T10 and T20 provide comparable baselines, datasets, manifests, and metrics for both tracks; they do not select a winning method.
- T40 may compare GAN and diffusion findings only after each track's experiment records are verified.
- T50 must not turn a method that works in one family into a universal claim without direct evidence.
