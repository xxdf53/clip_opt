# T50 Handoff: CAR Representative-Source Logit Grid

Task ID: T50
Objective: Implement a reproducible C2P-CLIP-Figure-5-style visualization comparing class-conditional raw-logit distributions for the C2P-CLIP Baseline and CAR.
Status: completed; real-data rendering needs review after prediction CSV synchronization

Inputs read: `AGENTS.md`; `docs/project/PROJECT_CHARTER.md`; `docs/project/CURRENT_STATE.md`; `docs/project/TASK_BOARD.md`; `docs/project/EXPERIMENT_PROTOCOL.md`; `paper_rewriting_output/figure_contract_logit_distribution.zh.md`; local C2P-CLIP Figure 5 and its caption.
Files changed: `scripts/plot_car_logit_grid.py`; `tests/test_plot_car_logit_grid.py`; `paper_rewriting_output/figure_contract_logit_distribution.zh.md`; `docs/project/TASK_BOARD.md`; this handoff.
Experiment IDs created or updated: none.
Code revision: the Git commit containing this handoff.
Commands or entry points used: `python scripts/plot_car_logit_grid.py --help`; focused `unittest` commands; Nature-figure source validator.
Data split and seed: no experiment was run; the script consumes registered prediction CSVs and does not choose sources from measured performance.
Artifacts and result locations: plotting entry point at `scripts/plot_car_logit_grid.py`; focused tests at `tests/test_plot_car_logit_grid.py`; final figure outputs are produced at the user-supplied `--output_prefix`.

Verified result: The script produces a two-row, five-column grid with C2P-CLIP above CAR. Defaults are Deepfakes, SITD, CRN, ADM, and VQDM, separated into GAN and diffusion protocol groups. It supports multiple CSV inputs per protocol, verifies required fields and finite values, rejects duplicate identities, requires exact Baseline/CAR sample-set and order agreement, retains every selected-source observation, uses shared bins/axes within each protocol, and records input hashes, source selection, sample counts, logit statistics, bin edges, outputs, and panel geometry. Eleven focused tests pass. Figure-source preflight reports 18 PASS, 3 WARN, and 0 FAIL; the remaining warnings concern optional TIFF output, dynamic CLI width, and static detection of the runtime DPI argument.

Interpretation: This is a qualitative behavior visualization. It may show whether CAR changes real/generated logit overlap or location on explicitly selected representative sources, but it does not independently prove hardness-specific causality, universal improvement, or superior feature extraction.
Decision and rationale: Use explicit source specifications rather than automatic top-gain selection. Keep GAN and diffusion logit scales separate because their models are trained under different protocols. Do not draw a threshold line.
Risks or unresolved issues: The current workspace lacks the registered real prediction CSVs. The exact generator strings must be checked on the server; defaults assume `deepfake`, `seeingdark`, `crn`, `adm`, and `vqdm`. After real rendering, run PDF glyph and collision audits and visually inspect every panel at final physical size. If GAN logits saturate, rerun the GAN side with `--gan_plot_kind ecdf` or `--gan_density_scale log` rather than clipping values.
Required next task: Synchronize the matched Baseline/CAR prediction CSVs, run the script, verify that each selected source contains both labels, execute final PDF/SVG/PNG QA, and only then add the figure and bounded interpretation to the manuscript.
