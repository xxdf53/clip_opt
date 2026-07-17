# Official CNN Synth Test Design

## Goal

Extend `scripts/test_airplane_official.py` so one official C2P-CLIP model load can evaluate the complete mixed-depth `CNN_synth_testset` hierarchy.

## Dataset Discovery

Recursively identify every directory whose direct children include both `0_real/` and `1_fake/`. Group each binary leaf by its first path component relative to the requested root. A root that directly contains the label directories is treated as one group named after the root itself.

## Inference

Load the official raw `state_dict` into `C2P_CLIP` once using a required local CLIP directory. Build image-only `ImageFolder` datasets for each binary leaf, concatenate leaves within a generator, and run sigmoid classification without Caption or tokenizer inputs.

## Reporting

Print image count, ACC, Real ACC, Fake ACC, and AP for each top-level generator. Print macro means across generators and aggregate metrics across every image. Reject missing model/CLIP paths, malformed label directories, empty classes, and invalid GPU indices with actionable messages.

## Compatibility And Verification

Support both named CLI arguments and the previous two positional paths. Unit tests cover direct, nested, mixed, and empty directory discovery plus metric calculation. Local verification includes syntax, CLI help, discovery against the 90,329-image dataset, and image-only DataLoader checks; full GPU model inference is not claimed locally.
