# Unified Binary Evaluation Design

## Goal

Make official, baseline LoRA, and local-feature LoRA checkpoints consume the
same ordered image set, deterministic preprocessing, batching policy, and
metric implementation while preserving their different checkpoint loaders and
forward signatures.

## Architecture

Add `utils/binary_evaluation.py` as the shared image-only evaluation layer.
It will reuse `discover_binary_groups()`, build one path-returning
`ImageFolder` per paired `0_real/1_fake` leaf, concatenate sorted leaves by
top-level generator, construct deterministic loaders, run a caller-provided
logit function, and optionally export per-image raw logits and sigmoid scores.

`scripts/test_airplane_official.py` will retain official raw-state-dict model
loading and call the shared evaluator with `model(images)`.
`scripts/test_checkpoint.py` will retain LoRA checkpoint loading and call the
same evaluator with `model(images, None, None, cla=True)`. Its evaluation path
will no longer use the caption/tokenizer-aware training DataLoader.

## Data Contract

- Binary leaves contain exactly `0_real/` and `1_fake/`.
- Class indices are exactly real=0 and fake=1.
- Generators, leaves, and samples are deterministically sorted.
- Samples are returned as `(image, label, path)`.
- Evaluation uses `shuffle=False` and `drop_last=False`.
- Preprocessing is `_TranslateDuplicate(224)`, center crop 224, tensor
  conversion, and the existing CLIP mean/std normalization.
- Nested semantic directories are evaluated by concatenating their binary
  leaves, never by treating semantic names as labels.

## Metrics And Output

Both scripts will use `compute_binary_metrics()` for ACC, Real ACC, Fake ACC,
and AP. They will report per-generator macro metrics and metrics over all
images. The shared evaluator retains raw logits, sigmoid scores, labels, paths,
and generator names. When `--predictions_csv` is supplied, it writes:

```text
generator,path,label,raw_logit,score
```

## Error Handling

Reject missing roots, unpaired label directories, unexpected class mappings,
empty groups, invalid batch/worker counts, and mismatched output batch sizes.
CSV parent directories are created only when export is explicitly requested.

## Compatibility

Existing official and LoRA CLI arguments remain valid. Add `--num_workers`
and `--predictions_csv` to the LoRA script. The official script adds only
`--predictions_csv`. Checkpoint formats and model architecture flags remain
unchanged.

## Verification

Unit tests cover deterministic path/label order, direct and nested leaves,
shared preprocessing, final partial batches, raw-logit preservation, forward
adapter use, and CSV fields. Existing tests, syntax compilation, CLI help, and
real CNN_synth hierarchy counts must pass. No local GPU inference claim is
made.
