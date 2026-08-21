# Experiment Protocol

## Comparison Discipline

- Compare candidates with fixed data splits, preprocessing, total training steps, optimizer schedule, seed set, and evaluation code.
- When a method adds a feature branch, include a capacity-matched global-only classifier baseline.
- Select thresholds and calibration parameters only on a separate validation set, then keep them fixed for all test generators.
- Run at least three seeds before promoting a method beyond a pilot.

## Validation Threshold And Calibration

For D20 candidate comparisons, validation calibration is mandatory when
reporting fixed-threshold accuracy, ECE, or Brier score. The validation set
must be independent of training, manifest construction, test evaluation, and
method selection. Fit the threshold and temperature once from validation
prediction CSVs with `scripts/calibrate_predictions.py fit`; apply the frozen
JSON parameters to every test CSV with the `apply` subcommand. Never fit or
change parameters using `diffusion_test_only` labels.

The active offline rule is:

```text
calibrated_logit = (raw_logit - tau) / T
calibrated_probability = sigmoid(calibrated_logit)
prediction = calibrated_probability > 0.5
```

`tau` maximizes validation balanced accuracy. With `tau` fixed, positive `T`
minimizes validation binary NLL. AP and AUROC should also be reported from
the raw probabilities because the calibration transform is monotonic; ACC,
Real ACC, Fake ACC, ECE, Brier, and NLL should be reported after applying the
frozen parameters. The calibration JSON records the validation CSV SHA-256.

## Required Reporting

For every completed experiment, record Macro AP, AUROC, Macro ACC, Real ACC, Fake ACC, ECE, Brier score, per-generator metrics, raw-logit class statistics, training cost, and inference latency where applicable.

## Candidate-Specific Gates

### Local Residual

Reject it as a main method if a capacity-matched comparison shows only a fixed-threshold ACC increase while Macro AP, calibration, or worst-generator behavior fails to improve consistently.

### Reconstruction Pairs

Keep reconstruction strictly in training. Compare against equal-size ordinary augmentation and test on unseen diffusion generators, GANs, and common image degradation.

### Hard-Fake Reweighting (D20)

HFR is a training-only objective. Its paired baseline must retain the same
data order, effective global batch size, training steps, optimizer settings,
model seed, and evaluation configuration. For the user-supplied two-GPU
baseline, `--batch_size 64` is the global DataParallel batch; it must remain
64 for HFR, so each of `CUDA_VISIBLE_DEVICES=0,1` processes 32 images.

The pre-registered HFR command for `EXP-D20-006` differs from that baseline
only in `--name`, `--hard_fake_loss_weight 1.0`, and
`--hard_fake_fraction 0.25`:

```bash
env PYTHONUNBUFFERED=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
python -u scripts/train.py \
  --name hfr \
  --dataroot ./sdv1.4 \
  --textroot ./caption_sd4 \
  --clip ./clip-vit-large-patch14 \
  --checkpoints_dir ./c2p_checkpoints \
  --gpu_ids 0,1 \
  --batch_size 64 \
  --num_threads 8 \
  --niter 1 \
  --total_steps 200 \
  --eval_freq 0 \
  --loss_freq 10 \
  --lr 0.0002 \
  --loadSize 256 \
  --cropSize 224 \
  --seed 123 \
  --claloss 4 \
  --cates Deepfake Camera \
  --lora_r 6 \
  --lora_alpha 6 \
  --lora_dropout 0.5 \
  --hard_fake_loss_weight 1.0 \
  --hard_fake_fraction 0.25
```

The current source restores the original global HFR implementation from
`28c26ee40dbdd840d4703f268b5c567995425027`; both HFR flags are active again.
Later semantic-coverage, compensation, and bias-neutral HFR variants remain
retired. The supplied baseline has no `--train_manifest`, so neither run is
eligible for a strict paired claim until both commands use the same
predeclared fixed manifest.

### Budget-Matched Symmetric Hard-Example Ablation (D20)

EXP-D20-014 tests whether HFR's fake-only asymmetry is necessary. It selects
the lowest-logit 25% of fake samples and highest-logit 25% of real samples
from each global batch. Each side receives weight 0.5, preserving the total
selected-sample count and configured auxiliary-loss coefficient of HFR weight
1.0 when classes are balanced. Actual loss magnitudes may differ by class, so
training logs must report both auxiliary losses and selected-logit means.

This is an ablation, not a replacement for HFR. All new options default to
disabled, and inference remains image-only. The pre-registered seed-123 pilot
at revision `f571bf5d90b0b5e679e0e11704f5338c79381db9` is:

```bash
env PYTHONUNBUFFERED=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
python -u scripts/train.py \
  --name symmetric_hard_manifest_ds271828_s123 \
  --dataroot ./sdv1.4 \
  --textroot ./caption_sd4 \
  --clip ./clip-vit-large-patch14 \
  --checkpoints_dir ./c2p_checkpoints \
  --gpu_ids 0,1 \
  --batch_size 64 \
  --num_threads 8 \
  --niter 1 \
  --total_steps 200 \
  --eval_freq 0 \
  --loss_freq 10 \
  --lr 0.0002 \
  --loadSize 256 \
  --cropSize 224 \
  --seed 123 \
  --data_seed 271828 \
  --train_manifest ./training_manifests/d20_sdv14_train12800_ds271828.txt \
  --claloss 4 \
  --cates Deepfake Camera \
  --lora_r 6 \
  --lora_alpha 6 \
  --lora_dropout 0.5 \
  --hard_fake_loss_weight 0.5 \
  --hard_fake_fraction 0.25 \
  --hard_real_loss_weight 0.5 \
  --hard_real_fraction 0.25
```

Compare raw and validation-fixed Macro ACC, Real/Fake ACC, AP/AUROC, ECE,
and Brier against the seed-123 baseline and HFR. Expand to seeds 42 and 2024
only if the pilot materially improves the Real/Fake tradeoff without erasing
HFR's ranking gain.

### Fake-Only Half-Weight Control (D20)

EXP-D20-015 isolates the two changes made by EXP-D20-014. It retains the
lowest-logit 25% fake selection but uses hard-fake weight 0.5 and leaves
hard-real disabled. Relative to HFR weight 1.0, only auxiliary strength
changes; relative to EXP-D20-014, only the hard-real branch is removed.

```bash
env PYTHONUNBUFFERED=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
python -u scripts/train.py \
  --name hard_fake_w0.5_manifest_ds271828_s123 \
  --dataroot ./sdv1.4 \
  --textroot ./caption_sd4 \
  --clip ./clip-vit-large-patch14 \
  --checkpoints_dir ./c2p_checkpoints \
  --gpu_ids 0,1 \
  --batch_size 64 \
  --num_threads 8 \
  --niter 1 \
  --total_steps 200 \
  --eval_freq 0 \
  --loss_freq 10 \
  --lr 0.0002 \
  --loadSize 256 \
  --cropSize 224 \
  --seed 123 \
  --data_seed 271828 \
  --train_manifest ./training_manifests/d20_sdv14_train12800_ds271828.txt \
  --claloss 4 \
  --cates Deepfake Camera \
  --lora_r 6 \
  --lora_alpha 6 \
  --lora_dropout 0.5 \
  --hard_fake_loss_weight 0.5 \
  --hard_fake_fraction 0.25
```

Seed 123 completed as EXP-D20-015. Compare raw and validation-fixed metrics
with the baseline, HFR weight 1.0, and EXP-D20-014 before deciding whether
hard-real contributes beyond reducing fake-only weight.

### Fake-Only Half-Weight Multi-Seed Confirmation (D20)

EXP-D20-016 and EXP-D20-017 repeat the exact EXP-D20-015 training protocol at
model seeds 42 and 2024. Keep `data_seed=271828`, the fixed manifest, and every
other training setting unchanged. Hard-real remains disabled.

```bash
for SEED in 42 2024; do
  if [ "$SEED" = "42" ]; then
    EXP_ID="EXP-D20-016"
  else
    EXP_ID="EXP-D20-017"
  fi

  env PYTHONUNBUFFERED=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1 \
  python -u scripts/train.py \
    --name "hard_fake_w0.5_manifest_ds271828_s${SEED}" \
    --dataroot ./sdv1.4 \
    --textroot ./caption_sd4 \
    --clip ./clip-vit-large-patch14 \
    --checkpoints_dir ./c2p_checkpoints \
    --gpu_ids 0,1 \
    --batch_size 64 \
    --num_threads 8 \
    --niter 1 \
    --total_steps 200 \
    --eval_freq 0 \
    --loss_freq 10 \
    --lr 0.0002 \
    --loadSize 256 \
    --cropSize 224 \
    --seed "$SEED" \
    --data_seed 271828 \
    --train_manifest ./training_manifests/d20_sdv14_train12800_ds271828.txt \
    --claloss 4 \
    --cates Deepfake Camera \
    --lora_r 6 \
    --lora_alpha 6 \
    --lora_dropout 0.5 \
    --hard_fake_loss_weight 0.5 \
    --hard_fake_fraction 0.25 \
    2>&1 | tee "./log_files/${EXP_ID}_training_s${SEED}.log"

  TRAIN_STATUS=${PIPESTATUS[0]}
  printf '%s training_exit_status=%s\n' "$EXP_ID" "$TRAIN_STATUS"
done
```

Do not add hard-real flags. Evaluate both runs image-only under the same raw
threshold and independent-validation calibration protocol used for EXP-D20-015.

### Image-Adaptive Prompt

First test image-conditioned prompts with a fixed inference path. Treat test-time token tuning as a separate protocol because it requires per-image multi-view forward passes and backward updates. Reset any adapted state for every image.
