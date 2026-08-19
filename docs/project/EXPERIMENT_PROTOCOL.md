# Experiment Protocol

## Comparison Discipline

- Compare candidates with fixed data splits, preprocessing, total training steps, optimizer schedule, seed set, and evaluation code.
- When a method adds a feature branch, include a capacity-matched global-only classifier baseline.
- Select thresholds and calibration parameters only on a separate validation set, then keep them fixed for all test generators.
- Run at least three seeds before promoting a method beyond a pilot.

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

### Image-Adaptive Prompt

First test image-conditioned prompts with a fixed inference path. Treat test-time token tuning as a separate protocol because it requires per-image multi-view forward passes and backward updates. Reset any adapted state for every image.
