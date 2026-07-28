# CPD Module Design

## Research hypothesis

C2P-CLIP aligns an image with one label-enhanced caption. The enhanced text
contains both the image content and the real/fake category prompt, so the LoRA
update may encode semantic shortcuts together with authenticity evidence.
Counterfactual Prompt Decomposition (CPD) tests whether separating these two
signals improves held-out-category, held-out-scene, and held-out-generator
generalization.

CPD is experimental. Its presence in the code does not establish that the
hypothesis is correct.

## Counterfactual text pair

For caption `c`, the frozen CLIP text encoder receives:

```text
t_real = T("Camera. " + c + " Camera.")
t_fake = T("Deepfake. " + c + " Deepfake.")
```

The order is always real first and fake second, matching binary labels 0 and 1.
The authenticity direction and content center are:

```text
d = normalize(t_fake - t_real)
m = normalize((t_fake + t_real) / 2)
```

The original C2P contrastive loss selects `t_real` for a real image and
`t_fake` for a fake image, so enabling CPD does not remove the original
image-caption supervision.

## LoRA task residual

The same augmented training image is encoded twice:

```text
v_lora = normalize(V_lora(x))
v_base = normalize(V_adapter_disabled(x))
delta_v = v_lora - stop_gradient(v_base)
```

The second visual forward is performed under `no_grad` and with the PEFT
adapter disabled. It exists only during CPD training.

## Objectives

For `s = 2y - 1`, the direction loss is:

```text
L_direction = mean(softplus(margin - s * dot(delta_v, d)))
```

This formulation has a finite useful gradient when the initially zero LoRA
residual has no well-defined cosine direction.

The content rejection loss is:

```text
L_content = mean(dot(delta_v, m)^2)
```

It only discourages the task-specific LoRA residual from following the paired
prompt's shared content direction. It does not remove semantics from the full
CLIP representation and does not claim statistical independence.

The complete implemented objective is:

```text
L = L_contrastive
  + alpha * L_BCE
  + lambda_anchor * L_SLAR
  + lambda_direction * L_direction
  + lambda_content * L_content
```

## Delayed linear warmup

Fixed CPD weights showed high variance across random seeds on held-out
generators. The stability variant uses a training-step multiplier `r(k)`:

```text
r(k) = 0                                      if k <= start
r(k) = min((k - start) / warmup, 1)          otherwise
```

Both CPD losses are multiplied by `r(k)`. With `start=400` and `warmup=400`,
the original C2P, classification, and Logit Anchor objectives train alone
through step 400; CPD then ramps to full strength at step 800. The paired text
and extra frozen visual forward are skipped while the effective CPD weight is
zero.

## Inference boundary

Inference remains:

```text
image -> CLIP vision encoder with trained LoRA -> linear classifier -> logit
```

No caption, tokenizer, text encoder, base-model forward, test-time adaptation,
generator identity, or target-domain threshold is used.

## Required ablations

1. C2P baseline.
2. C2P + SLAR.
3. C2P + CPD direction only.
4. C2P + CPD content only.
5. C2P + complete CPD.
6. C2P + SLAR + complete CPD.
7. Counterfactual prompts shuffled across images.
8. Random or swapped category-prompt controls.
9. Apply the same losses to the full image feature rather than the LoRA
   residual.

At least three seeds are required. Macro AP/AUROC and worst-generator results
are primary. Fixed-threshold ACC, real/fake ACC, ECE, Brier score, and logit
distributions are required diagnostics. Thresholds may only be selected on an
independent validation set.

## Falsification criteria

Stop treating CPD as a contribution if any of the following holds:

- direction-only CPD does not reproducibly improve held-out AP/AUROC;
- shuffled or random counterfactual prompts perform equally well;
- the signed residual projection does not increase above zero;
- content alignment falls but authenticity performance also falls;
- gains exist only on the training generator or at one chosen threshold;
- CPD and SLAR do not show separable or additive effects.
