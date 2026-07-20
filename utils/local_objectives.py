import torch
import torch.nn.functional as F


def pairwise_ranking_loss(logits, labels):
    """Rank every fake above every real sample in the current global batch."""
    logits = logits.flatten()
    labels = labels.flatten()
    real_logits = logits[labels < 0.5]
    fake_logits = logits[labels >= 0.5]
    if real_logits.numel() == 0 or fake_logits.numel() == 0:
        return logits.sum() * 0.0
    differences = fake_logits[:, None] - real_logits[None, :]
    return F.softplus(-differences).mean()


def confidence_preservation_loss(final_logits, global_logits):
    """Penalize local corrections most where the global branch is confident."""
    global_logits = global_logits.flatten()
    final_logits = final_logits.flatten()
    confidence = (
        2.0 * (torch.sigmoid(global_logits.detach()) - 0.5).abs()
    )
    residual = final_logits - global_logits
    return (confidence * residual.square()).mean()


def gate_sparsity_loss(gates):
    """Keep the local residual opt-in instead of always active."""
    return gates.flatten().square().mean()


def residual_candidate_loss(global_logits, local_logits, labels):
    """Train a full local correction without weakening it by the small gate.

    The protected global logit is treated as a fixed starting point. This
    auxiliary path therefore updates only the local residual branch.
    """
    global_logits = global_logits.detach().flatten()
    local_logits = local_logits.flatten()
    labels = labels.flatten()
    candidate_logits = global_logits + local_logits
    return F.binary_cross_entropy_with_logits(candidate_logits, labels)


def relative_gate_target(global_logits, local_logits, labels, margin=0.1):
    """Build a soft gate target from the local correction's loss reduction.

    A harmful correction receives target zero. A useful correction opens the
    gate in proportion to its per-sample BCE improvement and reaches one after
    ``margin`` loss units. Confident correct global predictions have very
    little loss left to improve, so their targets naturally remain near zero.
    """
    if margin <= 0:
        raise ValueError(f'margin must be positive, got {margin}')

    with torch.no_grad():
        global_logits = global_logits.flatten()
        local_logits = local_logits.flatten()
        labels = labels.flatten()
        global_losses = F.binary_cross_entropy_with_logits(
            global_logits, labels, reduction='none')
        candidate_losses = F.binary_cross_entropy_with_logits(
            global_logits + local_logits, labels, reduction='none')
        improvement = (global_losses - candidate_losses).clamp_min(0.0)
        return (improvement / margin).clamp_max(1.0)


def relative_gate_supervision_loss(gates, global_logits, local_logits, labels,
                                   margin=0.1):
    """Supervise the image-adaptive gate with relative local reliability."""
    targets = relative_gate_target(
        global_logits, local_logits, labels, margin=margin)
    gates = gates.flatten().clamp(min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(gates, targets), targets
