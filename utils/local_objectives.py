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
