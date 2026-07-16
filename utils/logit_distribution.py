import math
import statistics


def compute_logit_stats(real_logits, fake_logits):
    """Compute per-class population statistics and d-prime-like separation."""
    real = list(real_logits)
    fake = list(fake_logits)
    if not real or not fake:
        raise ValueError('both real and fake logits are required')

    real_mean = statistics.fmean(real)
    fake_mean = statistics.fmean(fake)
    real_std = statistics.pstdev(real)
    fake_std = statistics.pstdev(fake)
    average_std = (real_std + fake_std) / 2.0
    mean_gap = abs(real_mean - fake_mean)

    if average_std == 0.0:
        separation = math.inf if mean_gap > 0.0 else 0.0
    else:
        separation = mean_gap / average_std

    return {
        'real_mean': real_mean,
        'real_std': real_std,
        'fake_mean': fake_mean,
        'fake_std': fake_std,
        'separation': separation,
    }


def build_shared_bin_edges(distributions, bins):
    """Build evenly spaced histogram edges spanning every distribution."""
    if bins <= 0:
        raise ValueError('bins must be positive')

    values = [value for distribution in distributions for value in distribution]
    if not values:
        raise ValueError('at least one logit is required to build histogram bins')

    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.01, 0.5)
        lower -= padding
        upper += padding

    width = (upper - lower) / bins
    return [lower + index * width for index in range(bins + 1)]
