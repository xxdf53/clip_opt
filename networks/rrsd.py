"""Real-reference radial spectral deviation for image-only detection."""

import torch
import torch.nn as nn


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def radial_log_power_features(images, bands=16):
    """Extract normalized radial log-power features from CLIP input images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('images must have shape [batch, 3, height, width]')
    if bands <= 0:
        raise ValueError('bands must be positive')

    images = images.float()
    mean = images.new_tensor(_CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_CLIP_STD).view(1, 3, 1, 1)
    pixels = (images * std + mean).clamp(0.0, 1.0)
    luminance = (
        0.299 * pixels[:, 0]
        + 0.587 * pixels[:, 1]
        + 0.114 * pixels[:, 2]
    )
    luminance = luminance - luminance.mean(dim=(-2, -1), keepdim=True)

    height, spatial_width = luminance.shape[-2:]
    spectrum = torch.fft.rfft2(luminance, norm='ortho')
    power = spectrum.abs().square()
    vertical = torch.fft.fftfreq(
        height, device=images.device, dtype=images.dtype)
    horizontal = torch.fft.rfftfreq(
        spatial_width, device=images.device, dtype=images.dtype)
    radius = torch.sqrt(
        vertical[:, None].square() + horizontal[None, :].square())
    valid = radius > 0
    normalized_radius = radius / radius.max().clamp_min(1e-8)
    band_index = torch.floor(normalized_radius * bands).long()
    band_index = band_index.clamp(max=bands - 1)[valid]

    flattened_power = power.flatten(1)[:, valid.flatten()]
    sums = power.new_zeros((power.shape[0], bands))
    sums.scatter_add_(
        1,
        band_index.unsqueeze(0).expand(power.shape[0], -1),
        flattened_power,
    )
    counts = torch.bincount(band_index, minlength=bands).clamp_min(1)
    features = torch.log(sums / counts.to(sums.dtype) + 1e-8)
    features = features - features.mean(dim=1, keepdim=True)
    return features / features.std(
        dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


class RealReferenceSpectralResidual(nn.Module):
    """Bounded residual driven by deviation from a running real prototype."""

    def __init__(self, bands=16, hidden_dim=8, max_delta=0.5):
        super().__init__()
        if max_delta <= 0:
            raise ValueError('max_delta must be positive')
        self.bands = bands
        self.head = nn.Sequential(
            nn.Linear(bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head[-1].bias)
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer('real_prototype', torch.zeros(bands))
        self.register_buffer('real_count', torch.zeros(()))
        self.register_buffer('max_delta', torch.tensor(float(max_delta)))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.bands:
            raise ValueError(
                f'features must have shape [batch, {self.bands}]')
        deviation = (features - self.real_prototype).abs()
        correction = (
            self.max_delta
            * torch.tanh(self.gate)
            * torch.tanh(self.head(deviation).squeeze(1))
        )
        initialized = (self.real_count > 0).to(correction.dtype)
        return correction * initialized, deviation.mean(dim=1)

    @torch.no_grad()
    def update_real_prototype(self, real_sum, real_count):
        """Merge one globally aggregated real-only batch into the prototype."""
        real_sum = real_sum.to(
            device=self.real_prototype.device,
            dtype=self.real_prototype.dtype,
        )
        real_count = real_count.to(
            device=self.real_count.device,
            dtype=self.real_count.dtype,
        )
        if real_sum.shape != self.real_prototype.shape:
            raise ValueError(
                f'real_sum must have shape [{self.bands}]')
        if real_count.ndim != 0:
            raise ValueError('real_count must be scalar')
        if real_count.item() <= 0:
            return

        new_count = self.real_count + real_count
        new_prototype = (
            self.real_prototype * self.real_count + real_sum
        ) / new_count
        self.real_prototype.copy_(new_prototype)
        self.real_count.copy_(new_count)
