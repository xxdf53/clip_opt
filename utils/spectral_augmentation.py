"""Training-only spectral augmentations for generalizable image detection."""

import torch


_SPECTRAL_BAND_WIDTH = 0.12
_MIN_BAND_RADIUS = 0.05


def _radial_frequency_grid(height, width, device, dtype):
    """Return normalized radii for an rFFT spectrum of one image."""
    vertical = torch.fft.fftfreq(height, device=device, dtype=dtype)
    horizontal = torch.fft.rfftfreq(width, device=device, dtype=dtype)
    radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
    return radius / radius.max().clamp_min(torch.finfo(dtype).eps)


def spectral_band_dropout(images, probability):
    """Randomly remove one narrow radial Fourier band from selected images.

    The same label is retained.  Bands exclude the DC neighbourhood, use a
    fixed 12% normalized-radial width, and are sampled independently per
    image.  This makes a detector less dependent on a generator-specific
    frequency range while adding nothing to the inference path.
    """
    if images.ndim != 4:
        raise ValueError('images must have shape [batch, channels, height, width]')
    if not 0.0 <= probability <= 1.0:
        raise ValueError('spectral band dropout probability must be in [0, 1]')
    if probability == 0.0:
        zero = images.new_zeros(())
        return images, {
            'spectral_band_applied': zero,
            'spectral_band_mask_fraction': zero,
        }

    batch_size, _, height, width = images.shape
    applied = torch.rand(batch_size, device=images.device) < probability
    if not torch.any(applied):
        zero = images.new_zeros(())
        return images, {
            'spectral_band_applied': zero,
            'spectral_band_mask_fraction': zero,
        }

    radius = _radial_frequency_grid(
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    max_start = 1.0 - _SPECTRAL_BAND_WIDTH
    starts = _MIN_BAND_RADIUS + (
        max_start - _MIN_BAND_RADIUS
    ) * torch.rand(batch_size, device=images.device, dtype=images.dtype)
    band = (radius.unsqueeze(0) >= starts[:, None, None]) & (
        radius.unsqueeze(0) < (
            starts[:, None, None] + _SPECTRAL_BAND_WIDTH))
    band = band & applied[:, None, None]

    spectrum = torch.fft.rfft2(images, dim=(-2, -1), norm='ortho')
    spectrum = spectrum.masked_fill(band[:, None], 0)
    augmented = torch.fft.irfft2(
        spectrum,
        s=(height, width),
        dim=(-2, -1),
        norm='ortho',
    )
    return augmented, {
        'spectral_band_applied': applied.float().mean(),
        'spectral_band_mask_fraction': band.float().mean(),
    }
