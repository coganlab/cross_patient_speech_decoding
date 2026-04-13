"""Data augmentation transforms for time-series neural feature tensors.

Each function accepts a batch tensor of shape (B, T, F) and returns an
augmented copy of the same shape.
"""

import torch
import numpy as np
import scipy.ndimage
from torchvision.transforms import Resize
from torch.nn import functional as F


def time_warping(data, factor_range=(0.8, 1.2)):
    """
    data: (B, T, F)
    """
    B, T, _ = data.shape
    factors = torch.empty(B, device=data.device).uniform_(*factor_range)

    warped = []
    for i in range(B):
        fac = factors[i]
        new_T = int(T * fac)

        # Interpolate from T → new_T
        x = data[i].unsqueeze(0).transpose(1, 2)     # (1, F, T)
        x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        x = F.interpolate(x, size=T, mode='linear', align_corners=False)
        x = x.transpose(1, 2).squeeze(0)             # back to (T, F)

        warped.append(x)

    return torch.stack(warped, dim=0)


def time_masking(data, mask_ratio=0.1):
    """
    data: (B, T, F)
    """
    B, T, _ = data.shape
    mask_size = int(T * mask_ratio)
    out = data.clone()

    starts = torch.randint(0, T - mask_size + 1, (B,))
    for i in range(B):
        out[i, starts[i]:starts[i] + mask_size] = 0

    return out


def time_shifting(data, shift_max=20):
    """
    data: (B, T, F)
    """
    B, T, _ = data.shape
    shifts = torch.randint(-shift_max, shift_max + 1, (B,), device=data.device)

    # Use advanced indexing to roll each batch item independently
    idx = (torch.arange(T, device=data.device)[None, :] - shifts[:, None]) % T
    return data[torch.arange(B).unsqueeze(1), idx]


def noise_jitter(data, noise_level=0.01):
    """Adds Gaussian noise to the data.

    Args:
        data: Input tensor of shape (B, T, F).
        noise_level: Standard deviation of the additive noise.

    Returns:
        Noisy copy of the input tensor.
    """
    noise = torch.randn_like(data) * noise_level
    return data + noise


def scaling(data, scale_range=(0.9, 1.1)):
    """Randomly scales each sample in the batch by a uniform factor.

    Args:
        data: Input tensor of shape (B, T, F).
        scale_range: (min, max) bounds for the uniform scaling factor.

    Returns:
        Scaled copy of the input tensor.
    """
    B = data.size(0)
    scales = torch.empty(B, 1, 1, device=data.device).uniform_(*scale_range)
    return data * scales