"""Data augmentation functions for temporal neural signal data.

Provides time-domain augmentations including warping, masking, shifting,
noise injection, and scaling for neural decoding training data.
"""

import torch
import numpy as np
import scipy.ndimage
from torchvision.transforms import Resize


def time_warping(data, factor_range=(0.8, 1.2)):
    """Warps the temporal axis by a random factor, then resizes to original length.

    Args:
        data: Input tensor of shape (n_trials, n_timepoints, n_features).
        factor_range: Tuple of (min, max) warp factors. Defaults to (0.8, 1.2).

    Returns:
        Tensor: Time-warped data with original shape restored.
    """
    factor = np.random.uniform(*factor_range)
    warped = scipy.ndimage.zoom(data.numpy(), (1, factor, 1), order=1)
    # reize back to the original size
    warped = torch.tensor(warped)
    resize = Resize(data.size()[1:])
    warped = resize(warped)
    return warped


def time_masking(data, mask_ratio=0.1):
    """Zeros out a contiguous random segment of the temporal axis.

    Args:
        data: Input tensor of shape (n_trials, n_timepoints, n_features).
        mask_ratio: Fraction of timepoints to mask. Defaults to 0.1.

    Returns:
        Tensor: Data with a masked temporal segment set to zero.
    """
    n_time = data.size(1)
    mask_size = int(n_time * mask_ratio)
    mask_start = np.random.randint(0, n_time - mask_size)
    mask_end = mask_start + mask_size
    mask_data = data.clone()
    mask_data[:, mask_start:mask_end, :] = 0
    return mask_data


def time_shifting(data, shift_max=20):
    """Circularly shifts data along the temporal axis by a random offset.

    Args:
        data: Input tensor of shape (n_trials, n_timepoints, n_features).
        shift_max: Maximum shift in either direction. Defaults to 20.

    Returns:
        Tensor: Temporally shifted data.
    """
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(data, shifts=shift, dims=1)


def noise_jitter(data, noise_level=0.01):
    """Adds Gaussian noise to the data.

    Args:
        data: Input tensor of shape (n_trials, n_timepoints, n_features).
        noise_level: Standard deviation of the added noise. Defaults to 0.01.

    Returns:
        Tensor: Data with additive Gaussian noise.
    """
    noise = torch.randn_like(data) * noise_level
    return data + noise


def scaling(data, scale_range=(0.9, 1.1)):
    """Scales data by a random uniform factor.

    Args:
        data: Input tensor of shape (n_trials, n_timepoints, n_features).
        scale_range: Tuple of (min, max) scale factors. Defaults to (0.9, 1.1).

    Returns:
        Tensor: Scaled data.
    """
    scale = np.random.uniform(*scale_range)
    return data * scale