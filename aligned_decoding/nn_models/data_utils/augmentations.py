import torch
import numpy as np
import scipy.ndimage
from torchvision.transforms import Resize


def time_warping(data, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    warped = scipy.ndimage.zoom(data.numpy(), (1, factor, 1), order=1)
    # reize back to the original size
    warped = torch.tensor(warped)
    resize = Resize(data.size()[1:])
    warped = resize(warped)
    return warped


def time_masking(data, mask_ratio=0.1):
    n_time = data.size(1)
    mask_size = int(n_time * mask_ratio)
    mask_start = np.random.randint(0, n_time - mask_size)
    mask_end = mask_start + mask_size
    mask_data = data.clone()
    mask_data[:, mask_start:mask_end, :] = 0
    return mask_data


def time_shifting(data, shift_max=20):
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(data, shifts=shift, dims=1)


def noise_jitter(data, noise_level=0.01):
    noise = torch.randn_like(data) * noise_level
    return data + noise


def scaling(data, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return data * scale