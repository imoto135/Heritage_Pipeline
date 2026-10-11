"""Evaluation metrics for manuscript stain reduction.

Paper metrics:
- PSNR / SSIM: image similarity with ground truth (requires reference)
- Kurtosis: higher = more peaked histogram = less stain noise
- Brightness variance: higher = higher contrast = better visibility
- Entropy: lower = less randomness = cleaner background
"""

import numpy as np
from PIL import Image
from scipy.stats import kurtosis as scipy_kurtosis
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.measure import shannon_entropy


def load_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale."""
    if img.ndim == 3:
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
    return img.astype(np.float32)


def compute_kurtosis(img: np.ndarray) -> dict:
    """Compute kurtosis for each RGB channel.

    Higher kurtosis = more peaked = cleaner (less stain noise).
    """
    result = {}
    if img.ndim == 3:
        for i, ch in enumerate(['R', 'G', 'B']):
            result[f'kurtosis_{ch}'] = float(scipy_kurtosis(img[:, :, i].flatten(), fisher=False))
    else:
        result['kurtosis'] = float(scipy_kurtosis(img.flatten(), fisher=False))
    return result


def compute_brightness_variance(img: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute brightness variance (sigma).

    Higher variance = higher contrast = better visibility.
    If mask is provided, compute only on non-text (background) pixels.
    mask: 1 = background, 0 = text
    """
    gray = load_gray(img)
    if mask is not None:
        bg_pixels = gray[mask > 0]
        if len(bg_pixels) == 0:
            bg_pixels = gray.flatten()
    else:
        bg_pixels = gray.flatten()
    return float(np.std(bg_pixels))


def compute_entropy(img: np.ndarray) -> float:
    """Compute Shannon entropy of the luminance channel.

    Lower entropy = less randomness = cleaner background.
    """
    gray = load_gray(img).astype(np.uint8)
    return float(shannon_entropy(gray))


def compute_psnr(restored: np.ndarray, reference: np.ndarray) -> float:
    """Compute PSNR between restored and reference image."""
    return float(skimage_psnr(reference, restored, data_range=255))


def compute_ssim(restored: np.ndarray, reference: np.ndarray) -> float:
    """Compute SSIM between restored and reference image."""
    return float(skimage_ssim(reference, restored, channel_axis=2 if restored.ndim == 3 else None, data_range=255))


def compute_metrics(restored: np.ndarray, reference: np.ndarray = None, mask: np.ndarray = None) -> dict:
    """Compute all paper metrics.

    Args:
        restored: Restored image as uint8 numpy array (H, W, 3)
        reference: Ground truth image (optional, for PSNR/SSIM)
        mask: Background mask 1=bg 0=text (optional, for brightness variance)

    Returns:
        Dict of metric name -> value
    """
    metrics = {}
    metrics.update(compute_kurtosis(restored))
    metrics['brightness_std'] = compute_brightness_variance(restored, mask)
    metrics['entropy'] = compute_entropy(restored)

    if reference is not None:
        metrics['psnr'] = compute_psnr(restored, reference)
        metrics['ssim'] = compute_ssim(restored, reference)

    return metrics
