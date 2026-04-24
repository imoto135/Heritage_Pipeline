"""Post-processing utilities for manuscript restoration."""

import numpy as np
from PIL import Image


def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction to increase contrast.

    Paper specifies gamma=1.5 to compensate for pale character generation.
    y = x^(1/gamma)
    """
    lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    return lut[img]


def apply_gamma_pil(img: Image.Image, gamma: float = 1.5) -> Image.Image:
    """Apply gamma correction to a PIL Image."""
    arr = np.array(img)
    corrected = apply_gamma(arr, gamma)
    return Image.fromarray(corrected)
