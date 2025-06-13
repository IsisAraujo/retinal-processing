"""
Utility functions for HRF illumination correction analysis
"""

import cv2
import numpy as np
from typing import Tuple

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale if it's a color image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def calculate_gaussian_kernel_size(sigma: float) -> int:
    """
    Calculates an appropriate kernel size for Gaussian blur based on sigma.
    Ensures the kernel size is odd.
    """
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size

def normalize_reflectance(reflectance: np.ndarray) -> np.ndarray:
    """
    Normalizes reflectance values to the range [0, 1] using percentile-based scaling.
    """
    p2, p98 = np.percentile(reflectance, [2, 98])
    if (p98 - p2) < 1e-6:
        return np.zeros_like(reflectance)
    return np.clip((reflectance - p2) / (p98 - p2), 0, 1)

def convert_to_float_and_log(image: np.ndarray) -> np.ndarray:
    """
    Converts image to float64 and applies log transformation, handling zero values.
    """
    return np.log(image.astype(np.float64) + 1.0)

def normalize_image_for_display(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Normalizes a float image to 0-255 (CV_8U) for display, applying optional contrast/brightness.
    Clips values to 0-255 range.
    """
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Apply optional contrast/brightness adjustment
    adjusted_image = cv2.convertScaleAbs(normalized_image, alpha=alpha, beta=beta)
    return adjusted_image


