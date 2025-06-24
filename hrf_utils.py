"""
Utility functions for HRF illumination correction analysis
"""

import cv2
import numpy as np
from typing import Tuple

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Extracts the green channel from color images for enhanced retinal analysis.

    Scientific justification: "retinal components are more intense in the fundus
    image's green channel" (Interactive Blood Vessel Segmentation from Retinal
    Fundus Image Based on Canny Edge Detector, 2021, PMC8512020)

    Args:
        image: Input image array (BGR or grayscale)

    Returns:
        Green channel extracted or original image if already grayscale
    """
    if len(image.shape) == 3:
        # Extract green channel directly (BGR: 0=Blue, 1=Green, 2=Red)
        return image[:, :, 1]
    return image

def to_grayscale_original(image: np.ndarray) -> np.ndarray:
    """
    Original function kept for comparison or fallback if needed.
    Standard BGR to grayscale conversion using OpenCV weights.
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

# FUNÇÕES AUXILIARES PARA DEMONSTRAR O IMPACTO DA MUDANÇA

def compare_channels(image: np.ndarray) -> dict:
    """
    Helper function to compare different channel extractions.
    Useful for validating the change and demonstrating differences.

    Returns:
        dict with different grayscale versions of the image
    """
    if len(image.shape) != 3:
        return {"original": image}

    return {
        "green_channel": image[:, :, 1],           # Our new implementation
        "opencv_gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),  # Original method
        "red_channel": image[:, :, 2],             # For comparison
        "blue_channel": image[:, :, 0],            # For comparison
    }

def calculate_vessel_contrast(image: np.ndarray) -> float:
    """
    Helper function to calculate approximate vascular contrast.
    Useful for validating that green channel offers better contrast.

    Returns:
        Contrast value (higher = better for vascular analysis)
    """
    gray = to_grayscale(image)

    # Simple contrast calculation based on variance
    # Blood vessels have lower intensity than background
    mean_intensity = np.mean(gray)
    contrast = np.std(gray) / (mean_intensity + 1e-6)

    return contrast

# DOCUMENTAÇÃO DA MUDANÇA PARA O ARTIGO

def get_green_channel_justification() -> str:
    """
    Returns scientific justification for using green channel.
    Useful for including in paper methodology.
    """
    return """
    GREEN CHANNEL USAGE JUSTIFICATION:

    Scientific basis: "retinal components are more intense in the fundus
    image's green channel" (Interactive Blood Vessel Segmentation from
    Retinal Fundus Image Based on Canny Edge Detector, 2021, PMC8512020)

    Impact on metrics:
    - Vessel Clarity Index: More effective Frangi filter response
    - Microaneurysm Visibility: Enhanced detection sensitivity
    - Edge Preservation: Better defined vascular boundaries
    """
