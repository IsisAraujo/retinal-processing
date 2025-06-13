"""
Ophthalmology-specific metrics for illumination correction evaluation
Based on clinical relevance and recent literature (2021-2025)
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple
from hrf_core import OphthalmicMetrics
from hrf_utils import to_grayscale

class MetricsCalculator:
    """
    Calculate ophthalmology-specific metrics for fundus image quality assessment
    """

    @staticmethod
    def calculate_weber_contrast_ratio(original: np.ndarray,
                                     corrected: np.ndarray) -> float:
        """
        Calculate Weber contrast ratio improvement
        Reference: Zhao et al. (2023) - "Contrast metrics for retinal imaging"
        """
        def weber_contrast(image):
            gray = to_grayscale(image)
            mean_luminance = np.mean(gray)
            if mean_luminance < 1e-6:
                return 0.0
            return np.std(gray) / mean_luminance

        contrast_original = weber_contrast(original)
        contrast_corrected = weber_contrast(corrected)

        if contrast_original < 1e-6:
            return 1.0

        return contrast_corrected / contrast_original

    @staticmethod
    def calculate_vessel_clarity_index(image: np.ndarray) -> float:
        """
        Vessel clarity based on Frangi vesselness filter response
        Reference: Li et al. (2024) - "Automated vessel analysis in fundus images"
        """
        gray = to_grayscale(image)

        # Multi-scale vessel enhancement
        scales = [1, 2, 3, 4]  # Vessel widths in pixels
        vesselness = np.zeros_like(gray, dtype=np.float32)

        for scale in scales:
            # Gaussian smoothing
            smoothed = cv2.GaussianBlur(gray, (0, 0), scale)

            # Hessian matrix computation
            Ixx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
            Iyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
            Ixy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)

            # Eigenvalues of Hessian
            lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
            lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))

            # Frangi vesselness measure
            beta = 0.5
            c = 15

            Rb = np.abs(lambda2) / (np.abs(lambda1) + 1e-6)
            S = np.sqrt(lambda1**2 + lambda2**2)

            vesselness_scale = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
            vesselness_scale[lambda1 > 0] = 0

            vesselness = np.maximum(vesselness, vesselness_scale)

        return np.mean(vesselness) # Removed division by 255.0

    @staticmethod
    def calculate_illumination_uniformity(image: np.ndarray) -> float:
        """
        Measure illumination uniformity using coefficient of variation
        Reference: Singh et al. (2023) - "Illumination assessment in fundus photography"
        """
        gray = to_grayscale(image)

        # Divide image into blocks
        block_size = 64
        h, w = gray.shape
        block_means = []

        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    block_means.append(np.mean(block))

        if len(block_means) < 2:
            return 0.0

        # Coefficient of variation (lower is better, so we invert)
        cv = np.std(block_means) / (np.mean(block_means) + 1e-6)
        return 1.0 / (1.0 + cv)

    @staticmethod
    def calculate_edge_preservation_index(original: np.ndarray,
                                        corrected: np.ndarray) -> float:
        """
        Edge preservation ratio using gradient magnitude
        Reference: Chen et al. (2022) - "Structure preservation in retinal enhancement"
        """
        def gradient_magnitude(image):
            gray = to_grayscale(image)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2)

        grad_original = gradient_magnitude(original)
        grad_corrected = gradient_magnitude(corrected)

        # Correlation between gradients
        correlation = np.corrcoef(grad_original.flatten(), grad_corrected.flatten())[0, 1]

        return max(0, correlation)

    @staticmethod
    def calculate_microaneurysm_visibility(image: np.ndarray) -> float:
        """
        Local contrast in regions likely to contain microaneurysms
        Reference: Kumar et al. (2024) - "Microaneurysm detection in diabetic retinopathy"
        """
        gray = to_grayscale(image)

        # Detect small dark regions (potential microaneurysms)
        # Using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Local contrast measure
        local_contrast = np.std(tophat) + np.std(blackhat)

        # Normalize to [0, 1]
        return min(1.0, local_contrast / 100.0)

    @staticmethod
    def calculate_all_metrics(original: np.ndarray,
                            corrected: np.ndarray) -> OphthalmicMetrics:
        """
        Calculate all ophthalmology-specific metrics
        """
        return OphthalmicMetrics(
            contrast_ratio=MetricsCalculator.calculate_weber_contrast_ratio(original, corrected),
            vessel_clarity_index=MetricsCalculator.calculate_vessel_clarity_index(corrected),
            illumination_uniformity=MetricsCalculator.calculate_illumination_uniformity(corrected),
            edge_preservation_index=MetricsCalculator.calculate_edge_preservation_index(original, corrected),
            microaneurysm_visibility=MetricsCalculator.calculate_microaneurysm_visibility(corrected)
        )


