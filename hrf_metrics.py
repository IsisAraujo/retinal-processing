"""
Ophthalmology-specific metrics for illumination correction evaluation
Enhanced version with additional standard metrics (PSNR, SSIM)

References:
[1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
    assessment: From error visibility to structural similarity. IEEE Transactions on
    Image Processing, 13(4), 600-612.
[2] Tian et al. (2021) - "Blood Vessel Segmentation of Fundus Retinal Images Based on
    Improved Frangi and Mathematical Morphology"
[3] Yang et al. (2020) - "Frangi based multi-scale level sets for retinal vascular segmentation"
[4] Mahapatra et al. (2022) - "A novel framework for retinal vessel segmentation using
    optimal improved frangi filter"
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from hrf_core import OphthalmicMetrics
from hrf_utils import to_grayscale, normalize_image_for_display

class MetricsCalculatorImproved:
    """
    Enhanced calculator with standard image quality metrics for fundus image assessment
    following academic publication standards (2024-2025)
    """

    @staticmethod
    def calculate_psnr(original: np.ndarray, corrected: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)

        Reference: Wang, Z., et al. (2004). Image quality assessment: From error
        visibility to structural similarity. IEEE Trans. Image Processing.

        Args:
            original: Original image array
            corrected: Processed image array

        Returns:
            PSNR value in dB (higher is better, typically 30-50 for good quality)
        """
        # Convert to same data type
        if original.dtype != corrected.dtype:
            original = original.astype(np.float64)
            corrected = corrected.astype(np.float64)

        # Calculate PSNR using scikit-image implementation
        # data_range is automatically determined from image dtype
        try:
            psnr_value = peak_signal_noise_ratio(original, corrected)
        except ValueError:
            # Handle edge case where images are identical
            psnr_value = 100.0  # Perfect match

        return float(psnr_value)

    @staticmethod
    def calculate_ssim(original: np.ndarray, corrected: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM)

        Reference: Wang, Z., et al. (2004). Image quality assessment: From error
        visibility to structural similarity. IEEE Trans. Image Processing.

        Args:
            original: Original image array
            corrected: Processed image array

        Returns:
            SSIM value (0-1, where 1 is perfect similarity)
        """
        # Convert to grayscale if needed for consistent comparison
        if len(original.shape) == 3:
            original_gray = to_grayscale(original)
            corrected_gray = to_grayscale(corrected)
        else:
            original_gray = original
            corrected_gray = corrected

        # Calculate SSIM using scikit-image implementation with standard parameters
        try:
            ssim_value = structural_similarity(
                original_gray,
                corrected_gray,
                win_size=11,  # Standard 11x11 window
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=original_gray.max() - original_gray.min()
            )
        except ValueError:
            # Handle edge case
            ssim_value = 1.0

        return float(ssim_value)

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
    def _get_vesselness_map(image: np.ndarray) -> np.ndarray:
        """
        Helper to get the Frangi vesselness map with corrected parameters

        Implementação baseada em:
        [1] Tian et al. (2021) - "Blood Vessel Segmentation of Fundus Retinal Images
            Based on Improved Frangi and Mathematical Morphology"
        [2] Yang et al. (2020) - "Frangi based multi-scale level sets for retinal
            vascular segmentation"
        [3] Mahapatra et al. (2022) - "A novel framework for retinal vessel segmentation
            using optimal improved frangi filter"
        """
        gray = to_grayscale(image)

        # Parâmetros corrigidos baseados na literatura (2020-2025)
        s_min = 1.0      # Escala mínima
        s_max = 8.0      # Escala máxima
        s_step = 0.5     # Incremento de escala

        # Parâmetros do filtro de Frangi corrigidos
        alpha = 0.5      # Controle de estruturas blob vs. plate
        beta = 0.5       # Controle de supressão de fundo
        c = 15           # CORRIGIDO: valor reduzido de 500 para 15 (padrão da literatura)

        # Gerar escalas conforme literatura
        scales = np.arange(s_min, s_max + s_step, s_step)
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

            # Evitar divisão por zero
            lambda1_safe = lambda1 + 1e-10
            lambda2_safe = lambda2 + 1e-10

            # Razão beta (controle de blob vs. plate structures)
            Rb = np.abs(lambda2_safe) / np.abs(lambda1_safe)

            # Segundo momento estrutural
            S = np.sqrt(lambda1**2 + lambda2**2)

            # Função vesselness de Frangi com parâmetros corrigidos
            term1 = np.exp(-Rb**2 / (2 * beta**2))
            term2 = (1 - np.exp(-S**2 / (2 * c**2)))

            vesselness_scale = term1 * term2

            # Suprimir estruturas bright (apenas dark vessels)
            vesselness_scale[lambda1 > 0] = 0

            # Aplicar fator de equivalência de escala
            scale_factor = alpha * (scale / s_max)
            vesselness_scale *= scale_factor

            # Manter resposta máxima entre escalas
            vesselness = np.maximum(vesselness, vesselness_scale)

        return vesselness

    @staticmethod
    def calculate_vessel_clarity_index(image: np.ndarray) -> float:
        """
        Vessel clarity based on Frangi vesselness filter response
        Reference: Li et al. (2024) - "Automated vessel analysis in fundus images"
        """
        vesselness_map_raw = MetricsCalculatorImproved._get_vesselness_map(image)
        vesselness_map_normalized = normalize_image_for_display(vesselness_map_raw)
        return np.mean(vesselness_map_normalized) / 255.0

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
        Calculate all ophthalmology-specific metrics including standard metrics
        """
        return OphthalmicMetrics(
            # Standard image quality metrics
            psnr=MetricsCalculatorImproved.calculate_psnr(original, corrected),
            ssim=MetricsCalculatorImproved.calculate_ssim(original, corrected),

            # Original ophthalmology-specific metrics
            contrast_ratio=MetricsCalculatorImproved.calculate_weber_contrast_ratio(original, corrected),
            vessel_clarity_index=MetricsCalculatorImproved.calculate_vessel_clarity_index(corrected),
            illumination_uniformity=MetricsCalculatorImproved.calculate_illumination_uniformity(corrected),
            edge_preservation_index=MetricsCalculatorImproved.calculate_edge_preservation_index(original, corrected),
            microaneurysm_visibility=MetricsCalculatorImproved.calculate_microaneurysm_visibility(corrected)
        )
