"""
HRF Illumination Correction Analysis - Enhanced Core Module
Academic implementation for comparative analysis of model-based illumination correction methods
Enhanced version with additional metrics and improved statistical analysis

References:
[1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
    assessment: From error visibility to structural similarity. IEEE Transactions on
    Image Processing, 13(4), 600-612.
[2] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B (Methodological), 57(1), 289-300.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


HRF_RESOLUTION = (3504, 2336)
MIN_VESSEL_WIDTH = 3
OPTIC_DISC_RADIUS = 150

@dataclass
class OphthalmicMetrics:
    """
    Enhanced ophthalmology-specific metrics for illumination correction evaluation
    Now includes standard image quality metrics (PSNR, SSIM) alongside domain-specific metrics
    """
    # Standard image quality metrics (IEEE standards)
    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    ssim: float  # Structural Similarity Index (0-1)

    # Domain-specific ophthalmology metrics
    contrast_ratio: float  # Weber contrast ratio
    vessel_clarity_index: float  # Based on Frangi vesselness filter response
    illumination_uniformity: float  # Coefficient of variation of local luminance
    edge_preservation_index: float  # Gradient magnitude preservation ratio
    microaneurysm_visibility: float  # Local contrast in pathology regions

    def to_dict(self) -> Dict[str, float]:
        return {
            # Standard metrics
            'psnr': self.psnr,
            'ssim': self.ssim,

            # Domain-specific metrics
            'contrast_ratio': self.contrast_ratio,
            'vessel_clarity_index': self.vessel_clarity_index,
            'illumination_uniformity': self.illumination_uniformity,
            'edge_preservation_index': self.edge_preservation_index,
            'microaneurysm_visibility': self.microaneurysm_visibility
        }

    def get_standard_metrics(self) -> Dict[str, float]:
        """Get only standard image quality metrics"""
        return {
            'psnr': self.psnr,
            'ssim': self.ssim
        }

    def get_domain_metrics(self) -> Dict[str, float]:
        """Get only domain-specific ophthalmology metrics"""
        return {
            'contrast_ratio': self.contrast_ratio,
            'vessel_clarity_index': self.vessel_clarity_index,
            'illumination_uniformity': self.illumination_uniformity,
            'edge_preservation_index': self.edge_preservation_index,
            'microaneurysm_visibility': self.microaneurysm_visibility
        }

@dataclass
class ProcessingResult:
    """Container for processing results with enhanced metadata"""
    original_image: np.ndarray
    corrected_image: np.ndarray
    method_name: str
    metrics: OphthalmicMetrics
    processing_time_ms: float

    # Additional metadata for analysis
    image_id: str = ""
    dataset_category: str = ""  # 'healthy', 'diabetic_retinopathy', 'glaucoma'

@dataclass
class StatisticalResults:
    """
    Container for statistical analysis results following current standards
    """
    # Test results
    test_name: str
    statistic: float
    p_value: float
    effect_size: float  # Cohen's d or eta-squared

    # Multiple comparison corrections
    bonferroni_corrected_p: float
    fdr_corrected_p: float  # Benjamini-Hochberg correction

    # Statistical power analysis
    observed_power: float

    # Confidence intervals
    ci_lower: float
    ci_upper: float

    def is_significant_bonferroni(self, alpha: float = 0.05) -> bool:
        """Check significance with Bonferroni correction"""
        return self.bonferroni_corrected_p < alpha

    def is_significant_fdr(self, alpha: float = 0.05) -> bool:
        """Check significance with FDR correction (recommended)"""
        return self.fdr_corrected_p < alpha

class IlluminationMethod:
    """Base class for illumination correction methods"""

    def __init__(self, name: str):
        self.name = name

    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement process method")

    def get_default_parameters(self) -> Dict:
        raise NotImplementedError("Subclasses must implement get_default_parameters")

    def get_parameter_references(self) -> Dict[str, str]:
        """Get literature references for parameter choices"""
        raise NotImplementedError("Subclasses must implement get_parameter_references")
