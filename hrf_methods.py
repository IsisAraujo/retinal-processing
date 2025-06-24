"""
Implementation of model-based illumination correction methods
"""

import cv2
import numpy as np
from typing import List, Dict
from hrf_core import IlluminationMethod
from hrf_utils import calculate_gaussian_kernel_size, normalize_reflectance, convert_to_float_and_log

class CLAHEMethod(IlluminationMethod):
    """
    Contrast Limited Adaptive Histogram Equalization
    """

    def __init__(self):
        super().__init__("CLAHE")

    def get_default_parameters(self) -> Dict:
        return {
            'clip_limit': 3.0,
            'tile_grid_size': (8, 8)
        }

    def process(self, image: np.ndarray, clip_limit: float = 3.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:  # Alterado default para (8, 8)
        # Convert to HSV color space for processing (alterado de LAB para HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # Apply CLAHE to value channel only (alterado de l_channel para v_channel)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        v_channel = clahe.apply(v_channel)

        # Merge and convert back (alterado de LAB para HSV)
        hsv = cv2.merge([h_channel, s_channel, v_channel])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class SingleScaleRetinex(IlluminationMethod):
    """
    Single Scale Retinex (SSR) implementation
    """

    def __init__(self):
        super().__init__("SSR")

    def get_default_parameters(self) -> Dict:
        return {
            'sigma': 120,  # Alterado de 250 para 120
            'gain': 1.2,   # Adicionado parâmetro gain
            'offset': 0.1  # Adicionado parâmetro offset
        }

    def process(self, image: np.ndarray, sigma: float = 120,
                gain: float = 1.2, offset: float = 0.1) -> np.ndarray:  # Alterados os defaults
        # Convert to float for precision and apply log
        log_img_float = convert_to_float_and_log(image)

        # Gaussian filtering for illumination estimation
        kernel_size = calculate_gaussian_kernel_size(sigma)
        illumination = cv2.GaussianBlur(image.astype(np.float64) + 1.0, (kernel_size, kernel_size), sigma)
        log_illumination = np.log(illumination + 1e-6)

        # Compute reflectance in log domain
        reflectance = log_img_float - log_illumination

        # Apply gain and offset (alteração na aplicação dos novos parâmetros)
        reflectance = gain * reflectance + offset

        # Normalize using percentile-based scaling
        reflectance = normalize_reflectance(reflectance)

        return (reflectance * 255).astype(np.uint8)

class MultiScaleRetinex(IlluminationMethod):
    """
    Multi-Scale Retinex (MSR) implementation

    """

    def __init__(self):
        super().__init__("MSR")

    def get_default_parameters(self) -> Dict:
        return {
            'sigmas': [15, 80, 200],  # Alterado de [15, 80, 250] para [15, 80, 200]
            'gain': 1.5,              # Adicionado parâmetro gain
            'offset': 0.05            # Adicionado parâmetro offset
        }

    def process(self, image: np.ndarray, sigmas: List[float] = None,
                gain: float = 1.5, offset: float = 0.05) -> np.ndarray:  # Adicionados novos parâmetros
        if sigmas is None:
            sigmas = self.get_default_parameters()['sigmas']

        log_img_float = convert_to_float_and_log(image)
        msr_output = np.zeros_like(log_img_float)

        for sigma in sigmas:
            kernel_size = calculate_gaussian_kernel_size(sigma)
            illumination = cv2.GaussianBlur(image.astype(np.float64) + 1.0, (kernel_size, kernel_size), sigma)
            log_illumination = np.log(illumination + 1e-6)
            reflectance = log_img_float - log_illumination
            msr_output += reflectance

        msr_output /= len(sigmas)

        # Apply gain and offset (alteração na aplicação dos novos parâmetros)
        msr_output = gain * msr_output + offset

        # Normalize
        msr_output = normalize_reflectance(msr_output)

        return (msr_output * 255).astype(np.uint8)

class MultiScaleRetinexColorRestoration(IlluminationMethod):
    """
    MSRCR implementation with color restoration
    """

    def __init__(self):
        super().__init__("MSRCR")

    def get_default_parameters(self) -> Dict:
        return {
            'sigmas': [15, 80, 250],
            'alpha': 125,
            'beta': 46,
            'gain': 2.0,
            'offset': 30,
            'restoration_factor': 125, # Novo parâmetro
            'color_gain': 2.5         # Novo parâmetro
        }

    def process(self, image: np.ndarray, sigmas: List[float] = None,
                alpha: float = 125, beta: float = 46,
                gain: float = 2.0, offset: float = 30,
                restoration_factor: float = 125, color_gain: float = 2.5) -> np.ndarray:  # Adicionados novos parâmetros
        if sigmas is None:
            sigmas = self.get_default_parameters()['sigmas']

        img_float = image.astype(np.float64) + 1.0
        log_img_float = np.log(img_float)

        # Multi-scale retinex
        msr_output = np.zeros_like(log_img_float)
        for sigma in sigmas:
            kernel_size = calculate_gaussian_kernel_size(sigma)
            illumination = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
            reflectance = log_img_float - np.log(illumination + 1e-6)
            msr_output += reflectance

        msr_output /= len(sigmas)

        # Color restoration (utilização dos novos parâmetros)
        img_sum = np.sum(img_float, axis=2, keepdims=True)
        color_restoration = beta * (np.log(restoration_factor * img_float) - np.log(img_sum + 1e-6))

        # Combine MSR with color restoration (aplicação do color_gain)
        msrcr = gain * (msr_output * color_restoration * color_gain + offset)

        # Clip to valid range
        msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)

        return msrcr
