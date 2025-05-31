import cv2
import numpy as np
from skimage import measure, feature, filters
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy import ndimage
from typing import Dict, Tuple, List
import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

class RetinalIQAMetrics:
    """Métricas especializadas para IQA em imagens retinianas"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.weights = {
            'vessel_clarity': 0.25,
            'contrast_uniformity': 0.20,
            'detail_preservation': 0.20,
            'artifact_penalty': 0.15,
            'global_quality': 0.20
        }

    def calculate_comprehensive_metrics(self, original: np.ndarray,
                                      enhanced: np.ndarray) -> Dict[str, float]:
        """Calcula suite completa de métricas IQA"""
        metrics = {}

        # Calcular todas as métricas
        metrics.update(self._assess_vessel_clarity(original, enhanced))
        metrics.update(self._analyze_contrast_distribution(original, enhanced))
        metrics.update(self._assess_detail_preservation(original, enhanced))
        metrics.update(self._detect_artifacts(original, enhanced))
        metrics.update(self._assess_perceptual_quality(original, enhanced))

        # Métricas finais
        metrics['clinical_relevance_score'] = self._calculate_clinical_relevance(metrics)
        metrics['enhancement_effective'] = self._make_enhancement_decision(metrics)
        metrics['confidence_score'] = self._calculate_confidence(metrics)

        return metrics

    def _assess_vessel_clarity(self, original: np.ndarray,
                              enhanced: np.ndarray) -> Dict[str, float]:
        """Avalia clareza vascular"""
        def vessel_metrics(img):
            if self.device != 'cpu':
                img_tensor = torch.from_numpy(img).float().to(self.device)
                sigmas = torch.arange(1, 4, 0.5, device=self.device)
                vessel_response = self._frangi_filter_torch(img_tensor, sigmas).cpu().numpy()
            else:
                sigmas = np.arange(1, 4, 0.5)
                vessel_response = filters.frangi(img, sigmas=sigmas, black_ridges=True)

            return {
                'strength': np.mean(vessel_response),
                'connectivity': measure.label(vessel_response > 0.1).max(),
                'contrast': np.std(vessel_response)
            }

        vessel_orig = vessel_metrics(original)
        vessel_enh = vessel_metrics(enhanced)

        return {
            'vessel_clarity_gain': vessel_enh['strength'] / max(vessel_orig['strength'], 1e-6),
            'vessel_connectivity_ratio': vessel_enh['connectivity'] / max(vessel_orig['connectivity'], 1),
            'vessel_contrast_improvement': (vessel_enh['contrast'] - vessel_orig['contrast']) /
                                         max(vessel_orig['contrast'], 1e-6)
        }

    def _frangi_filter_torch(self, img_tensor, sigmas):
        """Filtro Frangi simplificado usando PyTorch"""
        result = torch.zeros_like(img_tensor)

        for sigma in sigmas:
            kernel_size = int(4 * sigma + 0.5) * 2 + 1
            kernel = self._gaussian_kernel_2d(kernel_size, sigma)

            smoothed = F.conv2d(
                img_tensor.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze()

            vesselness = torch.abs(smoothed)
            result = torch.max(result, vesselness)

        return result

    def _gaussian_kernel_2d(self, kernel_size, sigma):
        """Kernel gaussiano 2D"""
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        return kernel_2d / kernel_2d.sum()

    def _analyze_contrast_distribution(self, original: np.ndarray,
                                     enhanced: np.ndarray) -> Dict[str, float]:
        """Análise da distribuição de contraste"""
        if self.device != 'cpu':
            orig_tensor = torch.from_numpy(original).float().to(self.device)
            enh_tensor = torch.from_numpy(enhanced).float().to(self.device)

            contrast_orig = self._local_contrast_torch(orig_tensor).cpu().numpy()
            contrast_enh = self._local_contrast_torch(enh_tensor).cpu().numpy()
        else:
            contrast_orig = self._local_contrast_numpy(original)
            contrast_enh = self._local_contrast_numpy(enhanced)

        contrast_improvement = contrast_enh - contrast_orig
        uniformity = 1.0 / (1.0 + np.std(contrast_improvement))
        over_enhanced = np.sum(contrast_improvement > 2.0) / len(contrast_improvement)

        return {
            'contrast_uniformity': uniformity,
            'contrast_improvement_std': np.std(contrast_improvement),
            'over_enhancement_ratio': over_enhanced,
            'contrast_gain_distribution': np.mean(contrast_improvement)
        }

    def _local_contrast_torch(self, img_tensor, window_size=64):
        """Análise de contraste local usando PyTorch"""
        h, w = img_tensor.shape
        contrasts = []

        for i in range(0, h-window_size, window_size//2):
            for j in range(0, w-window_size, window_size//2):
                patch = img_tensor[i:i+window_size, j:j+window_size]
                if patch.numel() > 0:
                    mean_val = patch.mean()
                    std_val = patch.std()
                    contrast = std_val / (mean_val + 1e-6)
                    contrasts.append(contrast.item())

        return torch.tensor(contrasts, device=self.device)

    def _local_contrast_numpy(self, img, window_size=64):
        """Análise de contraste local usando NumPy"""
        h, w = img.shape
        contrasts = []

        for i in range(0, h-window_size, window_size//2):
            for j in range(0, w-window_size, window_size//2):
                patch = img[i:i+window_size, j:j+window_size]
                if patch.size > 0:
                    contrast = np.std(patch) / (np.mean(patch) + 1e-6)
                    contrasts.append(contrast)

        return np.array(contrasts)

    def _assess_detail_preservation(self, original: np.ndarray,
                                  enhanced: np.ndarray) -> Dict[str, float]:
        """Avalia preservação de detalhes"""
        # Edge preservation
        edges_orig = feature.canny(original, sigma=1.0)
        edges_enh = feature.canny(enhanced, sigma=1.0)
        edge_preservation = np.sum(edges_orig & edges_enh) / max(np.sum(edges_orig), 1)

        # Texture preservation
        def texture_complexity(img):
            lbp = feature.local_binary_pattern(img, P=8, R=1, method='uniform')
            return np.std(lbp)

        texture_orig = texture_complexity(original)
        texture_enh = texture_complexity(enhanced)
        texture_preservation = texture_enh / max(texture_orig, 1e-6)

        # High-frequency content
        def high_freq_content(img):
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            h, w = magnitude.shape

            # Máscara para altas frequências
            y, x = np.ogrid[:h, :w]
            center_h, center_w = h//2, w//2
            radius = min(h, w) // 4
            mask = (x - center_w)**2 + (y - center_h)**2 > radius**2

            return np.mean(magnitude * mask)

        hf_orig = high_freq_content(original)
        hf_enh = high_freq_content(enhanced)
        hf_preservation = hf_enh / max(hf_orig, 1e-6)

        return {
            'edge_preservation': edge_preservation,
            'texture_preservation': texture_preservation,
            'high_frequency_preservation': hf_preservation,
            'detail_preservation_score': np.mean([
                edge_preservation, texture_preservation, min(hf_preservation, 2.0)
            ])
        }

    def _detect_artifacts(self, original: np.ndarray,
                         enhanced: np.ndarray) -> Dict[str, float]:
        """Detecta artefatos introduzidos pelo processamento"""
        diff = cv2.absdiff(enhanced, original)

        # Blocking artifacts
        def detect_blocking(img, tile_size=8):
            h, w = img.shape
            edges = []

            # Bordas horizontais
            for i in range(tile_size, h, tile_size):
                if i < h-1:
                    edge_strength = np.mean(np.abs(np.diff(img[i-1:i+1, :], axis=0)))
                    edges.append(edge_strength)

            # Bordas verticais
            for j in range(tile_size, w, tile_size):
                if j < w-1:
                    edge_strength = np.mean(np.abs(np.diff(img[:, j-1:j+1], axis=1)))
                    edges.append(edge_strength)

            return np.mean(edges) if edges else 0

        blocking_score = detect_blocking(diff)

        # Over-sharpening
        laplacian_var = cv2.Laplacian(enhanced, cv2.CV_64F).var()
        laplacian_var_orig = cv2.Laplacian(original, cv2.CV_64F).var()
        over_sharpening = max(0, (laplacian_var / max(laplacian_var_orig, 1e-6)) - 1.5)

        # Halo artifacts
        halo_strength = np.std(cv2.GaussianBlur(diff, (15, 15), 0))

        return {
            'blocking_artifacts': blocking_score,
            'over_sharpening_penalty': over_sharpening,
            'halo_artifacts': halo_strength,
            'total_artifact_score': blocking_score + over_sharpening + halo_strength * 0.1
        }

    def _assess_perceptual_quality(self, original: np.ndarray,
                                  enhanced: np.ndarray) -> Dict[str, float]:
        """Avalia qualidade perceptual"""
        orig_float = original.astype(np.float32)
        enh_float = enhanced.astype(np.float32)
        data_range = 255.0

        # SSIM e PSNR
        ssim_score = ssim(orig_float, enh_float, data_range=data_range)
        psnr_score = psnr(orig_float, enh_float, data_range=data_range)

        # Similaridade de gradiente
        def calculate_gradient(img):
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2)

        grad_orig = calculate_gradient(orig_float)
        grad_enh = calculate_gradient(enh_float)
        gradient_similarity = ssim(grad_orig, grad_enh,
                                 data_range=grad_enh.max() - grad_enh.min())

        return {
            'ssim_score': ssim_score,
            'psnr_score': psnr_score,
            'gradient_similarity': gradient_similarity,
            'perceptual_quality_score': np.mean([ssim_score, psnr_score, gradient_similarity])
        }

    def _calculate_clinical_relevance(self, metrics: Dict[str, float]) -> float:
        """Score composto baseado na relevância clínica"""
        components = {
            'vessel_clarity_gain': 0.30,
            'detail_preservation_score': 0.25,
            'contrast_uniformity': 0.20,
            'perceptual_quality_score': 0.15,
            'total_artifact_score': -0.10
        }

        score = 0.0
        for metric, weight in components.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'total_artifact_score':
                    score += weight * min(value, 1.0)
                else:
                    score += weight * min(value, 2.0)

        return max(0.0, min(1.0, score))

    def _make_enhancement_decision(self, metrics: Dict[str, float]) -> bool:
        """Decisão binária: enhancement efetivo ou não"""
        criteria = {
            'vessel_clarity_gain': 1.1,
            'detail_preservation_score': 0.8,
            'clinical_relevance_score': 0.6,
            'total_artifact_score': 0.3
        }

        passed = 0
        for metric, threshold in criteria.items():
            if metric in metrics:
                if metric == 'total_artifact_score':
                    if metrics[metric] <= threshold:
                        passed += 1
                else:
                    if metrics[metric] >= threshold:
                        passed += 1

        return (passed / len(criteria)) >= 0.75

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calcula confiança na decisão"""
        key_metrics = [
            'vessel_clarity_gain', 'detail_preservation_score',
            'clinical_relevance_score', 'perceptual_quality_score'
        ]

        values = [metrics.get(m, 0.5) for m in key_metrics]
        variance = np.var(values)
        consistency = 1.0 / (1.0 + variance)
        mean_quality = np.mean(values)

        return min(1.0, max(0.0, 0.7 * consistency + 0.3 * mean_quality))

class GroundTruthGenerator:
    """Gera ground truth sintético para treinamento ViT"""

    def __init__(self):
        self.degradation_types = [
            'blur', 'noise', 'brightness', 'contrast_loss',
            'compression', 'tile_artifacts'
        ]

    def generate_training_pairs(self, original: np.ndarray,
                              n_positive: int = 5,
                              n_negative: int = 5) -> List[Tuple[np.ndarray, int, str]]:
        """Gera pares de treinamento com labels conhecidos"""
        pairs = []

        # Exemplos positivos (melhorias)
        for i in range(n_positive):
            enhanced = self._apply_beneficial_enhancement(original, 0.3 + i*0.2)
            pairs.append((enhanced, 1, f"beneficial_{i}"))

        # Exemplos negativos (degradações)
        for i in range(n_negative):
            degraded = self._apply_degradation(original, 0.2 + i*0.15)
            pairs.append((degraded, 0, f"degradation_{i}"))

        return pairs

    def _apply_beneficial_enhancement(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Aplica enhancement benéfico controlado"""
        # CLAHE otimizado
        clahe = cv2.createCLAHE(clipLimit=2.0 + strength, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        # Leve sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * (strength * 0.1)
        kernel[1,1] = 1 + strength * 0.8
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _apply_degradation(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Aplica degradação controlada"""
        degraded = img.copy().astype(np.float32)
        degradation_type = np.random.choice(self.degradation_types)

        if degradation_type == 'blur':
            ksize = int(3 + strength * 4) | 1
            degraded = cv2.GaussianBlur(degraded, (ksize, ksize), 0)
        elif degradation_type == 'noise':
            noise = np.random.normal(0, strength * 20, img.shape)
            degraded = degraded + noise
        elif degradation_type == 'brightness':
            degraded = degraded * (0.5 + strength * 0.3)
        elif degradation_type == 'contrast_loss':
            mean_val = np.mean(degraded)
            degraded = mean_val + (degraded - mean_val) * (1 - strength)
        elif degradation_type == 'compression':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(90 - strength * 40)]
            _, encimg = cv2.imencode('.jpg', degraded.astype(np.uint8), encode_param)
            degraded = cv2.imdecode(encimg, 0).astype(np.float32)
        elif degradation_type == 'tile_artifacts':
            tile_size = 16
            for i in range(0, img.shape[0], tile_size):
                for j in range(0, img.shape[1], tile_size):
                    if np.random.random() < strength:
                        degraded[i:i+tile_size, j:j+tile_size] *= (0.8 + np.random.random() * 0.4)

        return np.clip(degraded, 0, 255).astype(np.uint8)
