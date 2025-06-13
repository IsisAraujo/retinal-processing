"""
Módulo de Pré-processamento HRF
Implementa métricas específicas para oftalmologia e correção de iluminação
"""

import cv2
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Configurar logging para debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OphthalmologyMetrics:
    """
    Métricas específicas para avaliação oftalmológica
    Substitui PSNR/SSIM inadequados para correção de iluminação
    """
    contrast_enhancement_ratio: float
    vessel_visibility_index: float
    local_uniformity_score: float
    detail_preservation_ratio: float
    pathology_detectability_score: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário para facilitar análise"""
        return {
            'contrast_enhancement': self.contrast_enhancement_ratio,
            'vessel_visibility': self.vessel_visibility_index,
            'local_uniformity': self.local_uniformity_score,
            'detail_preservation': self.detail_preservation_ratio,
            'pathology_detectability': self.pathology_detectability_score,
            'processing_time_ms': self.processing_time_ms
        }

class HRFImageLoader:
    """
    Carregador otimizado e validado para dataset HRF
    Implementa verificações rigorosas de qualidade
    """

    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png')
    MIN_RESOLUTION = (1000, 1000)  # Resolução mínima para imagens HRF válidas

    @staticmethod
    def load_hrf_dataset(image_dir: str) -> Dict[str, np.ndarray]:
        """
        Carrega e valida imagens do dataset HRF com verificações rigorosas

        Args:
            image_dir: Diretório contendo imagens HRF

        Returns:
            Dicionário {nome_imagem: array_numpy}
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {image_dir}")

        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(HRFImageLoader.SUPPORTED_FORMATS)
        ]

        if not image_paths:
            raise ValueError(f"Nenhuma imagem válida encontrada em {image_dir}")

        images = {}
        failed_loads = []

        for path in sorted(image_paths):
            img_name = os.path.basename(path)
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)

                if img is None:
                    failed_loads.append(f"{img_name}: Falha na leitura")
                    continue

                # Validar resolução mínima
                if img.shape[0] < HRFImageLoader.MIN_RESOLUTION[0] or \
                   img.shape[1] < HRFImageLoader.MIN_RESOLUTION[1]:
                    failed_loads.append(f"{img_name}: Resolução insuficiente {img.shape}")
                    continue

                # Validar se não é imagem corrompida
                if np.std(img) < 5:  # Imagem muito uniforme, possivelmente corrompida
                    failed_loads.append(f"{img_name}: Imagem suspeita (baixa variância)")
                    continue

                images[img_name] = img
                logger.info(f"✓ Carregada: {img_name} - Shape: {img.shape}")

            except Exception as e:
                failed_loads.append(f"{img_name}: {str(e)}")

        if failed_loads:
            logger.warning(f"Falhas no carregamento: {failed_loads}")

        logger.info(f"Dataset carregado: {len(images)} imagens válidas")
        return images

class IlluminationCorrectionMethods:
    """
    Implementação otimizada dos métodos de correção de iluminação
    Com parâmetros otimizados para imagens de fundo de olho
    """

    @staticmethod
    def apply_clahe_optimized(image: np.ndarray,
                            clip_limit: float = 3.0,
                            tile_size: int = 16) -> np.ndarray:
        """
        CLAHE otimizado para imagens de fundo de olho
        Parâmetros ajustados para melhor performance em oftalmologia
        """
        # Converter para LAB - melhor para preservar informações de cor
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # CLAHE com parâmetros otimizados para fundoscopia
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        enhanced_l = clahe.apply(l_channel)

        # Reconstituir imagem
        lab[:, :, 0] = enhanced_l
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    @staticmethod
    def apply_ssr_optimized(image: np.ndarray, sigma: float = 250) -> np.ndarray:
        """
        Single Scale Retinex otimizado para detecção de patologias
        """
        # Converter para float64 para maior precisão
        img_float = image.astype(np.float64) / 255.0

        # Filtro Gaussiano otimizado
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        illumination = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)

        # Retinex com estabilização numérica
        epsilon = 1e-6
        reflectance = np.log(img_float + epsilon) - np.log(illumination + epsilon)

        # Normalização robusta (percentil-based)
        p1, p99 = np.percentile(reflectance, [1, 99])
        reflectance_norm = np.clip((reflectance - p1) / (p99 - p1), 0, 1)

        return (reflectance_norm * 255).astype(np.uint8)

    @staticmethod
    def apply_msr_optimized(image: np.ndarray,
                          sigmas: List[float] = None) -> np.ndarray:
        """
        Multi-Scale Retinex otimizado para oftalmologia
        """
        if sigmas is None:
            # Escalas otimizadas para estruturas do fundo de olho
            sigmas = [15, 80, 250]  # Detalhes finos, vasos médios, iluminação geral

        img_float = image.astype(np.float64) / 255.0
        retinex_sum = np.zeros_like(img_float)

        for sigma in sigmas:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            illumination = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
            epsilon = 1e-6
            reflectance = np.log(img_float + epsilon) - np.log(illumination + epsilon)
            retinex_sum += reflectance

        retinex_avg = retinex_sum / len(sigmas)

        # Normalização robusta
        p1, p99 = np.percentile(retinex_avg, [1, 99])
        retinex_norm = np.clip((retinex_avg - p1) / (p99 - p1), 0, 1)

        return (retinex_norm * 255).astype(np.uint8)

    @staticmethod
    def apply_msrcr_optimized(image: np.ndarray,
                            sigmas: List[float] = None,
                            alpha: float = 125,
                            beta: float = 46,
                            gain: float = 192,
                            offset: float = 30) -> np.ndarray:
        """
        MSRCR otimizado com restauração de cor para oftalmologia
        """
        if sigmas is None:
            sigmas = [15, 80, 250]

        img_float = image.astype(np.float64) + 1.0  # Evitar log(0)

        # Parte MSR
        retinex_sum = np.zeros_like(img_float)
        for sigma in sigmas:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            illumination = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
            epsilon = 1e-6
            reflectance = np.log(img_float) - np.log(illumination + epsilon)
            retinex_sum += reflectance

        retinex_avg = retinex_sum / len(sigmas)

        # Restauração de cor otimizada
        sum_channels = np.sum(img_float, axis=2, keepdims=True)
        epsilon = 1e-6
        color_restoration = beta * (
            np.log(alpha * img_float) - np.log(sum_channels + epsilon)
        )

        # Combinação final
        msrcr_output = gain * (retinex_avg * color_restoration + offset)

        # Normalização robusta
        msrcr_output = np.clip(msrcr_output, 0, 255).astype(np.uint8)

        return msrcr_output

class OphthalmologyMetricsCalculator:
    """
    Calculador de métricas específicas para oftalmologia
    Substitui métricas inadequadas (PSNR/SSIM) por métricas clinicamente relevantes
    """

    @staticmethod
    def calculate_contrast_enhancement_ratio(original: np.ndarray,
                                           enhanced: np.ndarray) -> float:
        """
        Razão de melhoria do contraste - métrica fundamental
        """
        def rms_contrast(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            return np.std(gray)

        contrast_original = rms_contrast(original)
        contrast_enhanced = rms_contrast(enhanced)

        return contrast_enhanced / (contrast_original + 1e-6)

    @staticmethod
    def calculate_vessel_visibility_index(image: np.ndarray) -> float:
        """
        Índice de visibilidade dos vasos sanguíneos
        Usa filtros específicos para detectar estruturas vasculares
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Filtro de realce de vasos (Frangi-like simplificado)
        # Usar gradientes direcionais para detectar estruturas lineares
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude do gradiente
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalizar e calcular média como proxy para visibilidade vascular
        vessel_score = np.mean(gradient_magnitude) / 255.0

        return vessel_score

    @staticmethod
    def calculate_local_uniformity_score(image: np.ndarray) -> float:
        """
        Score de uniformidade local da iluminação
        Menor valor = melhor correção de iluminação não-uniforme
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Dividir imagem em blocos e calcular variância da média dos blocos
        h, w = gray.shape
        block_size = 64
        block_means = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_means.append(np.mean(block))

        # Uniformidade = 1 / (variância das médias dos blocos + epsilon)
        uniformity = 1.0 / (np.std(block_means) + 1e-6)

        return uniformity

    @staticmethod
    def calculate_detail_preservation_ratio(original: np.ndarray,
                                          enhanced: np.ndarray) -> float:
        """
        Razão de preservação de detalhes finos
        Usa Laplaciano para detectar bordas/detalhes
        """
        def detail_measure(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return np.std(laplacian)

        detail_original = detail_measure(original)
        detail_enhanced = detail_measure(enhanced)

        # Preservação ideal = 1.0, > 1.0 = enhancement, < 1.0 = perda
        return detail_enhanced / (detail_original + 1e-6)

    @staticmethod
    def calculate_pathology_detectability_score(image: np.ndarray) -> float:
        """
        Score de detectabilidade de patologias
        Baseado em contraste local e detecção de anomalias
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Usar filtro morfológico para realçar estruturas pequenas (microaneurismas, etc.)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Score baseado na quantidade de estruturas pequenas detectadas
        pathology_score = np.mean(tophat) / 255.0

        return pathology_score

def process_single_image(image_name: str,
                        original_image: np.ndarray) -> Dict[str, any]:
    """
    Processa uma única imagem com todos os métodos e calcula métricas

    Returns:
        Dicionário com imagens processadas e métricas
    """
    methods = IlluminationCorrectionMethods()
    calculator = OphthalmologyMetricsCalculator()

    results = {
        'original': original_image,
        'processed_images': {},
        'metrics': {},
        'processing_times': {}
    }

    # Aplicar cada método
    method_configs = {
        'clahe': (methods.apply_clahe_optimized, {}),
        'ssr': (methods.apply_ssr_optimized, {}),
        'msr': (methods.apply_msr_optimized, {}),
        'msrcr': (methods.apply_msrcr_optimized, {})
    }

    for method_name, (method_func, kwargs) in method_configs.items():
        logger.info(f"  Aplicando {method_name.upper()}...")

        # Medir tempo de processamento
        start_time = time.time()
        processed_image = method_func(original_image.copy(), **kwargs)
        processing_time = (time.time() - start_time) * 1000  # em ms

        results['processed_images'][method_name] = processed_image
        results['processing_times'][method_name] = processing_time

        # Calcular métricas específicas para oftalmologia
        metrics = OphthalmologyMetrics(
            contrast_enhancement_ratio=calculator.calculate_contrast_enhancement_ratio(
                original_image, processed_image
            ),
            vessel_visibility_index=calculator.calculate_vessel_visibility_index(
                processed_image
            ),
            local_uniformity_score=calculator.calculate_local_uniformity_score(
                processed_image
            ),
            detail_preservation_ratio=calculator.calculate_detail_preservation_ratio(
                original_image, processed_image
            ),
            pathology_detectability_score=calculator.calculate_pathology_detectability_score(
                processed_image
            ),
            processing_time_ms=processing_time
        )

        results['metrics'][method_name] = metrics

        logger.info(f"    ✓ {method_name.upper()} - "
                   f"Contraste: {metrics.contrast_enhancement_ratio:.2f}x, "
                   f"Tempo: {processing_time:.1f}ms")

    return results

def main(dataset_path: str) -> Optional[Dict[str, any]]:
    """
    Função principal reestruturada para rigor acadêmico

    Args:
        dataset_path: Caminho para o dataset HRF

    Returns:
        Resultados completos do processamento
    """
    logger.info("🔬 INICIANDO PROCESSAMENTO HRF - VERSÃO ACADÊMICA")
    logger.info("=" * 60)

    try:
        # Carregar dataset com validação rigorosa
        loader = HRFImageLoader()
        images = loader.load_hrf_dataset(dataset_path)

        if not images:
            logger.error("❌ Nenhuma imagem válida carregada")
            return None

        # Processar todas as imagens
        all_results = {}
        total_start_time = time.time()

        for i, (img_name, original_img) in enumerate(images.items(), 1):
            logger.info(f"\n[{i}/{len(images)}] Processando: {img_name}")

            image_results = process_single_image(img_name, original_img)
            all_results[img_name] = image_results

        total_time = time.time() - total_start_time

        # Resumo final
        logger.info("\n" + "=" * 60)
        logger.info("✅ PROCESSAMENTO CONCLUÍDO")
        logger.info(f"📊 Imagens processadas: {len(images)}")
        logger.info(f"⏱️  Tempo total: {total_time:.2f}s")
        logger.info(f"⚡ Tempo médio/imagem: {total_time/len(images):.2f}s")
        logger.info("🎯 Métricas específicas para oftalmologia calculadas")
        logger.info("=" * 60)

        return all_results

    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO: {str(e)}")
        return None

if __name__ == "__main__":
    # Teste standalone
    dataset_path = 'data/hrf_dataset/images'
    results = main(dataset_path)

    if results:
        logger.info("✅ Preprocessing módulo funcionando corretamente")
    else:
        logger.error("❌ Falha no preprocessing")
