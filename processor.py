import cv2
import numpy as np
import time  # Adicionar importação de time
from pathlib import Path
from typing import Tuple, List, Dict, Any  # Adicionar Any para retorno dos métodos
try:
    import cupy as cp
except ImportError:
    cp = None

class RetinalProcessor:
    def __init__(self, config):
        self.config = config
        self.use_cuda = cp is not None
        self._init_clahe()

    def _init_clahe(self):
        """Inicializa CLAHE com fallback para CPU"""
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_params['clip_limit'],
            tileGridSize=self.config.clahe_params['tile_grid']
        )

    def extract_green(self, image: np.ndarray) -> np.ndarray:
        """Extrai e retorna o canal verde com verificação de formato"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Imagem deve estar no formato BGR")
        return image[:, :, 1].copy()

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Aplica CLAHE otimizado com possível aceleração por GPU"""
        if self.use_cuda:
            return self._gpu_clahe(image)
        return self.clahe.apply(image)

    def _gpu_clahe(self, image: np.ndarray) -> np.ndarray:
        """Implementação CUDA do CLAHE usando CuPy"""
        try:
            # OpenCV não tem CLAHE para CUDA diretamente
            # Vamos converter para CPU, aplicar CLAHE e voltar para GPU
            gpu_img = cp.asarray(image)
            cpu_img = cp.asnumpy(gpu_img)
            processed = self.clahe.apply(cpu_img)
            return processed  # Retornamos o resultado da CPU
        except Exception as e:
            print(f"Falha CUDA: {e} - Usando CPU")
            return self.clahe.apply(image)

    def process_image(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Pipeline completo para uma imagem"""
        if not path.exists():
            raise FileNotFoundError(path)

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Imagem inválida: {path}")

        green = self.extract_green(img)
        processed = self.apply_clahe(green)
        return green, processed

    def process_batch(self) -> Dict:
        """Processa todo o dataset"""
        results = {'sucessos': 0, 'falhas': 0, 'tempos': [], 'images': []}  # Adicionar 'images' para armazenar métricas
        output_dir = self.config.paths['output']
        output_dir.mkdir(exist_ok=True)

        for path in self._find_images():
            try:
                start = time.perf_counter()
                green, processed = self.process_image(path)

                # Salvar a imagem processada
                output_path = output_dir / path.name
                cv2.imwrite(str(output_path), processed)

                # Calcular métricas da imagem
                metrics = self._calculate_image_metrics(green, processed)
                metrics['filename'] = path.name
                results['images'].append(metrics)

                results['sucessos'] += 1
                results['tempos'].append(time.perf_counter() - start)
            except Exception as e:
                print(f"Erro em {path.name}: {e}")
                results['falhas'] += 1

        return results

    def _calculate_image_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de qualidade para uma imagem processada"""
        # Cálculo do contraste (desvio padrão)
        contrast_orig = np.std(original.astype(float))
        contrast_proc = np.std(processed.astype(float))
        contrast_ratio = contrast_proc / max(contrast_orig, 1.0)  # Evitar divisão por zero

        # Cálculo do PSNR
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            psnr = 100.0  # Valor alto para MSE zero
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # Mudança de intensidade média
        intensity_change = np.mean(processed.astype(float) - original.astype(float))

        return {
            'contrast_ratio': contrast_ratio,
            'psnr': psnr,
            'intensity_change': intensity_change,
            'entropy_gain': 0.0  # Valor padrão
        }

    def _find_images(self) -> List[Path]:
        """Localiza imagens válidas no diretório de entrada"""
        exts = self.config.clahe_params['extensions']
        return [f for ext in exts for f in self.config.paths['input'].glob(f'*{ext}')]
