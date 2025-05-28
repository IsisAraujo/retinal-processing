import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import entropy

class RetinalProcessor:
    def __init__(self, config):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_params['clip_limit'],
            tileGridSize=self.config.clahe_params['tile_grid']
        )

    def extract_green(self, image: np.ndarray) -> np.ndarray:
        """Extrai canal verde com validação"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Imagem deve estar no formato BGR")
        return image[:, :, 1]

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Aplica CLAHE com normalização automática"""
        # Normaliza para 8-bit se necessário
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return self.clahe.apply(image)

    def process_image(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Pipeline otimizado para imagem retiniana"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Falha ao carregar: {path}")

        # Validação de qualidade mínima
        if img.shape[0] < 100 or img.shape[1] < 100:
            raise ValueError(f"Imagem muito pequena: {img.shape}")

        green = self.extract_green(img)
        processed = self.apply_clahe(green)

        # Verificação de melhoria
        if np.std(processed) < np.std(green) * 0.8:
            print(f"Aviso: CLAHE reduziu contraste em {path.name}")

        return green, processed

    def process_batch(self, max_workers: int = 4) -> Dict:
        """Processamento paralelo do dataset"""
        results = {'sucessos': 0, 'falhas': 0, 'tempos': [], 'metricas': []}
        output_dir = self.config.paths['output']
        output_dir.mkdir(exist_ok=True)

        images = self._find_images()
        if not images:
            raise ValueError("Nenhuma imagem encontrada")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_single, img): img
                      for img in images}

            for future in futures:
                try:
                    metrics = future.result()
                    if metrics:
                        results['sucessos'] += 1
                        results['tempos'].append(metrics['tempo'])
                        results['metricas'].append(metrics)
                except Exception as e:
                    results['falhas'] += 1
                    print(f"Erro: {e}")

        return results

    def _process_single(self, path: Path) -> Dict:
        """Processa uma única imagem com métricas"""
        start = time.perf_counter()
        green, processed = self.process_image(path)

        # Salvar resultado
        output_path = self.config.paths['output'] / path.name
        cv2.imwrite(str(output_path), processed)

        # Calcular métricas
        metrics = self._calculate_metrics(green, processed)
        metrics['arquivo'] = path.name
        metrics['tempo'] = time.perf_counter() - start

        return metrics

    def _calculate_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Métricas científicas para avaliação de qualidade"""
        # Contraste (Michelson)
        contrast_orig = (original.max() - original.min()) / (original.max() + original.min() + 1e-7)
        contrast_proc = (processed.max() - processed.min()) / (processed.max() + processed.min() + 1e-7)

        # PSNR
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(max(mse, 1e-10)))

        # Entropia (medida de informação)
        hist_orig, _ = np.histogram(original, bins=256, range=(0, 256))
        hist_proc, _ = np.histogram(processed, bins=256, range=(0, 256))

        entropy_orig = entropy(hist_orig + 1e-10)
        entropy_proc = entropy(hist_proc + 1e-10)

        # Nitidez (gradiente)
        grad_orig = np.std(cv2.Laplacian(original, cv2.CV_64F))
        grad_proc = np.std(cv2.Laplacian(processed, cv2.CV_64F))

        return {
            'ganho_contraste': contrast_proc / max(contrast_orig, 1e-7),
            'psnr': psnr,
            'ganho_entropia': entropy_proc - entropy_orig,
            'ganho_nitidez': grad_proc / max(grad_orig, 1e-7)
        }

    def _find_images(self) -> List[Path]:
        """Busca eficiente de imagens válidas"""
        exts = {ext.lower() for ext in self.config.clahe_params['extensions']}
        return [f for f in self.config.paths['input'].iterdir()
                if f.is_file() and f.suffix.lower() in exts]
