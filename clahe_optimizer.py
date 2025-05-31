import cv2
import numpy as np
import time
from typing import Dict, List, Any
from tqdm import tqdm
from pathlib import Path
from utils import convert_numpy_types, safe_save_json

class CLAHEOptimizer:
    """Classe responsável por encontrar parâmetros CLAHE ótimos usando IQA"""

    def __init__(self, config, iqa_metrics):
        self.config = config
        self.iqa_metrics = iqa_metrics
        self.clahe_variants = self._generate_clahe_variants()
        self.parameter_cache = {}

    def _generate_clahe_variants(self) -> List[Dict]:
        """Gera variantes de parâmetros CLAHE"""
        variants = []
        clip_limits = self.config.clahe_params['optimization_grid']['clip_limits']
        tile_sizes = self.config.clahe_params['optimization_grid']['tile_grids']

        for clip in clip_limits:
            for tile in tile_sizes:
                variants.append({
                    'clip_limit': clip,
                    'tile_grid': tile,
                    'name': f"clahe_c{clip}_t{tile[0]}x{tile[1]}"
                })

        return variants

    def find_optimal_params(self, image: np.ndarray, image_id: str) -> Dict[str, Any]:
        """Encontra parâmetros CLAHE ótimos usando IQA"""
        cache_key = f"{image_id}_{hash(image.tobytes()) % 10000}"
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key]

        best_score = -1
        best_result = None
        results = []

        with tqdm(total=len(self.clahe_variants),
                  desc=f"Optimizing CLAHE for {image_id}",
                  unit="variant") as pbar:

            for variant in self.clahe_variants:
                try:
                    # Aplicar CLAHE
                    clahe = cv2.createCLAHE(
                        clipLimit=variant['clip_limit'],
                        tileGridSize=variant['tile_grid']
                    )
                    enhanced = clahe.apply(image)

                    # Avaliar qualidade
                    metrics = self.iqa_metrics.calculate_comprehensive_metrics(image, enhanced)
                    composite_score = self._calculate_composite_score(metrics)

                    result = {
                        'params': variant,
                        'enhanced': enhanced,
                        'metrics': metrics,
                        'composite_score': composite_score
                    }
                    results.append(result)

                    if composite_score > best_score:
                        best_score = composite_score
                        best_result = result

                    pbar.set_postfix({
                        'best_score': f"{best_score:.3f}",
                        'current': f"{composite_score:.3f}"
                    })

                except Exception as e:
                    pbar.write(f"⚠️ Error testing {variant['name']}: {e}")
                finally:
                    pbar.update(1)

        if best_result:
            print(f"✅ Optimal: {best_result['params']['name']} (score: {best_result['composite_score']:.3f})")

        self.parameter_cache[cache_key] = best_result
        self._save_parameter_analysis(image_id, results)
        return best_result

    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calcula score composto para ranking"""
        weights = {
            'clinical_relevance_score': 0.30,
            'vessel_clarity_gain': 0.25,
            'detail_preservation_score': 0.20,
            'perceptual_quality_score': 0.15,
            'confidence_score': 0.10
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                normalized_value = min(1.0, max(0.0, metrics[metric]))
                score += weight * normalized_value
                total_weight += weight

        # Penalizar artefatos
        if 'total_artifact_score' in metrics:
            artifact_penalty = min(0.2, metrics['total_artifact_score'] * 0.1)
            score -= artifact_penalty

        return max(0.0, min(1.0, score / max(total_weight, 1.0)))

    def _save_parameter_analysis(self, image_id: str, results: List[Dict]) -> None:
        """Salva análise detalhada dos parâmetros"""
        analysis_dir = self.config.paths['parameter_analysis']
        analysis_dir.mkdir(exist_ok=True)

        analysis_data = {
            'image_id': image_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_variants_tested': len(results),
            'results': []
        }

        for result in results:
            serializable_result = {
                'params': result['params'],
                'composite_score': result['composite_score'],
                'key_metrics': {
                    k: result['metrics'].get(k, 0)
                    for k in ['clinical_relevance_score', 'vessel_clarity_gain',
                             'enhancement_effective', 'confidence_score']
                }
            }
            analysis_data['results'].append(serializable_result)

        # Ordenar por score
        analysis_data['results'].sort(key=lambda x: x['composite_score'], reverse=True)

        output_file = analysis_dir / f"{image_id}_parameter_analysis.json"
        safe_save_json(convert_numpy_types(analysis_data), output_file)
