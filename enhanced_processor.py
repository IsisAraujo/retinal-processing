import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm
import torch

from enhanced_metrics import RetinalIQAMetrics, GroundTruthGenerator
from utils import numpy_to_python, safe_save_json

class EnhancedRetinalProcessor:
    """Processador retinal com IQA integrado"""

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iqa_metrics = RetinalIQAMetrics(device=self.device)
        self.gt_generator = GroundTruthGenerator()
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

    def process_image_with_iqa(self, path: Path) -> Dict[str, Any]:
        """Processa imagem com avaliação IQA completa"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Falha ao carregar: {path}")

        green = self._extract_green(img)
        best_result = self._find_optimal_clahe_params(green, path.stem)
        iqa_metrics = self.iqa_metrics.calculate_comprehensive_metrics(
            green, best_result['enhanced']
        )

        # Gerar dados de treinamento se necessário
        training_data = None
        if hasattr(self.config, 'generate_training_data') and self.config.generate_training_data:
            training_data = self._generate_training_data(green, path.stem)

        return {
            'path': path,
            'original': green,
            'enhanced': best_result['enhanced'],
            'optimal_params': best_result['params'],
            'iqa_metrics': iqa_metrics,
            'enhancement_effective': iqa_metrics['enhancement_effective'],
            'confidence_score': iqa_metrics['confidence_score'],
            'training_data': training_data
        }

    def _find_optimal_clahe_params(self, image: np.ndarray, image_id: str) -> Dict[str, Any]:
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

    def _generate_training_data(self, image: np.ndarray, image_id: str) -> Dict[str, Any]:
        """Gera dados de treinamento para ViT"""
        training_pairs = self.gt_generator.generate_training_pairs(
            image, n_positive=8, n_negative=8
        )

        labeled_data = []
        for enhanced_img, label, description in training_pairs:
            metrics = self.iqa_metrics.calculate_comprehensive_metrics(image, enhanced_img)
            labeled_data.append({
                'image_id': f"{image_id}_{description}",
                'enhanced_image': enhanced_img,
                'ground_truth_label': label,
                'description': description,
                'iqa_metrics': metrics,
                'enhancement_predicted': metrics['enhancement_effective'],
                'confidence': metrics['confidence_score']
            })

        positive_samples = sum(1 for item in labeled_data if item['ground_truth_label'] == 1)

        return {
            'samples': labeled_data,
            'statistics': {
                'total_samples': len(labeled_data),
                'positive_samples': positive_samples,
                'negative_samples': len(labeled_data) - positive_samples,
                'balance_ratio': positive_samples / max(len(labeled_data) - positive_samples, 1)
            }
        }

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
        safe_save_json(numpy_to_python(analysis_data), output_file)

    def process_batch_with_iqa(self, max_workers: int = 4, batch_size: int = 1, use_async: bool = False) -> Dict[str, Any]:
        """Processamento em lote com IQA usando WorkerManager para gestão de recursos

        Args:
            max_workers: Número máximo de workers paralelos
            batch_size: Tamanho do lote para processamento
            use_async: Usar processamento assíncrono

        Returns:
            Dicionário com resultados do processamento em lote
        """
        results = {
            'successful_processing': 0,
            'failed_processing': 0,
            'total_images': 0,
            'effective_enhancements': 0,
            'processing_times': [],
            'iqa_results': [],
            'training_dataset': []
        }

        # Obter worker_manager se disponível
        worker_manager = None
        try:
            from worker_manager import get_worker_manager
            worker_manager = get_worker_manager()

            # Otimizar parâmetros de processamento com base no WorkerManager
            if worker_manager:
                if max_workers == 4:  # Valor padrão não foi alterado
                    max_workers = worker_manager.get_cpu_workers()
                use_async = worker_manager.should_use_async()
                batch_size = worker_manager.get_batch_size()
        except ImportError:
            print("⚠️ WorkerManager não disponível - usando configuração padrão")

        images = self._find_images()
        results['total_images'] = len(images)

        if not images:
            raise ValueError("Nenhuma imagem encontrada para processamento")

        print(f"🔄 Processando {len(images)} imagens (workers: {max_workers}, batch: {batch_size})")

        # Processar em batches para maior eficiência
        if batch_size > 1:
            image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
        else:
            image_batches = [[img] for img in images]

        with tqdm(total=len(images), desc="Processando imagens", unit="img") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if use_async:
                    # Modo assíncrono para processamento em paralelo mais eficiente
                    futures = {executor.submit(self._process_batch_chunk, batch): batch
                              for batch in image_batches}

                    for future in futures:
                        try:
                            start_time = time.perf_counter()
                            batch_results = future.result()
                            processing_time = time.perf_counter() - start_time

                            # Atualizar estatísticas do WorkerManager se disponível
                            if worker_manager:
                                worker_manager.update_task_stats(processing_time, success=True)

                            # Processar resultados do batch
                            batch_size = len(batch_results)
                            results['successful_processing'] += batch_size
                            results['processing_times'].extend([processing_time/batch_size] * batch_size)

                            for result in batch_results:
                                results['iqa_results'].append(result['iqa_metrics'])

                                if result['enhancement_effective']:
                                    results['effective_enhancements'] += 1

                                if result.get('training_data'):
                                    results['training_dataset'].extend(result['training_data']['samples'])

                                self._save_processed_image(result)

                        except Exception as e:
                            results['failed_processing'] += len(futures[future])
                            pbar.write(f"❌ Erro no processamento em lote: {e}")
                        finally:
                            pbar.update(len(futures[future]))
                else:
                    # Modo tradicional - processar um por um
                    futures = {executor.submit(self._process_single_with_iqa, img): img
                              for img in images}

                    for future in futures:
                        try:
                            start_time = time.perf_counter()
                            result = future.result()
                            processing_time = time.perf_counter() - start_time

                            # Atualizar estatísticas do WorkerManager se disponível
                            if worker_manager:
                                worker_manager.update_task_stats(processing_time, success=True)

                            results['successful_processing'] += 1
                            results['processing_times'].append(processing_time)
                            results['iqa_results'].append(result['iqa_metrics'])

                            if result['enhancement_effective']:
                                results['effective_enhancements'] += 1

                            if result.get('training_data'):
                                results['training_dataset'].extend(result['training_data']['samples'])

                            self._save_processed_image(result)

                            pbar.set_postfix({
                                'effective': f"{results['effective_enhancements']}/{results['successful_processing']}",
                                'avg_time': f"{np.mean(results['processing_times']):.1f}s"
                            })

                        except Exception as e:
                            results['failed_processing'] += 1
                            pbar.write(f"❌ Erro: {e}")
                        finally:
                            pbar.update(1)

        # Verificar recursos disponíveis após processamento
        if worker_manager:
            # Verificar otimizações potenciais para futuros processamentos
            if results['processing_times'] and np.mean(results['processing_times']) > 5.0:
                resource_advice = worker_manager.optimize_for_memory_usage()
                if resource_advice['action'] != 'maintain_current':
                    print(f"⚠️ Recomendação: {resource_advice['action']} - Ajustar para {resource_advice['suggested_workers']} workers")

        # Análise agregada
        results['summary_statistics'] = self._calculate_batch_statistics(results)

        effectiveness = results['effective_enhancements'] / max(results['successful_processing'], 1) * 100
        avg_time = np.mean(results['processing_times']) if results['processing_times'] else 0

        print(f"\n📊 RESUMO: {results['successful_processing']} imagens, "
              f"{effectiveness:.1f}% efetivas, {avg_time:.1f}s média")

        return results

    def _process_batch_chunk(self, image_batch: List[Path]) -> List[Dict]:
        """Processa um chunk de imagens como um lote"""
        results = []
        for img_path in image_batch:
            try:
                result = self._process_single_with_iqa(img_path)
                results.append(result)
            except Exception as e:
                print(f"⚠️ Erro ao processar {img_path.name}: {e}")
        return results

    def _process_single_with_iqa(self, path: Path) -> Dict[str, Any]:
        """Processa uma única imagem com IQA"""
        return self.process_image_with_iqa(path)

    def _save_processed_image(self, result: Dict[str, Any]) -> None:
        """Salva imagem processada com metadados"""
        output_dir = self.config.paths['output']
        output_dir.mkdir(exist_ok=True)

        # Salvar imagem
        output_path = output_dir / result['path'].name
        cv2.imwrite(str(output_path), result['enhanced'])

        # Salvar metadados
        metadata_path = output_dir / f"{result['path'].stem}_iqa_metadata.json"
        metadata = {
            'original_file': str(result['path']),
            'processed_file': str(output_path),
            'optimal_parameters': result['optimal_params'],
            'iqa_assessment': {
                'enhancement_effective': result['enhancement_effective'],
                'confidence_score': result['confidence_score'],
                'clinical_relevance': result['iqa_metrics']['clinical_relevance_score']
            },
            'key_metrics': {
                k: v for k, v in result['iqa_metrics'].items()
                if k in ['vessel_clarity_gain', 'detail_preservation_score',
                        'perceptual_quality_score', 'total_artifact_score']
            }
        }

        safe_save_json(numpy_to_python(metadata), metadata_path)

    def _calculate_batch_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calcula estatísticas agregadas do lote"""
        if not results['iqa_results']:
            return {}

        metrics_arrays = {}
        for metric_name in ['clinical_relevance_score', 'vessel_clarity_gain',
                           'confidence_score', 'detail_preservation_score']:
            values = [m.get(metric_name, 0) for m in results['iqa_results']]
            if values:
                metrics_arrays[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        effectiveness_rate = results['effective_enhancements'] / max(results['successful_processing'], 1)
        avg_processing_time = np.mean(results['processing_times']) if results['processing_times'] else 0

        return numpy_to_python({
            'effectiveness_rate': effectiveness_rate,
            'average_processing_time': avg_processing_time,
            'metrics_statistics': metrics_arrays
        })

    def _extract_green(self, image: np.ndarray) -> np.ndarray:
        """Extrai canal verde com validação"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Imagem deve estar no formato BGR")
        return image[:, :, 1]

    def _find_images(self) -> List[Path]:
        """Busca imagens válidas"""
        exts = {ext.lower() for ext in self.config.clahe_params['extensions']}
        return [f for f in self.config.paths['input'].iterdir()
                if f.is_file() and f.suffix.lower() in exts]

class IQAReportGenerator:
    """Gerador de relatórios IQA"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def generate_iqa_report(self, batch_results: Dict[str, Any],
                           config) -> Tuple[Path, Path]:
        """Gera relatório científico focado em IQA"""
        report_path = self.output_dir / f"iqa_report_{self.timestamp}.txt"
        json_path = self.output_dir / f"iqa_data_{self.timestamp}.json"

        # Relatório em texto
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_report(batch_results, config))

        # Dados estruturados
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(numpy_to_python(self._prepare_json_data(batch_results)), f, indent=2)

        print(f"📄 Relatório: {report_path}")
        return report_path, json_path

    def _generate_text_report(self, results: Dict[str, Any], config) -> str:
        """Gera relatório em texto"""
        sections = [
            "=" * 80,
            "RELATÓRIO DE IMAGE QUALITY ASSESSMENT (IQA)",
            "=" * 80,
            f"Data/Hora: {time.strftime('%d/%m/%Y %H:%M:%S')}",
            f"Dataset: {config.paths['input']}",
            "",
            "RESUMO EXECUTIVO",
            "-" * 40,
            f"Imagens processadas: {results['successful_processing']}",
            f"Enhancements efetivos: {results['effective_enhancements']}",
            f"Taxa de efetividade: {results['effective_enhancements']/max(results['successful_processing'], 1)*100:.1f}%",
            f"Dados de treinamento: {len(results.get('training_dataset', []))} amostras",
            ""
        ]

        # Estatísticas detalhadas
        if 'summary_statistics' in results and 'metrics_statistics' in results['summary_statistics']:
            sections.append("MÉTRICAS AGREGADAS")
            sections.append("-" * 40)

            stats = results['summary_statistics']['metrics_statistics']
            for metric_name, metric_stats in stats.items():
                sections.extend([
                    f"{metric_name.replace('_', ' ').title()}:",
                    f"  Média: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}",
                    f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]",
                    ""
                ])

        sections.extend([
            "=" * 80,
            f"Relatório gerado em {time.strftime('%d/%m/%Y %H:%M:%S')}",
            "=" * 80
        ])

        return "\n".join(sections)

    def _prepare_json_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara dados para JSON"""
        serializable_results = {}

        for key, value in results.items():
            if key == 'training_dataset':
                serializable_training = []
                for sample in value:
                    serializable_sample = sample.copy()
                    # Remover imagens numpy (muito grandes)
                    if 'enhanced_image' in serializable_sample:
                        del serializable_sample['enhanced_image']
                    serializable_training.append(serializable_sample)
                serializable_results[key] = serializable_training
            else:
                serializable_results[key] = value

        return {
            'timestamp': self.timestamp,
            'report_type': 'Image Quality Assessment',
            'methodology': 'Enhanced CLAHE with automated IQA validation',
            'results': serializable_results
        }
