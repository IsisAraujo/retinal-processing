import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils import convert_numpy_types, safe_save_json

class BatchProcessor:
    """Processa lotes de imagens com gestão de recursos"""

    def __init__(self, processor, resource_manager):
        self.processor = processor
        self.resource_manager = resource_manager

    def process_batch(self, images: List[Path]) -> Dict[str, Any]:
        """Processa um lote de imagens com gerenciamento de recursos"""
        # Inicializar resultados
        results = self._initialize_results()
        results['total_images'] = len(images)

        if not images:
            raise ValueError("Nenhuma imagem encontrada para processamento")

        # Obter configurações de processamento otimizadas
        max_workers = self.resource_manager.get_optimal_workers()
        use_async = self.resource_manager.should_use_async()
        batch_size = self.resource_manager.get_batch_size()

        print(f"🔄 Processando {len(images)} imagens (workers: {max_workers}, batch: {batch_size})")

        # Processar em batches para maior eficiência
        if batch_size > 1:
            image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
        else:
            image_batches = [[img] for img in images]

        # Executar processamento
        with tqdm(total=len(images), desc="Processando imagens", unit="img") as pbar:
            if use_async:
                self._process_async(image_batches, results, pbar)
            else:
                self._process_sequential(images, results, pbar, max_workers)

        # Verificar recursos após processamento
        self.resource_manager.print_recommendations()

        # Análise agregada
        results['summary_statistics'] = self._calculate_batch_statistics(results)

        # Exibir resumo
        self._print_summary(results)

        return results

    def _initialize_results(self) -> Dict[str, Any]:
        """Inicializa estrutura de resultados"""
        return {
            'successful_processing': 0,
            'failed_processing': 0,
            'total_images': 0,
            'effective_enhancements': 0,
            'processing_times': [],
            'iqa_results': [],
            'training_dataset': []
        }

    def _process_async(self, image_batches: List[List[Path]], results: Dict[str, Any], pbar):
        """Executa processamento assíncrono de lotes"""
        max_workers = self.resource_manager.get_optimal_workers()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_batch_chunk, batch): batch
                      for batch in image_batches}

            for future in futures:
                try:
                    start_time = time.perf_counter()
                    batch_results = future.result()
                    processing_time = time.perf_counter() - start_time

                    # Atualizar estatísticas
                    self.resource_manager.update_task_stats(processing_time, success=True)

                    # Processar resultados do batch
                    batch_size = len(batch_results)
                    results['successful_processing'] += batch_size
                    results['processing_times'].extend([processing_time/batch_size] * batch_size)

                    self._update_batch_results(results, batch_results)

                except Exception as e:
                    results['failed_processing'] += len(futures[future])
                    pbar.write(f"❌ Erro no processamento em lote: {e}")
                finally:
                    pbar.update(len(futures[future]))

    def _process_sequential(self, images: List[Path], results: Dict[str, Any], pbar, max_workers: int):
        """Executa processamento sequencial"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.processor.process_image_with_iqa, img): img
                      for img in images}

            for future in futures:
                try:
                    start_time = time.perf_counter()
                    result = future.result()
                    processing_time = time.perf_counter() - start_time

                    # Atualizar estatísticas
                    self.resource_manager.update_task_stats(processing_time, success=True)

                    results['successful_processing'] += 1
                    results['processing_times'].append(processing_time)
                    results['iqa_results'].append(result['iqa_metrics'])

                    if result['enhancement_effective']:
                        results['effective_enhancements'] += 1

                    if result.get('training_data'):
                        results['training_dataset'].extend(result['training_data']['samples'])

                    self.processor._save_processed_image(result)

                    pbar.set_postfix({
                        'effective': f"{results['effective_enhancements']}/{results['successful_processing']}",
                        'avg_time': f"{np.mean(results['processing_times']):.1f}s"
                    })

                except Exception as e:
                    results['failed_processing'] += 1
                    pbar.write(f"❌ Erro: {e}")
                finally:
                    pbar.update(1)

    def _process_batch_chunk(self, image_batch: List[Path]) -> List[Dict]:
        """Processa um chunk de imagens como um lote"""
        results = []
        for img_path in image_batch:
            try:
                result = self.processor.process_image_with_iqa(img_path)
                results.append(result)
                self.processor._save_processed_image(result)
            except Exception as e:
                print(f"⚠️ Erro ao processar {img_path.name}: {e}")
        return results

    def _update_batch_results(self, results: Dict[str, Any], batch_results: List[Dict]):
        """Atualiza resultados com dados do batch processado"""
        for result in batch_results:
            results['iqa_results'].append(result['iqa_metrics'])

            if result['enhancement_effective']:
                results['effective_enhancements'] += 1

            if result.get('training_data'):
                results['training_dataset'].extend(result['training_data']['samples'])

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

        return convert_numpy_types({
            'effectiveness_rate': effectiveness_rate,
            'average_processing_time': avg_processing_time,
            'metrics_statistics': metrics_arrays
        })

    def _print_summary(self, results: Dict[str, Any]):
        """Exibe resumo do processamento"""
        effectiveness = results['effective_enhancements'] / max(results['successful_processing'], 1) * 100
        avg_time = np.mean(results['processing_times']) if results['processing_times'] else 0

        print(f"\n📊 RESUMO: {results['successful_processing']} imagens, "
              f"{effectiveness:.1f}% efetivas, {avg_time:.1f}s média")
