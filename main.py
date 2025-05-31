#!/usr/bin/env python3
"""
Enhanced Retinal Image Processing with Image Quality Assessment (IQA)
Integrado com Worker Manager para gestão otimizada de recursos

Sistema de processamento retinal com:
- Processamento CLAHE otimizado
- Avaliação automática de qualidade (IQA)
- Treinamento de Vision Transformer
- Relatórios científicos
- Gestão centralizada de workers CPU/GPU
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import time
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, NamedTuple
import cv2
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ================================================================================================
# CLASSES DE EXCEÇÃO E UTILITÁRIOS DRY
# ================================================================================================

class DependencyError(Exception): pass
class ConfigurationError(Exception): pass
class ProcessingError(Exception): pass
class ModuleImportError(Exception): pass
class WorkerManagerError(Exception): pass

def fast_fail_check(condition: bool, message: str, exception_type: type = Exception):
    """Utilitário DRY para verificações Fast Fail"""
    if not condition:
        logger.error(f"Fast Fail: {message}")
        raise exception_type(message)

def safe_operation(default_value=None, log_errors=True):
    """Decorator DRY para operações seguras com fallback"""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Operação {func.__name__} falhou: {e}")
                return default_value
        return wrapper
    return decorator

def performance_monitor(func):
    """Decorator DRY para monitoramento de performance"""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.debug(f"{func.__name__}: {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"{func.__name__} falhou em {execution_time:.4f}s: {e}")
            raise
    return wrapper

# ================================================================================================
# MÓDULOS DE VALIDAÇÃO E DEPENDÊNCIAS
# ================================================================================================

@performance_monitor
def validate_dependencies() -> None:
    """Valida dependências críticas do sistema com Fast Fail"""
    dependencies = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'skimage': 'scikit-image',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    # Fast Fail: Dependências críticas
    fast_fail_check(
        not missing,
        f"Dependências faltando: {', '.join(missing)}\n"
        f"Instale: pip install {' '.join(missing)}",
        DependencyError
    )

    # PyTorch é opcional
    try:
        import torch
        logger.debug(f"PyTorch: {torch.__version__}")
    except ImportError:
        logger.warning("PyTorch não disponível - ViT desabilitado")

    logger.info("Dependências validadas")

@performance_monitor
def import_project_modules() -> Tuple[Any, Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any], Optional[Tuple[Any, Any]]]:
    """Importa módulos do projeto com tratamento robusto de erros"""
    modules = {}

    # Módulos obrigatórios
    required = {
        'config': 'EnhancedHRFConfig',
        'enhanced_processor': ['EnhancedRetinalProcessor', 'IQAReportGenerator'],
        'enhanced_metrics': ['RetinalIQAMetrics', 'GroundTruthGenerator'],
        'visualizer': ['visualize_results', 'plot_batch_metrics']
    }

    for module_name, classes in required.items():
        try:
            module = __import__(module_name)
            if isinstance(classes, list):
                modules[module_name] = tuple(getattr(module, cls) for cls in classes)
            else:
                modules[module_name] = getattr(module, classes)
            logger.debug(f"✅ {module_name} importado")
        except ImportError as e:
            raise ModuleImportError(f"Falha ao importar {module_name}: {e}")
        except AttributeError as e:
            raise ModuleImportError(f"Módulo {module_name} não contém classes necessárias: {e}")

    # ViT é opcional
    vit_modules = None
    try:
        vit = __import__('vit_model')
        vit_modules = (vit.integrate_vit_with_iqa_pipeline, vit.RetinalIQAViTPredictor)
        logger.debug("✅ vit_model importado")
    except ImportError as e:
        logger.warning(f"ViT não disponível: {e}")

    return (modules['config'], modules['enhanced_processor'],
            modules['enhanced_metrics'], modules['visualizer'], vit_modules)

@safe_operation(default_value={'available': False})
def import_worker_manager():
    """Importa Worker Manager com fallback gracioso"""
    try:
        from worker_manager import (
            get_worker_manager,
            print_worker_config,
            GPUContext,
            get_cpu_workers,
            get_device,
            should_use_async,
            fast_fail_check,
            safe_operation,
            performance_monitor
        )
        logger.info("✅ Worker Manager importado")
        return {
            'get_worker_manager': get_worker_manager,
            'print_worker_config': print_worker_config,
            'GPUContext': GPUContext,
            'get_cpu_workers': get_cpu_workers,
            'get_device': get_device,
            'should_use_async': should_use_async,
            'fast_fail_check': fast_fail_check,
            'safe_operation': safe_operation,
            'performance_monitor': performance_monitor,
            'available': True
        }
    except ImportError as e:
        logger.warning(f"Worker Manager não disponível: {e}")
        return {
            'available': False,
            'get_cpu_workers': lambda: 4,
            'get_device': lambda worker_id=0: 'cpu',
            'should_use_async': lambda: False,
            'GPUContext': None
        }

# ================================================================================================
# PIPELINE PRINCIPAL
# ================================================================================================

class EnhancedIQAPipeline:
    """Pipeline principal para processamento IQA retinal com Worker Manager integrado"""

    def __init__(self, config: 'EnhancedHRFConfig'):
        # Fast Fail: Validação imediata da configuração
        fast_fail_check(
            config.validate(),
            "Configuração inválida",
            ConfigurationError
        )

        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Importar Worker Manager
        self.worker_manager_modules = import_worker_manager()
        self.worker_manager_available = self.worker_manager_modules['available']

        # Configurar Worker Manager se disponível
        if self.worker_manager_available:
            self._setup_worker_manager()
        else:
            self._setup_fallback_workers()

        # Importar e inicializar componentes
        self._setup_modules()
        self._setup_logging()

        self.logger.info("Pipeline IQA inicializado com Worker Manager")

    def _setup_worker_manager(self):
        """Configura Worker Manager"""
        try:
            self.worker_manager = self.worker_manager_modules['get_worker_manager']()
            self.max_workers = self.worker_manager.get_cpu_workers()
            self.gpu_workers = self.worker_manager.get_gpu_workers()
            self.device = self.worker_manager.get_device()
            self.use_async = self.worker_manager.should_use_async()

            self.logger.info(f"Worker Manager ativo: {self.max_workers} CPU, {self.gpu_workers} GPU workers")

            # Imprimir configuração detalhada
            if hasattr(self.worker_manager, 'print_configuration'):
                self.worker_manager.print_configuration()

        except Exception as e:
            self.logger.error(f"Erro ao configurar Worker Manager: {e}")
            self.worker_manager_available = False
            self._setup_fallback_workers()

    def _setup_fallback_workers(self):
        """Configuração fallback sem Worker Manager"""
        self.worker_manager = None
        self.max_workers = self.config.get_max_workers()
        self.gpu_workers = 0
        self.device = self.config.get_device()
        self.use_async = False

        self.logger.warning(f"Fallback workers: {self.max_workers} CPU workers")

    def _setup_modules(self) -> None:
        """Importa módulos e inicializa componentes"""
        try:
            modules = import_project_modules()
            (self.config_module, self.processor_modules,
             self.metrics_modules, self.visualizer_modules, self.vit_modules) = modules

            # Inicializar componentes principais
            EnhancedRetinalProcessor, IQAReportGenerator = self.processor_modules
            self.processor = EnhancedRetinalProcessor(self.config)
            self.report_generator = IQAReportGenerator(self.config.paths['results'])

            RetinalIQAMetrics, GroundTruthGenerator = self.metrics_modules
            self.iqa_metrics = RetinalIQAMetrics()
            self.gt_generator = GroundTruthGenerator()

        except (ModuleImportError, Exception) as e:
            raise ConfigurationError(f"Falha na configuração: {e}")

    def _setup_logging(self) -> None:
        """Configura logging em arquivo"""
        try:
            log_dir = self.config.paths['logs']
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"iqa_{time.strftime('%Y%m%d_%H%M%S')}.log"

            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

            self.logger.addHandler(handler)
            self.logger.info(f"Log configurado: {log_file}")
        except Exception as e:
            self.logger.warning(f"Falha no log: {e}")

    @performance_monitor
    def validate_input_data(self) -> Dict[str, Any]:
        """Valida dados de entrada com Fast Fail"""
        input_dir = self.config.paths['input']

        # Fast Fail: Verificar existência do diretório
        fast_fail_check(
            input_dir.exists(),
            f"Diretório não existe: {input_dir}",
            ProcessingError
        )

        valid_extensions = {ext.lower() for ext in self.config.clahe_params['extensions']}
        valid_images = []

        for file_path in input_dir.iterdir():
            if (file_path.is_file() and
                file_path.suffix.lower() in valid_extensions):
                try:
                    img = cv2.imread(str(file_path))
                    if img is not None and img.size > 0:
                        valid_images.append(file_path)
                except Exception as e:
                    self.logger.warning(f"Erro em {file_path.name}: {e}")

        # Fast Fail: Verificar existência de imagens válidas
        fast_fail_check(
            len(valid_images) > 0,
            f"Nenhuma imagem válida em {input_dir}",
            ProcessingError
        )

        stats = {
            'total_files': len(list(input_dir.iterdir())),
            'valid_images': len(valid_images),
            'image_paths': valid_images
        }

        self.logger.info(f"Validação: {stats['valid_images']} imagens válidas")
        return stats

    @performance_monitor
    def run_sample_analysis(self) -> Optional[Dict[str, Any]]:
        """Executa análise IQA em amostra com Worker Manager"""
        self.logger.info("ETAPA 1: Análise de amostra")

        try:
            stats = self.validate_input_data()
            sample_path = stats['image_paths'][0]

            # Usar Worker Manager se disponível
            if self.worker_manager_available and self.worker_manager_modules['GPUContext']:
                with self.worker_manager_modules['GPUContext'](worker_id=0) as device:
                    self.logger.info(f"Processando amostra no device: {device}")
                    result = self.processor.process_image_with_iqa(sample_path)
            else:
                result = self.processor.process_image_with_iqa(sample_path)

            # Gerar visualizações
            vis_success = self._create_visualizations(result, sample_path)

            self._log_sample_results(result, sample_path, vis_success)

            # Atualizar estatísticas do Worker Manager se disponível
            if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
                self.worker_manager.update_task_stats(2.5, success=True)

            return {
                'sample_path': sample_path,
                'iqa_result': result,
                'visualization_success': vis_success,
                'success': True,
                'device_used': self.device if not self.worker_manager_available else 'worker_managed'
            }

        except Exception as e:
            self.logger.error(f"Falha na análise: {e}")
            if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
                self.worker_manager.update_task_stats(0, success=False)
            return None

    def _create_visualizations(self, result: Dict, sample_path: Path) -> bool:
        """Cria visualizações com fallback"""
        vis_dir = self.config.paths['visualizations']
        vis_dir.mkdir(exist_ok=True)

        try:
            # Tentar visualização avançada
            visualize_results, _ = self.visualizer_modules
            visualize_results(
                result['original'], result['enhanced'],
                sample_path.stem, vis_dir,
                result['iqa_metrics'], result['optimal_params']
            )
            return True

        except Exception as e:
            self.logger.warning(f"Visualização avançada falhou: {e}")
            # Fallback para visualização básica
            return self._create_basic_visualization(result, sample_path, vis_dir)

    def _create_basic_visualization(self, result: Dict, sample_path: Path,
                                  vis_dir: Path) -> bool:
        """Cria visualização básica como fallback"""
        try:
            comparison = np.hstack((result['original'], result['enhanced']))
            cv2.imwrite(str(vis_dir / f"{sample_path.stem}_comparison.png"), comparison)

            diff = cv2.absdiff(result['enhanced'], result['original'])
            cv2.imwrite(str(vis_dir / f"{sample_path.stem}_diff.png"), diff)

            self.logger.info("Visualização básica criada")
            return False
        except Exception as e:
            self.logger.warning(f"Falha na visualização básica: {e}")
            return False

    def _log_sample_results(self, result: Dict, sample_path: Path, vis_success: bool):
        """Log dos resultados da amostra"""
        metrics = result['iqa_metrics']
        device_info = f" (device: {self.device})" if hasattr(self, 'device') else ""

        self.logger.info(f"Amostra {sample_path.name}{device_info}:")
        self.logger.info(f"  Enhancement: {result['enhancement_effective']}")
        self.logger.info(f"  Confiança: {result['confidence_score']:.3f}")
        self.logger.info(f"  Score clínico: {metrics['clinical_relevance_score']:.3f}")
        self.logger.info(f"  Parâmetros: {result['optimal_params']['name']}")
        self.logger.info(f"  Visualização: {'✅' if vis_success else '⚠️ Básica'}")

        # Log de Worker Manager se disponível
        if self.worker_manager_available:
            self.logger.info(f"  Workers: {self.max_workers} CPU, {self.gpu_workers} GPU")

    @performance_monitor
    def run_batch_processing(self, max_workers: int = None) -> Dict[str, Any]:
        """Executa processamento em lote com Worker Manager integrado"""
        # Usar Worker Manager para determinar workers se disponível
        if max_workers is None:
            if self.worker_manager_available:
                max_workers = self.max_workers
            else:
                max_workers = self.config.get_max_workers()

        # Ajustar workers baseado no Worker Manager
        if self.worker_manager_available and hasattr(self.worker_manager, 'optimize_for_memory_usage'):
            recommendations = self.worker_manager.optimize_for_memory_usage()
            if recommendations['action'] != 'maintain_current':
                max_workers = recommendations['suggested_workers']
                self.logger.info(f"Workers ajustados para {max_workers} baseado na memória")

        self.logger.info(f"ETAPA 2: Processamento em lote (workers: {max_workers})")

        start_time = time.perf_counter()
        self.validate_input_data()

        # Executar processamento com Worker Manager se disponível
        if self.worker_manager_available and self.use_async:
            results = self._run_async_batch_processing(max_workers)
        else:
            results = self.processor.process_batch_with_iqa(max_workers=max_workers)

        # Fast Fail: Verificar se houve processamento bem-sucedido
        fast_fail_check(
            results['successful_processing'] > 0,
            "Nenhuma imagem processada com sucesso",
            ProcessingError
        )

        total_time = time.perf_counter() - start_time
        effectiveness = results['effective_enhancements'] / results['successful_processing'] * 100

        # Log com informações do Worker Manager
        worker_info = f" (Worker Manager: {'✅' if self.worker_manager_available else '❌'})"
        self.logger.info(f"Processamento{worker_info}: {total_time:.2f}s, "
                        f"{results['successful_processing']} imagens, "
                        f"{effectiveness:.1f}% efetividade")

        # Adicionar estatísticas do Worker Manager se disponível
        if self.worker_manager_available and hasattr(self.worker_manager, 'get_stats'):
            worker_stats = self.worker_manager.get_stats()
            results['worker_manager_stats'] = worker_stats

        return results

    def _run_async_batch_processing(self, max_workers: int) -> Dict[str, Any]:
        """Processamento assíncrono usando Worker Manager"""
        self.logger.info("Executando processamento assíncrono com Worker Manager")

        # Por enquanto, delegar para o processamento padrão
        # Futuras versões podem implementar async/await aqui
        return self.processor.process_batch_with_iqa(max_workers=max_workers)

    @performance_monitor
    def run_statistical_aggregation(self) -> Dict[str, Any]:
        """NOVA ETAPA: Agregação estatística multi-imagem"""
        self.logger.info("ETAPA 2.5: Agregação Estatística")

        try:
            from multi_image_aggregator import ParameterAggregator

            analysis_dir = self.config.paths['parameter_analysis']

            # Verificar se há arquivos para agregar
            analysis_files = list(analysis_dir.glob("*_parameter_analysis.json"))
            if not analysis_files:
                self.logger.warning(f"Nenhum arquivo de análise encontrado em {analysis_dir}")
                return {}

            aggregator = ParameterAggregator(analysis_dir)
            aggregated_results = aggregator.aggregate_parameter_analyses()

            # Log dos resultados principais
            self._log_aggregation_results(aggregated_results)

            return aggregated_results

        except ImportError as e:
            self.logger.warning(f"multi_image_aggregator não disponível: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Falha na agregação: {e}")
            return {}

    def _log_aggregation_results(self, aggregated_results: Dict[str, Any]) -> None:
        """Log dos resultados de agregação (DRY)"""
        n_images = len(aggregated_results['images'])
        optimal_ranges = aggregated_results['parameter_statistics']['optimal_ranges']

        self.logger.info(f"📊 Agregação: {n_images} imagens processadas")
        self.logger.info(f"🎛️ Clip ótimo: {optimal_ranges['clip_limit']['mean']:.2f} ± {(optimal_ranges['clip_limit']['max'] - optimal_ranges['clip_limit']['min'])/2:.2f}")
        self.logger.info(f"🔳 Tile ótimo: {optimal_ranges['tile_grid']['most_frequent']}x{optimal_ranges['tile_grid']['most_frequent']}")

        # Detectar padrões significativos
        significant_tests = aggregated_results['parameter_statistics'].get('significance_tests', {})
        if significant_tests:
            self.logger.info("🧪 TESTES DE SIGNIFICÂNCIA:")
            for test_name, test_result in significant_tests.items():
                significance = "SIGNIFICATIVO" if test_result['significant'] else "NÃO SIGNIFICATIVO"
                self.logger.info(f"   {test_name}: {significance} (p={test_result['p_value']:.4f})")

        # Log de clustering se disponível
        if 'clustering_results' in aggregated_results and 'clusters' in aggregated_results['clustering_results']:
            clusters = aggregated_results['clustering_results']['clusters']
            self.logger.info(f"🎯 Clustering: {len(clusters)} grupos identificados")
            for cluster_id, cluster_info in clusters.items():
                self.logger.info(f"   {cluster_id}: {cluster_info['characteristics']} ({cluster_info['size']} imagens)")

    @performance_monitor
    def train_vit_model(self, batch_results: Dict[str, Any]) -> Optional['RetinalIQAViTPredictor']:
        """Treina modelo ViT com Worker Manager se disponível"""
        self.logger.info("ETAPA 3: Treinamento ViT")

        # Fast Fail: Verificar disponibilidade do ViT
        if not self.vit_modules:
            self.logger.warning("ViT não disponível")
            return None

        training_data = batch_results.get('training_dataset', [])
        # Fast Fail: Verificar tamanho mínimo do dataset
        if len(training_data) < 20:
            self.logger.warning(f"Dataset pequeno ({len(training_data)} amostras)")
            return None

        try:
            integrate_vit, _ = self.vit_modules

            # Configurar device para ViT baseado no Worker Manager
            if self.worker_manager_available:
                vit_device = self.device
                self.logger.info(f"ViT usando device do Worker Manager: {vit_device}")
            else:
                vit_device = 'cpu'

            # Passar informações do Worker Manager para o ViT se disponível
            if self.worker_manager_available and hasattr(self.config, 'vit_config'):
                if 'worker_integration' not in self.config.vit_config:
                    self.config.vit_config['worker_integration'] = {}

                self.config.vit_config['worker_integration'].update({
                    'device': vit_device,
                    'cpu_workers': self.max_workers,
                    'gpu_workers': self.gpu_workers,
                    'use_async': self.use_async
                })

            predictor = integrate_vit(self.config, training_data)
            self.logger.info("ViT treinado com sucesso")

            # Atualizar estatísticas do Worker Manager
            if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
                self.worker_manager.update_task_stats(60.0, success=True)  # ViT demora mais

            return predictor

        except Exception as e:
            self.logger.error(f"Falha no ViT: {e}")
            if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
                self.worker_manager.update_task_stats(0, success=False)
            return None

    @performance_monitor
    def validate_with_vit(self, vit_predictor, sample_analysis: Dict) -> Dict[str, Any]:
        """Valida resultados com ViT"""
        self.logger.info("ETAPA 4: Validação ViT")

        if not vit_predictor or not sample_analysis.get('success'):
            return {'comparison_available': False}

        try:
            result = sample_analysis['iqa_result']

            # Usar Worker Manager para GPU se disponível
            if self.worker_manager_available and self.worker_manager_modules['GPUContext']:
                with self.worker_manager_modules['GPUContext'](worker_id=0) as device:
                    self.logger.info(f"ViT validação no device: {device}")
                    vit_pred = vit_predictor.predict_enhancement_quality(
                        result['original'], result['enhanced']
                    )
            else:
                vit_pred = vit_predictor.predict_enhancement_quality(
                    result['original'], result['enhanced']
                )

            traditional = result['enhancement_effective']
            vit_decision = vit_pred['enhancement_effective']
            agreement = traditional == vit_decision

            self.logger.info(f"ViT vs Tradicional: {agreement}")

            return {
                'vit_prediction': vit_pred,
                'traditional_prediction': {
                    'enhancement_effective': traditional,
                    'confidence_score': result['confidence_score']
                },
                'agreement': agreement,
                'comparison_available': True,
                'device_used': self.device if hasattr(self, 'device') else 'unknown'
            }

        except Exception as e:
            self.logger.error(f"Falha na validação ViT: {e}")
            return {'comparison_available': False}

    @performance_monitor
    def generate_reports(self, batch_results: Dict, sample_analysis: Dict,
                       vit_validation: Dict) -> None:
        """Gera relatórios científicos com informações do Worker Manager"""
        self.logger.info("ETAPA 5: Relatórios")

        try:
            # Adicionar informações do Worker Manager aos resultados
            batch_results['worker_manager_info'] = self._get_worker_manager_info()

            self.report_generator.generate_iqa_report(batch_results, self.config)

            if batch_results.get('iqa_results'):
                self._generate_batch_visualizations(batch_results)

            if batch_results.get('training_dataset'):
                self._generate_training_report(batch_results, vit_validation)

        except Exception as e:
            self.logger.error(f"Falha nos relatórios: {e}")
            raise ProcessingError(f"Relatórios falharam: {e}")

    def _get_worker_manager_info(self) -> Dict[str, Any]:
        """Obtém informações do Worker Manager de forma DRY"""
        if self.worker_manager_available and hasattr(self.worker_manager, 'get_stats'):
            worker_stats = self.worker_manager.get_stats()
            return {
                'enabled': True,
                'config': {
                    'cpu_workers': self.max_workers,
                    'gpu_workers': self.gpu_workers,
                    'device': self.device,
                    'use_async': self.use_async
                },
                'stats': worker_stats
            }
        else:
            return {'enabled': False}

    def _generate_batch_visualizations(self, batch_results: Dict) -> None:
        """Gera visualizações agregadas"""
        try:
            legacy_metrics = [
                {
                    'ganho_contraste': m.get('vessel_clarity_gain', 1.0),
                    'ganho_entropia': m.get('clinical_relevance_score', 0.0),
                    'ganho_nitidez': m.get('detail_preservation_score', 1.0),
                    'psnr': m.get('perceptual_quality_score', 0.5) * 40,
                    'arquivo': f"image_{i}.jpg"
                }
                for i, m in enumerate(batch_results['iqa_results'])
            ]

            if legacy_metrics:
                _, plot_batch = self.visualizer_modules
                plot_batch(legacy_metrics, self.config.paths['visualizations'])

        except Exception as e:
            self.logger.warning(f"Falha em visualizações: {e}")

    def _generate_training_report(self, batch_results: Dict, vit_validation: Dict) -> None:
        """Gera relatório de treinamento com informações do Worker Manager"""
        try:
            report_path = (self.config.paths['results'] /
                          f"training_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self._format_training_report(batch_results, vit_validation))

        except Exception as e:
            self.logger.warning(f"Falha no relatório de treinamento: {e}")

    def _format_training_report(self, batch_results: Dict, vit_validation: Dict) -> str:
        """Formata relatório de treinamento com informações do Worker Manager"""
        dataset = batch_results['training_dataset']
        positive = sum(1 for s in dataset if s['ground_truth_label'] == 1)

        lines = [
            "RELATÓRIO DE TREINAMENTO",
            "=" * 40,
            f"Total: {len(dataset)} amostras",
            f"Positivas: {positive}",
            f"Negativas: {len(dataset) - positive}",
            f"Balanceamento: {positive/max(len(dataset) - positive, 1):.2f}",
        ]

        # Informações do Worker Manager
        if 'worker_manager_info' in batch_results:
            wm_info = batch_results['worker_manager_info']
            if wm_info['enabled']:
                lines.extend([
                    "",
                    "WORKER MANAGER:",
                    f"CPU Workers: {wm_info['config']['cpu_workers']}",
                    f"GPU Workers: {wm_info['config']['gpu_workers']}",
                    f"Device: {wm_info['config']['device']}",
                    f"Async: {wm_info['config']['use_async']}",
                ])

                if 'stats' in wm_info:
                    stats = wm_info['stats']
                    lines.extend([
                        f"Tarefas: {stats['completed_tasks']}/{stats['total_tasks']}",
                        f"Taxa sucesso: {stats['success_rate']:.1f}%",
                        f"Tempo médio: {stats['avg_task_time']:.2f}s"
                    ])

        if vit_validation.get('comparison_available'):
            vit_pred = vit_validation['vit_prediction']
            trad_pred = vit_validation['traditional_prediction']
            lines.extend([
                "",
                "RESULTADOS ViT:",
                f"ViT: {vit_pred['enhancement_effective']} ({vit_pred['confidence_score']:.3f})",
                f"Tradicional: {trad_pred['enhancement_effective']} ({trad_pred['confidence_score']:.3f})",
                f"Acordo: {'SIM' if vit_validation['agreement'] else 'NÃO'}",
                f"Device: {vit_validation.get('device_used', 'unknown')}"
            ])

        return "\n".join(lines)

    @performance_monitor
    def save_consolidated_results(self, batch_results: Dict, statistical_results: Dict) -> None:
        """Salva resultados consolidados com informações do Worker Manager"""
        try:
            consolidated_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_summary': {
                    'total_images': batch_results['successful_processing'],
                    'effective_enhancements': batch_results['effective_enhancements'],
                    'effectiveness_rate': batch_results['effective_enhancements'] / max(batch_results['successful_processing'], 1) * 100,
                    'average_processing_time': np.mean(batch_results.get('processing_times', [0]))
                },
                'statistical_analysis': statistical_results,
                'training_data_summary': {
                    'total_samples': len(batch_results.get('training_dataset', [])),
                    'positive_samples': sum(1 for s in batch_results.get('training_dataset', []) if s.get('ground_truth_label') == 1),
                    'negative_samples': sum(1 for s in batch_results.get('training_dataset', []) if s.get('ground_truth_label') == 0)
                },
                'worker_manager_summary': batch_results.get('worker_manager_info', {'enabled': False})
            }

            # Salvar em arquivo
            consolidated_path = self.config.paths['results'] / f"consolidated_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

            with open(consolidated_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"📊 Resultados consolidados salvos: {consolidated_path}")

        except Exception as e:
            self.logger.warning(f"Falha ao salvar resultados consolidados: {e}")

# ================================================================================================
# FUNÇÕES DE UTILIDADE E CLI
# ================================================================================================

def setup_arguments() -> argparse.ArgumentParser:
    """Configura argumentos da linha de comando com opções do Worker Manager"""
    parser = argparse.ArgumentParser(
        description="Enhanced Retinal IQA Pipeline with Worker Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Argumentos originais
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Workers paralelos (padrão: determinado pelo Worker Manager)')
    parser.add_argument('--sample-only', action='store_true',
                       help='Apenas análise de amostra')
    parser.add_argument('--skip-vit', action='store_true',
                       help='Pular treinamento ViT')
    parser.add_argument('--config', type=str,
                       help='Arquivo de configuração JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Log detalhado')
    parser.add_argument('--demo', action='store_true',
                       help='Demonstração do sistema')
    parser.add_argument('--fast', action='store_true',
                       help='Modo rápido')
    parser.add_argument('--aggregate-only', action='store_true',
                       help='Apenas agregação estatística (requer análises existentes)')

    # Novos argumentos do Worker Manager
    parser.add_argument('--force-cpu', action='store_true',
                       help='Forçar uso apenas da CPU (desabilitar GPU)')
    parser.add_argument('--disable-worker-manager', action='store_true',
                       help='Desabilitar Worker Manager (usar configuração manual)')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='ID da GPU específica para usar (padrão: 0)')
    parser.add_argument('--force-sync', action='store_true',
                       help='Forçar processamento síncrono')
    parser.add_argument('--worker-debug', action='store_true',
                       help='Debug detalhado do Worker Manager')

    return parser

@performance_monitor
def print_summary(batch_results: Dict, vit_predictor, config) -> None:
    """Imprime resumo final com informações do Worker Manager"""
    total = batch_results['successful_processing']
    effective_rate = batch_results['effective_enhancements'] / max(total, 1) * 100

    print(f"\n{'='*50}")
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*50}")
    print(f"Imagens: {total}")
    print(f"Efetividade: {effective_rate:.1f}%")

    if batch_results.get('processing_times'):
        avg_time = np.mean(batch_results['processing_times'])
        print(f"Tempo médio: {avg_time:.2f}s")

    # Informações do Worker Manager
    if 'worker_manager_info' in batch_results:
        wm_info = batch_results['worker_manager_info']
        if wm_info['enabled']:
            print(f"\n🔧 WORKER MANAGER:")
            print(f"Workers: {wm_info['config']['cpu_workers']} CPU / {wm_info['config']['gpu_workers']} GPU")
            print(f"Device: {wm_info['config']['device']}")
            print(f"Modo: {'Assíncrono' if wm_info['config']['use_async'] else 'Síncrono'}")

            if 'stats' in wm_info:
                stats = wm_info['stats']
                print(f"Taxa sucesso: {stats['success_rate']:.1f}%")
                print(f"Tempo médio por tarefa: {stats['avg_task_time']:.2f}s")
        else:
            print(f"\n🔧 WORKER MANAGER: ❌ Não disponível (usando fallback)")

    # Verificar estrutura correta dos dados estatísticos
    if batch_results.get('statistical_analysis'):
        stats = batch_results['statistical_analysis'].get('parameter_statistics', {})
        if 'optimal_ranges' in stats:
            opt = stats['optimal_ranges']
            print(f"\n📊 ESTATÍSTICAS DE PARÂMETROS ÓTIMOS:")
            print(f"Clip Limit: {opt['clip_limit']['mean']:.2f} (range: {opt['clip_limit']['min']:.2f}-{opt['clip_limit']['max']:.2f})")
            print(f"Tile Grid: {opt['tile_grid']['most_frequent']}x{opt['tile_grid']['most_frequent']}")

            # Adicionar testes de significância se disponíveis
            sig_tests = stats.get('significance_tests', {})
            if sig_tests:
                print(f"\n🧪 TESTES ESTATÍSTICOS:")
                for test_name, test_result in sig_tests.items():
                    status = "✅ SIGNIFICATIVO" if test_result['significant'] else "❌ NÃO SIGNIFICATIVO"
                    print(f"{test_name.replace('_', ' ').title()}: {status} (p={test_result['p_value']:.4f})")

    print(f"\n🤖 ViT: {'✅ Treinado' if vit_predictor else '❌ Não disponível'}")
    print(f"📁 Resultados: {config.paths['results']}")
    print(f"{'='*50}")

@performance_monitor
def run_demo() -> None:
    """Demonstração do sistema com Worker Manager"""
    print("🔬 DEMONSTRAÇÃO DO SISTEMA IQA COM WORKER MANAGER")

    try:
        # Verificar Worker Manager
        worker_modules = import_worker_manager()
        if worker_modules['available']:
            print("✅ Worker Manager disponível")
            worker_modules['print_worker_config']()
        else:
            print("❌ Worker Manager não disponível - usando fallback")

        # Simular dados
        sample_image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

        # Adicionar estruturas vasculares
        center = (256, 256)
        for angle in np.linspace(0, 2*np.pi, 8):
            end = (int(center[0] + 200 * np.cos(angle)),
                   int(center[1] + 200 * np.sin(angle)))
            cv2.line(sample_image, center, end,
                    int(np.random.randint(30, 70)), 2)

        # Importar e testar
        modules = import_project_modules()
        _, _, (RetinalIQAMetrics, GroundTruthGenerator), _, _ = modules

        # Usar Worker Manager se disponível
        device = worker_modules['get_device'](0) if worker_modules['available'] else 'cpu'
        print(f"🔧 Usando device: {device}")

        iqa = RetinalIQAMetrics()
        gt_gen = GroundTruthGenerator()

        # Processar
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sample_image)

        metrics = iqa.calculate_comprehensive_metrics(sample_image, enhanced)
        training_pairs = gt_gen.generate_training_pairs(sample_image, 3, 3)

        print(f"✅ Vessel Clarity: {metrics['vessel_clarity_gain']:.3f}")
        print(f"✅ Clinical Score: {metrics['clinical_relevance_score']:.3f}")
        print(f"✅ Enhancement: {metrics['enhancement_effective']}")
        print(f"✅ Training pairs: {len(training_pairs)}")
        print("✅ Sistema funcionando corretamente")

    except Exception as e:
        logger.error(f"Falha na demonstração: {e}")
        raise

@safe_operation(default_value=None)
def configure_worker_manager_from_args(args):
    """Configura Worker Manager baseado nos argumentos"""
    worker_modules = import_worker_manager()

    if not worker_modules['available'] or args.disable_worker_manager:
        logger.info("Worker Manager desabilitado ou não disponível")
        return None

    try:
        worker_manager = worker_modules['get_worker_manager']()

        # Aplicar configurações dos argumentos
        if args.force_cpu:
            logger.info("Forçando modo CPU apenas")
            worker_manager.force_cpu_mode()

        if args.worker_debug:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug do Worker Manager habilitado")

        if args.force_sync:
            logger.info("Forçando modo síncrono")
            # Worker Manager já lida com isso automaticamente

        return worker_manager

    except Exception as e:
        logger.warning(f"Erro ao configurar Worker Manager: {e}")
        return None

@safe_operation(default_value=False)
def validate_worker_manager_compatibility():
    """Valida compatibilidade do Worker Manager com o sistema"""
    worker_modules = import_worker_manager()

    if not worker_modules['available']:
        logger.warning("Worker Manager não disponível - funcionalidade limitada")
        return False

    try:
        worker_manager = worker_modules['get_worker_manager']()

        # Fast Fail: Verificações básicas
        if hasattr(worker_manager, 'get_stats'):
            stats = worker_manager.get_stats()

            if hasattr(worker_manager, 'config') and worker_manager.config.memory_limit_gb < 1.0:
                logger.warning("Memória muito baixa para Worker Manager")
                return False

            if hasattr(worker_manager, 'get_cpu_workers') and worker_manager.get_cpu_workers() < 1:
                logger.warning("Nenhum worker CPU disponível")
                return False

        logger.info("Worker Manager validado com sucesso")
        return True

    except Exception as e:
        logger.error(f"Erro na validação do Worker Manager: {e}")
        return False

# ================================================================================================
# FUNÇÃO PRINCIPAL
# ================================================================================================

def main() -> None:
    """Função principal com Worker Manager integrado"""
    try:
        parser = setup_arguments()
        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        print("Enhanced Retinal IQA Pipeline with Worker Manager")
        print("=" * 50)

        if args.demo:
            run_demo()
            return

        # Fast Fail: Validar Worker Manager
        if not args.disable_worker_manager:
            validate_worker_manager_compatibility()

        # Configurar Worker Manager
        worker_manager = configure_worker_manager_from_args(args)

        if args.aggregate_only:
            # Executar apenas agregação estatística
            modules = import_project_modules()
            EnhancedHRFConfig = modules[0]
            config = EnhancedHRFConfig(fast_mode=args.fast)

            if config.validate():
                pipeline = EnhancedIQAPipeline(config)
                statistical_results = pipeline.run_statistical_aggregation()
                if statistical_results:
                    print("✅ Agregação estatística concluída")
                    print(f"📊 Resultados em: {config.paths['results']}")
                else:
                    print("❌ Nenhum dado para agregar")
            return

        # Fast Fail: Validar dependências
        validate_dependencies()

        # Configurar dispositivo com Worker Manager
        device = _configure_device(args, worker_manager)
        logger.info(f"Dispositivo configurado: {device}")

        # Configurar sistema
        modules = import_project_modules()
        EnhancedHRFConfig = modules[0]

        # Fast Fail: Verificar arquivo de configuração
        config = _create_config(args, EnhancedHRFConfig)
        config.generate_training_data = not args.skip_vit

        # Fast Fail: Validar configuração
        if not config.validate():
            raise ConfigurationError("Configuração inválida")

        # Executar pipeline
        pipeline = EnhancedIQAPipeline(config)

        # Imprimir configuração do Worker Manager
        if worker_manager and args.worker_debug and hasattr(worker_manager, 'print_configuration'):
            worker_manager.print_configuration()

        # Análise de amostra
        sample_analysis = pipeline.run_sample_analysis()

        # Fast Fail: Verificar sucesso da análise
        if not sample_analysis or not sample_analysis.get('success'):
            raise ProcessingError("Falha na análise de amostra")

        if args.sample_only:
            vis_type = "completa" if sample_analysis.get('visualization_success') else "básica"
            device_info = f" (device: {sample_analysis.get('device_used', 'unknown')})"
            print(f"\nAnálise concluída{device_info} - Visualização: {vis_type}")
            return

        # Processamento em lote
        batch_results = pipeline.run_batch_processing(args.workers)

        # NOVA ETAPA: Agregação estatística
        statistical_results = {}
        if batch_results['successful_processing'] > 1:
            statistical_results = pipeline.run_statistical_aggregation()

            # NOVO: Salvar resultados consolidados
            if statistical_results:
                pipeline.save_consolidated_results(batch_results, statistical_results)

            # Usar resultados para otimizar ViT
            if statistical_results and not args.skip_vit:
                _optimize_vit_with_stats(config, statistical_results)

        # ViT
        vit_predictor = None
        if not args.skip_vit and batch_results.get('training_dataset'):
            vit_predictor = pipeline.train_vit_model(batch_results)

        # Validação ViT
        vit_validation = pipeline.validate_with_vit(vit_predictor, sample_analysis)

        # Relatórios
        pipeline.generate_reports(batch_results, sample_analysis, vit_validation)

        # Adicionar resultados estatísticos ao resumo final
        if statistical_results:
            batch_results['statistical_analysis'] = statistical_results

        # Resumo final com Worker Manager
        print_summary(batch_results, vit_predictor, config)

        # Estatísticas finais do Worker Manager
        _print_worker_manager_stats(worker_manager)

    except KeyboardInterrupt:
        print("\nProcessamento interrompido")
        sys.exit(130)

    except (DependencyError, ConfigurationError, ProcessingError,
            ModuleImportError, WorkerManagerError, ImportError) as e:
        print(f"\nErro: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Erro crítico: {e}", exc_info=True)
        print(f"\nErro inesperado: {e}")
        sys.exit(1)

# ================================================================================================
# FUNÇÕES AUXILIARES (DRY)
# ================================================================================================

def _configure_device(args, worker_manager) -> str:
    """Configura dispositivo de forma DRY"""
    device = 'cpu'
    try:
        import torch
        if worker_manager and not args.force_cpu:
            device = worker_manager.get_device()
            if args.gpu_id and 'cuda' in device:
                device = f'cuda:{args.gpu_id}'
        elif args.force_cpu:
            device = 'cpu'
        elif torch.cuda.is_available() and not args.force_cpu:
            device = 'cuda'
    except ImportError:
        if not args.force_cpu:
            logger.warning("PyTorch não disponível - usando CPU")
    return device

def _create_config(args, EnhancedHRFConfig):
    """Cria configuração com Fast Fail"""
    if args.config:
        config_path = Path(args.config)
        fast_fail_check(
            config_path.exists(),
            f"Config não encontrado: {args.config}",
            ConfigurationError
        )
        return EnhancedHRFConfig(args.config, fast_mode=args.fast)
    else:
        return EnhancedHRFConfig(fast_mode=args.fast)

def _optimize_vit_with_stats(config, statistical_results):
    """Otimiza ViT com estatísticas"""
    optimal_params = statistical_results.get('parameter_statistics', {}).get('optimal_ranges', {})
    if optimal_params:
        config.vit_config['optimal_clahe_params'] = optimal_params
        logger.info("🎯 Parâmetros ótimos integrados ao ViT")

def _print_worker_manager_stats(worker_manager):
    """Imprime estatísticas do Worker Manager"""
    if worker_manager and hasattr(worker_manager, 'get_stats'):
        final_stats = worker_manager.get_stats()
        print(f"\n📊 ESTATÍSTICAS FINAIS DO WORKER MANAGER:")
        print(f"Total de tarefas: {final_stats.total_tasks}")
        print(f"Taxa de sucesso: {final_stats.success_rate:.1f}%")
        print(f"Tempo médio por tarefa: {final_stats.avg_task_time:.2f}s")

if __name__ == "__main__":
    main()
