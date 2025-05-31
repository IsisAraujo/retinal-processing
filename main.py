#!/usr/bin/env python3
"""
Enhanced Retinal IQA Pipeline com Worker Manager
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Exceções customizadas
class DependencyError(Exception): pass
class ConfigurationError(Exception): pass
class ProcessingError(Exception): pass

def validate_dependencies() -> None:
    """Valida dependências críticas com Fast Fail"""
    dependencies = {'cv2', 'numpy', 'scipy', 'skimage'}
    missing = [d for d in dependencies if not __import__(d)]
    if missing:
        raise DependencyError(f"Dependências faltando: {', '.join(missing)}")

def import_worker_manager() -> Dict[str, Any]:
    """Importa Worker Manager com fallback"""
    try:
        from worker_manager import get_worker_manager, print_worker_config
        return {'available': True, 'get_worker_manager': get_worker_manager}
    except ImportError:
        return {'available': False}

class EnhancedIQAPipeline:
    """Pipeline principal para processamento IQA retinal"""

    def __init__(self, config: Any):
        self.config = config
        self.worker_manager = self._setup_worker_manager()
        self._initialize_processor()
        logger.info("Pipeline inicializado")

    def _setup_worker_manager(self) -> Any:
        """Configura Worker Manager"""
        wm = import_worker_manager()
        if wm['available']:
            manager = wm['get_worker_manager']()
            logger.info(f"Workers: {manager.get_cpu_workers()} CPU")
            return manager
        return None

    def _initialize_processor(self) -> None:
        """Inicializa o processador retinal"""
        try:
            from enhanced_processor import EnhancedRetinalProcessor, IQAReportGenerator
            self.processor = EnhancedRetinalProcessor(self.config)
            self.report_generator = IQAReportGenerator(self.config.paths['output'])
            logger.info("Processador retinal inicializado")
        except ImportError as e:
            logger.error(f"Erro ao inicializar processador: {e}")
            raise ConfigurationError("Falha ao carregar componentes de processamento")

    def run_sample_analysis(self) -> Dict[str, Any]:
        """Executa análise de amostra em uma imagem"""
        logger.info("Iniciando análise de amostra...")

        # Encontrar primeira imagem para análise
        images = list(self.config.paths['input'].glob("*.png"))
        if not images:
            images = list(self.config.paths['input'].glob("*.tif"))

        if not images:
            raise ProcessingError("Nenhuma imagem encontrada para análise")

        sample_image = images[0]
        logger.info(f"Analisando imagem de amostra: {sample_image.name}")

        try:
            start_time = time.time()
            result = self.processor.process_image_with_iqa(sample_image)
            elapsed = time.time() - start_time

            logger.info(f"Análise concluída em {elapsed:.2f}s")
            logger.info(f"Melhoria efetiva: {result['enhancement_effective']}")
            logger.info(f"Score clínico: {result['iqa_metrics']['clinical_relevance_score']:.3f}")

            return {
                'success': True,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'processing_time': elapsed,
                'image_name': sample_image.name,
                'enhancement_effective': result['enhancement_effective'],
                'optimal_params': result['optimal_params']
            }
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            raise ProcessingError(f"Falha na análise da amostra: {e}")

    def run_batch_processing(self) -> Dict[str, Any]:
        """Executa processamento em lote"""
        logger.info("Iniciando processamento em lote...")

        try:
            # Configurar worker manager para BatchProcessor
            max_workers = self.worker_manager.get_cpu_workers() if self.worker_manager else 4
            batch_size = self.worker_manager.get_batch_size() if self.worker_manager else 1
            use_async = self.worker_manager.should_use_async() if self.worker_manager else False

            # Executar processamento em lote
            logger.info(f"Configuração: {max_workers} workers, batch size {batch_size}, async={use_async}")
            start_time = time.time()
            results = self.processor.process_batch_with_iqa(
                max_workers=max_workers,
                batch_size=batch_size,
                use_async=use_async
            )
            elapsed = time.time() - start_time

            # Gerar relatório
            report_path, _ = self.report_generator.generate_iqa_report(results, self.config)

            effectiveness_rate = results['effective_enhancements'] / max(results['successful_processing'], 1) * 100
            logger.info(f"Processamento completo em {elapsed:.2f}s")
            logger.info(f"Processadas: {results['successful_processing']} imagens")
            logger.info(f"Taxa de efetividade: {effectiveness_rate:.1f}%")
            logger.info(f"Relatório: {report_path}")

            return {
                'processed': results['successful_processing'],
                'effective': results['effective_enhancements'],
                'effectiveness_rate': effectiveness_rate,
                'processing_time': elapsed,
                'report_path': str(report_path)
            }
        except Exception as e:
            logger.error(f"Erro no processamento em lote: {e}")
            raise ProcessingError(f"Falha no processamento em lote: {e}")

def setup_arguments() -> argparse.ArgumentParser:
    """Configura argumentos CLI"""
    parser = argparse.ArgumentParser(
        description="Enhanced Retinal IQA Pipeline"
    )
    parser.add_argument('--config', help='Arquivo de configuração JSON')
    parser.add_argument('--sample-only', action='store_true', help='Apenas análise de amostra')
    parser.add_argument('--verbose', action='store_true', help='Log detalhado')
    return parser

def load_configuration(config_path: Optional[str] = None) -> Any:
    """Carrega configuração do sistema"""
    try:
        # Se tiver um arquivo de configuração, use-o
        if config_path:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Converter para objeto de configuração
            from types import SimpleNamespace
            config = SimpleNamespace()

            # Definir caminhos
            config.paths = {
                'input': Path(config_data.get('input_dir', './input')),
                'output': Path(config_data.get('output_dir', './output')),
                'parameter_analysis': Path(config_data.get('analysis_dir', './analysis'))
            }

            # Definir parâmetros CLAHE
            config.clahe_params = config_data.get('clahe_params', {
                'extensions': ['.png', '.tif', '.jpg'],
                'optimization_grid': {
                    'clip_limits': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                    'tile_grids': [(4, 4), (8, 8), (12, 12), (16, 16), (24, 24)]
                }
            })

            # Outras configurações
            config.generate_training_data = config_data.get('generate_training_data', True)

        else:
            # Configuração padrão
            from types import SimpleNamespace
            config = SimpleNamespace()

            # Definir caminhos
            config.paths = {
                'input': Path('./data/images'),
                'output': Path('./output'),
                'parameter_analysis': Path('./analysis')
            }

            # Criar diretórios se não existirem
            for path in config.paths.values():
                path.mkdir(parents=True, exist_ok=True)

            # Definir parâmetros CLAHE
            config.clahe_params = {
                'extensions': ['.png', '.tif', '.jpg'],
                'optimization_grid': {
                    'clip_limits': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                    'tile_grids': [(4, 4), (8, 8), (12, 12), (16, 16), (24, 24)]
                }
            }

            # Outras configurações
            config.generate_training_data = False

        return config

    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        raise ConfigurationError(f"Falha na configuração: {e}")

def main() -> None:
    """Função principal"""
    parser = setup_arguments()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        validate_dependencies()

        # Carregar configuração
        config = load_configuration(args.config)

        # Inicializar pipeline
        pipeline = EnhancedIQAPipeline(config)

        if args.sample_only:
            # Executar apenas análise de amostra
            sample_result = pipeline.run_sample_analysis()
            logger.info(f"Análise concluída. Device: {sample_result['device']}")
            logger.info(f"Parâmetros ótimos: {sample_result['optimal_params']['name']}")
            return

        # Executar processamento em lote completo
        batch_result = pipeline.run_batch_processing()
        logger.info(f"Processamento finalizado: {batch_result['processed']} imagens, "
                   f"{batch_result['effectiveness_rate']:.1f}% efetivas")

    except (DependencyError, ConfigurationError, ProcessingError) as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
