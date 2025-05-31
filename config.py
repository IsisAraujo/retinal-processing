from pathlib import Path
import json
from typing import Dict, Any, Optional
import multiprocessing
import shutil
import logging
import os

# Importação condicional do Worker Manager
try:
    from worker_manager import get_worker_manager, WorkerMode, fast_fail_check
    WORKER_MANAGER_AVAILABLE = True
except ImportError:
    WORKER_MANAGER_AVAILABLE = False
    logging.warning("Worker Manager não disponível - usando fallback")

logger = logging.getLogger(__name__)

class ConfigConstants:
    """Valores constantes para configuração do sistema."""
    MIN_MEMORY_GB = 2.0
    MIN_DISK_SPACE_GB = 2.0
    MIN_CPU_CORES = 2
    MAX_CLIP_LIMIT = 10.0
    MIN_CLIP_LIMIT = 0.1
    DEFAULT_EXPECTED_RESOLUTION = (3504, 2336)

class EnhancedHRFConfig:
    """Configuração centralizada para processamento de imagens retinianas."""

    def __init__(self, config_file: Optional[str] = None, fast_mode: bool = False):
        self._check_system_requirements()
        self._initialize_base_configs(fast_mode)
        self.paths = self._setup_paths()

        if config_file:
            self._load_config_file(config_file)

        self._setup_derived_configs()
        logger.info("Configuração HRF inicializada")

    def _check_system_requirements(self):
        """Verificação crítica de requisitos do sistema."""
        # Espaço em disco
        free_gb = shutil.disk_usage(os.getcwd()).free / (1024**3)
        if free_gb < ConfigConstants.MIN_DISK_SPACE_GB:
            raise RuntimeError(f"Espaço em disco insuficiente: {free_gb:.1f}GB")

        # Núcleos de CPU
        cpu_count = multiprocessing.cpu_count()
        if cpu_count < ConfigConstants.MIN_CPU_CORES:
            raise RuntimeError(f"Núcleos de CPU insuficientes: {cpu_count}")

    def _initialize_base_configs(self, fast_mode: bool):
        """Configurações fundamentais do pipeline."""
        self.hrf_specs = {
            'expected_resolution': ConfigConstants.DEFAULT_EXPECTED_RESOLUTION,
            'quality_threshold': 0.8
        }

        self.clahe_params = {
            'default_clip_limit': 2.0,
            'default_tile_grid': (8, 8),
            'extensions': None,  # Aceita qualquer extensão
            'optimization_grid': self._get_clahe_optimization_grid(fast_mode)
        }

        self.iqa_config = {
            'quality_thresholds': {
                'vessel_clarity_min': 1.1,
                'clinical_relevance_min': 0.6
            }
        }

        self.processing_config = self._get_processing_config()

    def _get_clahe_optimization_grid(self, fast_mode: bool) -> Dict[str, list]:
        """Gera grade de parâmetros para otimização CLAHE."""
        return {
            'clip_limits': [1.5, 2.0, 3.0] if fast_mode else [1.0, 2.0, 3.0, 4.0, 5.0],
            'tile_grids': [(8, 8), (12, 12)] if fast_mode else [(8, 8), (16, 16)]
        }

    def _get_processing_config(self) -> Dict[str, Any]:
        """Configuração de processamento integrada com Worker Manager."""
        config = {
            'max_workers': 4,
            'device': 'cpu'
        }

        if WORKER_MANAGER_AVAILABLE:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()
            config.update({
                'max_workers': worker_config.cpu_workers,
                'device': worker_config.device
            })

        return config

    def _setup_paths(self) -> Dict[str, Path]:
        """Configura estrutura de diretórios essenciais."""
        return {
            'input': Path('data/images'),
            'output': Path('processed'),
            'results': Path('results')
        }

    def _load_config_file(self, config_file: str):
        """Carrega configuração personalizada de arquivo JSON."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        for key, value in config_data.items():
            if key == 'paths':
                self.paths.update({k: Path(v) for k, v in value.items()})
            else:
                setattr(self, key, value)

    def _setup_derived_configs(self):
        """Calcula parâmetros derivados da configuração."""
        self.total_clahe_variants = (
            len(self.clahe_params['optimization_grid']['clip_limits']) *
            len(self.clahe_params['optimization_grid']['tile_grids'])
        )

    def validate(self) -> bool:
        """Validação crítica da configuração."""
        if not self.paths['input'].exists():
            logger.error(f"Diretório de entrada não existe: {self.paths['input']}")
            return False

        if len(self._get_valid_images()) == 0:
            logger.error("Nenhuma imagem válida encontrada")
            return False

        return True

    def _get_valid_images(self) -> list:
        """Lista imagens válidas no diretório de entrada."""
        if not self.paths['input'].exists():
            return []
        exts = self.clahe_params.get('extensions')
        return [
            f for f in self.paths['input'].iterdir()
            if f.is_file() and (exts is None or f.suffix.lower() in exts)
        ]

# Funções utilitárias para criação de configurações
def create_default_config(output_file: str = 'iqa_config.json'):
    """Cria configuração padrão e salva em arquivo."""
    config = EnhancedHRFConfig()
    with open(output_file, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    return config

def load_config(config_file: str) -> EnhancedHRFConfig:
    """Carrega configuração de arquivo existente."""
    return EnhancedHRFConfig(config_file)
