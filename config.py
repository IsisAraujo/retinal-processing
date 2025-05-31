from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
import multiprocessing
import shutil
import logging
import os

# Import worker manager for centralized worker logic
try:
    from worker_manager import (
        get_worker_manager, WorkerManager, WorkerMode,
        fast_fail_check, safe_operation, performance_monitor
    )
    WORKER_MANAGER_AVAILABLE = True
except ImportError:
    WORKER_MANAGER_AVAILABLE = False
    logging.warning("worker_manager não disponível - usando fallback")

    # Criar funções de fallback para manter compatibilidade
    def fast_fail_check(condition, message, exception_type=ValueError):
        if not condition:
            raise exception_type(message)

    def safe_operation(default_value=None, log_errors=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logging.warning(f"Operação {func.__name__} falhou: {e}")
                    return default_value
            return wrapper
        return decorator

    def performance_monitor(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

logger = logging.getLogger(__name__)

# ================================================================================================
# CONSTANTES E CONFIGURAÇÕES
# ================================================================================================

class ConfigConstants:
    """Constantes para configuração do sistema HRF"""

    # Parâmetros de hardware
    MIN_MEMORY_GB = 2.0
    MIN_DISK_SPACE_GB = 2.0
    MIN_CPU_CORES = 2

    # Limites de validação
    MAX_CLIP_LIMIT = 10.0
    MIN_CLIP_LIMIT = 0.1
    MIN_VESSEL_CLARITY = 0.5
    MAX_VESSEL_CLARITY = 3.0
    MAX_SPLIT_SUM = 0.9

    # Batch sizes
    BATCH_SIZE_HIGH_MEM = 64
    BATCH_SIZE_MED_MEM = 32
    BATCH_SIZE_LOW_MEM = 16

    # Thresholds de memória
    HIGH_MEMORY_THRESHOLD_GB = 16
    MED_MEMORY_THRESHOLD_GB = 8

    # Tamanhos de imagem
    DEFAULT_EXPECTED_RESOLUTION = (3504, 2336)
    MIN_RESOLUTION = (1000, 1000)
    MAX_RESOLUTION = (4000, 3000)

# ================================================================================================
# CLASSE DE CONFIGURAÇÃO PRINCIPAL
# ================================================================================================

class EnhancedHRFConfig:
    """Configuração para pipeline IQA de imagens retinianas com Worker Manager integrado"""

    def __init__(self, config_file: Optional[str] = None, fast_mode: bool = False):
        """
        Inicializa a configuração com Fast Fail para parâmetros críticos

        Args:
            config_file: Caminho para arquivo de configuração opcional
            fast_mode: Modo rápido com conjunto de parâmetros reduzido
        """
        # Fast Fail: Verificar disponibilidade do disco antes de prosseguir
        self._check_system_requirements()

        # Inicializar configurações base
        self._initialize_base_configs(fast_mode)

        # Estrutura de diretórios
        self.paths = self._setup_paths()

        # Carregar de arquivo se fornecido (após inicialização base)
        if config_file:
            self._load_config_file_safely(config_file)

        # Configurar parâmetros derivados (deve ser o último)
        self._setup_derived_configs()

        logger.info("Configuração HRF inicializada com sucesso")

    def _check_system_requirements(self):
        """Fast Fail: Verificação de requisitos do sistema"""
        # Verificar espaço em disco
        try:
            free_gb = shutil.disk_usage(os.getcwd()).free / (1024**3)
            fast_fail_check(
                free_gb >= ConfigConstants.MIN_DISK_SPACE_GB,
                f"Espaço em disco insuficiente: {free_gb:.1f}GB (mín: {ConfigConstants.MIN_DISK_SPACE_GB}GB)"
            )
        except Exception as e:
            logger.warning(f"Não foi possível verificar espaço em disco: {e}")

        # Verificar CPU cores
        cpu_count = multiprocessing.cpu_count()
        fast_fail_check(
            cpu_count >= ConfigConstants.MIN_CPU_CORES,
            f"Número de cores insuficiente: {cpu_count} (mín: {ConfigConstants.MIN_CPU_CORES})"
        )

    def _initialize_base_configs(self, fast_mode: bool):
        """Inicializa todas as configurações base com princípio DRY"""
        # Configurações base
        self.hrf_specs = self._setup_hrf_specs()

        # Parâmetros CLAHE
        self.clahe_params = self._setup_clahe_params(fast_mode)

        # Configurações IQA
        self.iqa_config = self._setup_iqa_config()

        # Dados de treinamento
        self.training_data_config = self._setup_training_config()

        # ViT (preparação futura)
        self.vit_config = self._setup_vit_config()

        # Processamento com Worker Manager integrado
        self.processing_config = self._setup_processing_config_with_workers()

        # Relatórios
        self.reporting_config = self._setup_reporting_config()

        # Flag para geração de dados
        self.generate_training_data = False

    def _setup_hrf_specs(self) -> Dict[str, Any]:
        """Configura especificações básicas de imagens HRF"""
        return {
            'expected_resolution': ConfigConstants.DEFAULT_EXPECTED_RESOLUTION,
            'min_resolution': ConfigConstants.MIN_RESOLUTION,
            'max_resolution': ConfigConstants.MAX_RESOLUTION,
            'quality_threshold': 0.8,
            'bit_depth': 8,
            'color_channels': 3
        }

    def _setup_clahe_params(self, fast_mode: bool) -> Dict[str, Any]:
        """Configura parâmetros CLAHE com base no modo de execução"""
        # Extensões suportadas
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.JPEG']

        # Parâmetros base
        base_params = {
            'default_clip_limit': 2.0,
            'default_tile_grid': (8, 8),
            'extensions': extensions
        }

        # Grade de otimização dependente do modo
        if fast_mode:
            optimization_grid = {
                'clip_limits': [1.5, 2.0, 3.0],
                'tile_grids': [(8, 8), (12, 12)]
            }
        else:
            optimization_grid = {
                'clip_limits': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                'tile_grids': [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (16, 16)]
            }

        # Validar parâmetros críticos (Fast Fail)
        for clip in optimization_grid['clip_limits']:
            fast_fail_check(
                ConfigConstants.MIN_CLIP_LIMIT <= clip <= ConfigConstants.MAX_CLIP_LIMIT,
                f"Clip limit inválido: {clip} (deve estar entre {ConfigConstants.MIN_CLIP_LIMIT} e {ConfigConstants.MAX_CLIP_LIMIT})"
            )

        base_params['optimization_grid'] = optimization_grid

        # Adicionar params para compatibilidade
        base_params.update({
            'clip_limit': base_params['default_clip_limit'],
            'tile_grid': base_params['default_tile_grid']
        })

        return base_params

    def _setup_iqa_config(self) -> Dict[str, Any]:
        """Configura parâmetros IQA com validação Fast Fail"""
        vessel_clarity_min = 1.1  # Valor crítico para análise de vasos

        # Fast Fail: Validar threshold de clareza de vasos
        fast_fail_check(
            ConfigConstants.MIN_VESSEL_CLARITY <= vessel_clarity_min <= ConfigConstants.MAX_VESSEL_CLARITY,
            f"Threshold vessel clarity inválido: {vessel_clarity_min}"
        )

        return {
            'vessel_analysis': {
                'frangi_sigmas': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                'vessel_threshold': 0.1,
                'connectivity_min_size': 100
            },
            'contrast_analysis': {
                'window_size': 64,
                'overlap_ratio': 0.5,
                'uniformity_threshold': 0.7,
                'over_enhancement_threshold': 2.0
            },
            'artifact_detection': {
                'blocking_tile_size': 8,
                'halo_kernel_size': 15,
                'over_sharpening_threshold': 1.5
            },
            'quality_thresholds': {
                'vessel_clarity_min': vessel_clarity_min,
                'detail_preservation_min': 0.8,
                'clinical_relevance_min': 0.6,
                'artifact_max': 0.3,
                'confidence_min': 0.7
            }
        }

    def _setup_training_config(self) -> Dict[str, Any]:
        """Configura parâmetros de treinamento com validação Fast Fail"""
        validation_split = 0.2
        test_split = 0.1

        # Fast Fail: Verificar se os splits somam menos de 1
        fast_fail_check(
            validation_split + test_split <= ConfigConstants.MAX_SPLIT_SUM,
            f"Splits de dados inválidos: soma ({validation_split + test_split}) deve ser <= {ConfigConstants.MAX_SPLIT_SUM}"
        )

        return {
            'positive_samples_per_image': 8,
            'negative_samples_per_image': 8,
            'degradation_types': [
                'blur', 'noise', 'brightness', 'contrast_loss',
                'compression', 'tile_artifacts'
            ],
            'enhancement_strength_range': (0.3, 0.8),
            'degradation_strength_range': (0.2, 0.6),
            'validation_split': validation_split,
            'test_split': test_split
        }

    @safe_operation(default_value={})
    def _setup_vit_config(self) -> Dict[str, Any]:
        """Configura parâmetros ViT com adaptação dinâmica baseada no Worker Manager"""
        # Configuração base
        base_config = {
            'model_architecture': {
                'patch_size': 16,
                'embed_dim': 768,
                'num_heads': 12,
                'num_layers': 12,
                'mlp_ratio': 4.0,
                'dropout': 0.1
            },
            'training_params': {
                'batch_size': ConfigConstants.BATCH_SIZE_MED_MEM,  # Valor padrão
                'learning_rate': 1e-4,
                'weight_decay': 0.05,
                'epochs': 100,
                'warmup_epochs': 10,
                'scheduler': 'cosine'
            },
            'data_augmentation': {
                'rotation_range': 15,
                'brightness_range': 0.2,
                'contrast_range': 0.2,
                'horizontal_flip': True,
                'vertical_flip': False,
                'gaussian_noise_std': 0.01
            }
        }

        # Ajustar com base no Worker Manager se disponível
        if WORKER_MANAGER_AVAILABLE:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Adaptar batch size baseado na configuração do worker
            batch_size = self._determine_optimal_batch_size(worker_config)
            base_config['training_params']['batch_size'] = batch_size

            # Adicionar informações do worker ao config
            base_config['worker_integration'] = {
                'device': worker_config.device,
                'mode': worker_config.mode.name,
                'cpu_workers': worker_config.cpu_workers,
                'gpu_workers': worker_config.gpu_workers,
                'use_async': worker_config.use_async
            }

            logger.info(f"ViT configurado com Worker Manager: {worker_config.mode.name}, batch_size={batch_size}")

        return base_config

    def _determine_optimal_batch_size(self, worker_config) -> int:
        """Determina o batch size ótimo baseado na configuração de workers (DRY)"""
        if worker_config.device == 'cpu':
            return min(ConfigConstants.BATCH_SIZE_LOW_MEM, worker_config.batch_size)
        elif worker_config.mode == WorkerMode.MULTI_GPU:
            return ConfigConstants.BATCH_SIZE_HIGH_MEM * worker_config.gpu_workers
        elif worker_config.memory_limit_gb >= ConfigConstants.HIGH_MEMORY_THRESHOLD_GB:
            return ConfigConstants.BATCH_SIZE_HIGH_MEM
        elif worker_config.memory_limit_gb >= ConfigConstants.MED_MEMORY_THRESHOLD_GB:
            return ConfigConstants.BATCH_SIZE_MED_MEM
        else:
            return ConfigConstants.BATCH_SIZE_LOW_MEM

    def _setup_paths(self) -> Dict[str, Path]:
        """Configura estrutura de diretórios com validação"""
        paths = {
            'input': Path('data/images'),
            'output': Path('processed'),
            'results': Path('results'),
            'visualizations': Path('visualizations'),
            'logs': Path('results/logs'),
            'parameter_analysis': Path('results/parameter_analysis'),
            'training_data': Path('data/training'),
            'models': Path('models'),
            'checkpoints': Path('models/checkpoints'),
            'predictions': Path('results/predictions')
        }

        # Fast Fail: Verificar permissões do diretório
        try:
            # Testar criação do diretório de saída (crítico)
            paths['output'].mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Erro de permissão ao criar diretório: {e}")

        return paths

    @safe_operation(default_value={})
    def _setup_processing_config_with_workers(self) -> Dict[str, Any]:
        """Configura parâmetros de processamento integrado com Worker Manager"""
        # Configuração base/fallback
        base_config = {
            'max_workers': 4,
            'memory_limit_gb': 8,
            'cache_results': True,
            'save_intermediate': False,
            'parallel_iqa': True,
            'optimization_timeout': 300,
            'use_worker_manager': WORKER_MANAGER_AVAILABLE,
            'device': 'cpu',
            'worker_mode': 'cpu_only',
            'batch_processing': False,
            'async_processing': False
        }

        # Integração com Worker Manager se disponível
        if WORKER_MANAGER_AVAILABLE:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Obter specs do sistema (método seguro com fallback)
            system_specs = self._get_system_specs_safely(worker_manager)

            # Atualizar configuração com dados do Worker Manager
            base_config.update({
                'max_workers': worker_config.cpu_workers,
                'memory_limit_gb': worker_config.memory_limit_gb,
                'device': worker_config.device,
                'worker_mode': worker_config.mode.name,
                'batch_processing': worker_config.batch_size > 1,
                'async_processing': worker_config.use_async,
                'gpu_workers': worker_config.gpu_workers,
                'parallel_iqa': system_specs.get('cpu_count', 2) > 2,

                # Configurações específicas do Worker Manager
                'worker_manager_config': {
                    'cpu_workers': worker_config.cpu_workers,
                    'gpu_workers': worker_config.gpu_workers,
                    'mode': worker_config.mode.name,
                    'batch_size': worker_config.batch_size,
                    'use_async': worker_config.use_async,
                    'system_specs': system_specs
                }
            })

            # Ajustes baseados no modo do worker
            if worker_config.mode == WorkerMode.CPU_ONLY:
                base_config['parallel_iqa'] = worker_config.cpu_workers > 1
            elif worker_config.mode in [WorkerMode.GPU_SHARED, WorkerMode.GPU_DEDICATED, WorkerMode.HYBRID_OPTIMIZED]:
                base_config['parallel_iqa'] = True
                base_config['optimization_timeout'] = 600  # Mais tempo para GPU

            logger.info(f"Processamento configurado: {worker_config.cpu_workers} CPU workers, "
                       f"{worker_config.gpu_workers} GPU workers, modo: {worker_config.mode.name}")

        return base_config

    @safe_operation(default_value={'cpu_count': 4, 'memory_available_gb': 8, 'gpu_available': False, 'gpu_count': 0})
    def _get_system_specs_safely(self, worker_manager) -> Dict[str, Any]:
        """Obtém especificações do sistema de forma segura"""
        # Tenta obter do worker_manager, com fallback para valores padrão
        if hasattr(worker_manager, 'system_specs'):
            return worker_manager.system_specs

        # Tenta obter hardware profile se disponível
        if hasattr(worker_manager, 'hardware_profile'):
            hw = worker_manager.hardware_profile
            return {
                'cpu_count': hw.cpu_cores,
                'memory_available_gb': hw.total_memory_gb,
                'gpu_available': hw.gpu_count > 0,
                'gpu_count': hw.gpu_count,
                'gpu_memory_gb': hw.gpu_memory_gb
            }

        # Fallback para valores obtidos do sistema
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3) if 'psutil' in globals() else 8,
            'gpu_available': torch.cuda.is_available() if 'torch' in globals() else False,
            'gpu_count': torch.cuda.device_count() if 'torch' in globals() else 0
        }

    def _setup_reporting_config(self) -> Dict[str, Any]:
        """Configura parâmetros de relatórios"""
        return {
            'generate_visualizations': True,
            'save_parameter_analysis': True,
            'detailed_metrics': True,
            'export_formats': ['txt', 'json', 'csv'],
            'include_sample_images': True,
            'dpi': 150
        }

    @safe_operation()
    def _load_config_file_safely(self, config_file: str):
        """Carrega configurações de arquivo de forma segura"""
        # Fast Fail: Verificar existência do arquivo
        file_path = Path(config_file)
        fast_fail_check(
            file_path.exists(),
            f"Arquivo de configuração não encontrado: {config_file}"
        )

        try:
            self.load_from_file(config_file)
            logger.info(f"Configuração carregada de {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao decodificar JSON em {config_file}: {e}")

    def _setup_derived_configs(self):
        """Configura parâmetros derivados com integração do Worker Manager"""
        # Total de variantes CLAHE
        clip_count = len(self.clahe_params['optimization_grid']['clip_limits'])
        tile_count = len(self.clahe_params['optimization_grid']['tile_grids'])
        self.total_clahe_variants = clip_count * tile_count

        # Ajustes baseados no Worker Manager
        self._adjust_configs_based_on_worker_manager()

    def _adjust_configs_based_on_worker_manager(self):
        """Ajusta configurações baseadas no Worker Manager (DRY)"""
        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            worker_config = self.processing_config['worker_manager_config']

            # Batch size ViT já foi configurado no setup_vit_config
            # Não precisamos refazer essa parte

            # Workers já foram configurados no setup_processing_config

        else:
            # Fallback para configuração manual
            memory_gb = self.processing_config['memory_limit_gb']

            # Determinar batch size baseado na memória disponível
            if memory_gb >= ConfigConstants.HIGH_MEMORY_THRESHOLD_GB:
                self.vit_config['training_params']['batch_size'] = ConfigConstants.BATCH_SIZE_HIGH_MEM
            elif memory_gb >= ConfigConstants.MED_MEMORY_THRESHOLD_GB:
                self.vit_config['training_params']['batch_size'] = ConfigConstants.BATCH_SIZE_MED_MEM
            else:
                self.vit_config['training_params']['batch_size'] = ConfigConstants.BATCH_SIZE_LOW_MEM

            # Workers baseado no hardware (fallback)
            max_cores = multiprocessing.cpu_count()
            self.processing_config['max_workers'] = min(
                self.processing_config['max_workers'], max_cores - 1
            )

    # ================================================================================================
    # API PÚBLICA - ACESSO A CONFIGURAÇÕES
    # ================================================================================================

    def get_worker_config(self) -> Dict[str, Any]:
        """Retorna configuração de workers (compatibilidade)"""
        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            return self.processing_config['worker_manager_config']
        else:
            return {
                'cpu_workers': self.processing_config['max_workers'],
                'gpu_workers': 0,
                'mode': 'cpu_only',
                'device': 'cpu',
                'batch_size': 1,
                'use_async': False
            }

    def get_max_workers(self) -> int:
        """Retorna número máximo de workers (compatibilidade com código existente)"""
        return self.processing_config['max_workers']

    def get_device(self) -> str:
        """Retorna device principal (compatibilidade)"""
        return self.processing_config['device']

    def should_use_async(self) -> bool:
        """Verifica se deve usar processamento assíncrono"""
        return self.processing_config.get('async_processing', False)

    def get_batch_size(self) -> int:
        """Retorna tamanho do batch para processamento"""
        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            worker_config = self.processing_config['worker_manager_config']
            return worker_config.get('batch_size', 1)
        return 1

    @performance_monitor
    def print_worker_summary(self):
        """Imprime resumo da configuração de workers"""
        print("\n🔧 CONFIGURAÇÃO DE WORKERS")
        print("-" * 40)

        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            worker_config = self.processing_config['worker_manager_config']
            system_specs = worker_config['system_specs']

            print(f"💻 Sistema:")
            print(f"   CPU: {system_specs['cpu_count']} cores")
            print(f"   RAM: {system_specs['memory_available_gb']:.1f}GB disponível")
            print(f"   GPU: {'✅' if system_specs['gpu_available'] else '❌'} "
                  f"({system_specs['gpu_count']} dispositivos)")

            print(f"\n⚙️ Workers (Worker Manager):")
            print(f"   CPU Workers: {worker_config['cpu_workers']}")
            print(f"   GPU Workers: {worker_config['gpu_workers']}")
            print(f"   Modo: {worker_config['mode']}")
            print(f"   Device: {self.processing_config['device']}")
            print(f"   Batch Size: {worker_config['batch_size']}")
            print(f"   Async: {'✅' if worker_config['use_async'] else '❌'}")
        else:
            print(f"⚙️ Workers (Fallback):")
            print(f"   CPU Workers: {self.processing_config['max_workers']}")
            print(f"   Device: {self.processing_config['device']}")
            print(f"   Worker Manager: ❌ Não disponível")

        print(f"\n📊 ViT:")
        print(f"   Batch Size: {self.vit_config['training_params']['batch_size']}")
        print(f"   Device: {self.processing_config['device']}")

    # ================================================================================================
    # VALIDAÇÃO E VERIFICAÇÃO
    # ================================================================================================

    @performance_monitor
    def validate(self) -> bool:
        """Validação rigorosa das configurações com Worker Manager"""
        errors = []

        # Validar diretório de entrada (caminho crítico)
        if not self.paths['input'].exists():
            errors.append(f"Diretório não encontrado: {self.paths['input']}")

        # Validar parâmetros CLAHE (já validados na inicialização - DRY)

        # Validar thresholds IQA (já validados na inicialização - DRY)

        # Validar splits (já validados na inicialização - DRY)

        # Verificar espaço em disco
        if not self._check_disk_space():
            errors.append("Espaço em disco insuficiente")

        # Validação específica do Worker Manager
        self._validate_worker_manager(errors)

        # Criar diretórios
        try:
            for path in self.paths.values():
                path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            errors.append(f"Erro de permissão: {e}")

        # Verificar imagens válidas
        valid_images = self._count_valid_images()
        if valid_images == 0:
            errors.append("Nenhuma imagem válida encontrada")

        if errors:
            logger.error("Falha na validação de configuração")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        # Imprimir resumo de sucesso
        worker_mode = "Worker Manager" if (WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']) else "Fallback"
        logger.info(f"Configuração validada ({worker_mode}): {valid_images} imagens, "
                    f"{self.total_clahe_variants} variantes CLAHE")

        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            self.print_worker_summary()

        return True

    @safe_operation(default_value=False)
    def _validate_worker_manager(self, errors: List[str]):
        """Valida configurações do Worker Manager (DRY)"""
        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Verificar se a configuração é viável
            if worker_config.memory_limit_gb < 1.0:
                errors.append("Memória insuficiente para Worker Manager")

            # Verificar se a configuração é consistente
            if worker_config.mode != WorkerMode.CPU_ONLY and not torch.cuda.is_available():
                errors.append("Modo GPU configurado, mas CUDA não está disponível")

    def _check_disk_space(self, min_gb: float = ConfigConstants.MIN_DISK_SPACE_GB) -> bool:
        """Verifica espaço em disco"""
        try:
            free_gb = shutil.disk_usage(self.paths['output']).free / (1024**3)
            return free_gb >= min_gb
        except Exception as e:
            logger.warning(f"Erro ao verificar espaço em disco: {e}")
            return True  # Assumir que tem espaço em caso de erro

    def _count_valid_images(self) -> int:
        """Conta imagens válidas com validação de existência do diretório"""
        if not self.paths['input'].exists():
            return 0

        valid_extensions = {ext.lower() for ext in self.clahe_params['extensions']}
        return sum(1 for f in self.paths['input'].iterdir()
                  if f.is_file() and f.suffix.lower() in valid_extensions)

    # ================================================================================================
    # PERSISTÊNCIA
    # ================================================================================================

    @performance_monitor
    def save_to_file(self, config_file: str) -> None:
        """Salva configurações em arquivo com validação Fast Fail"""
        # Fast Fail: Verificar permissão de escrita
        try:
            # Testar escrita em arquivo temporário
            temp_file = Path(config_file + ".tmp")
            with open(temp_file, 'w') as f:
                f.write("{}")
            temp_file.unlink()  # Remover arquivo temporário
        except (PermissionError, IOError) as e:
            raise IOError(f"Sem permissão para escrever em {config_file}: {e}")

        serializable = {}
        for key, value in self.__dict__.items():
            if key == 'paths':
                serializable[key] = {k: str(v) for k, v in value.items()}
            else:
                serializable[key] = value

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Configuração salva em {config_file}")

    def load_from_file(self, config_file: str) -> None:
        """Carrega configurações de arquivo com validação Fast Fail"""
        # Fast Fail: Verificar existência do arquivo
        file_path = Path(config_file)
        fast_fail_check(
            file_path.exists(),
            f"Arquivo de configuração não encontrado: {config_file}"
        )

        # Fast Fail: Verificar se é um JSON válido
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Formato JSON inválido em {config_file}: {e}")

        for key, value in loaded.items():
            if key == 'paths':
                self.paths.update({k: Path(v) for k, v in value.items()})
            elif hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

        # Reconfigurar parâmetros derivados após o carregamento
        self._setup_derived_configs()
        logger.info(f"Configuração carregada de {config_file}")

    # ================================================================================================
    # UTILITÁRIOS E RELATÓRIOS
    # ================================================================================================

    @performance_monitor
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Resumo das configurações de otimização com Worker Manager"""
        base_summary = {
            'total_clahe_variants': self.total_clahe_variants,
            'clip_limits_range': [
                min(self.clahe_params['optimization_grid']['clip_limits']),
                max(self.clahe_params['optimization_grid']['clip_limits'])
            ],
            'tile_grid_sizes': self.clahe_params['optimization_grid']['tile_grids'],
            'iqa_thresholds': self.iqa_config['quality_thresholds'],
            'parallel_workers': self.processing_config['max_workers'],
            'expected_training_samples': (
                self.training_data_config['positive_samples_per_image'] +
                self.training_data_config['negative_samples_per_image']
            ) * self._count_valid_images()
        }

        # Adicionar informações do Worker Manager se disponível
        if WORKER_MANAGER_AVAILABLE and self.processing_config['use_worker_manager']:
            worker_config = self.processing_config['worker_manager_config']
            base_summary.update({
                'worker_manager': {
                    'enabled': True,
                    'mode': worker_config['mode'],
                    'cpu_workers': worker_config['cpu_workers'],
                    'gpu_workers': worker_config['gpu_workers'],
                    'device': self.processing_config['device'],
                    'async_processing': worker_config['use_async'],
                    'batch_size': worker_config['batch_size']
                }
            })
        else:
            base_summary['worker_manager'] = {'enabled': False}

        return base_summary

# ================================================================================================
# CLASSES DE COMPATIBILIDADE E FUNÇÕES DE UTILIDADE
# ================================================================================================

class HRFConfig(EnhancedHRFConfig):
    """Compatibilidade com código existente"""
    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)

@safe_operation(default_value=None)
def create_default_config():
    """Cria configuração padrão com Worker Manager"""
    config = EnhancedHRFConfig()
    config.save_to_file('iqa_config.json')
    return config

@safe_operation(default_value=None)
def load_user_config(config_file: str = 'iqa_config.json'):
    """Carrega configuração do usuário com Fast Fail"""
    # Fast Fail: Verificar existência do arquivo
    if not Path(config_file).exists():
        logger.warning(f"Arquivo de configuração {config_file} não encontrado. Criando configuração padrão.")
        return create_default_config()

    return EnhancedHRFConfig(config_file)

@performance_monitor
def demo_config_with_workers():
    """Demonstração da configuração com Worker Manager"""
    print("🔧 DEMONSTRAÇÃO CONFIG + WORKER MANAGER")
    print("=" * 50)

    try:
        # Fast Fail: Verificar disponibilidade do Worker Manager
        if not WORKER_MANAGER_AVAILABLE:
            print("⚠️ Worker Manager não disponível - usando modo fallback")

        config = EnhancedHRFConfig(fast_mode=True)

        if config.validate():
            print("\n📊 Resumo da Otimização:")
            summary = config.get_optimization_summary()

            print(f"   CLAHE Variants: {summary['total_clahe_variants']}")
            print(f"   Workers: {summary['parallel_workers']}")

            if summary['worker_manager']['enabled']:
                wm = summary['worker_manager']
                print(f"   Worker Manager: ✅ {wm['mode']}")
                print(f"   CPU/GPU Workers: {wm['cpu_workers']}/{wm['gpu_workers']}")
                print(f"   Device: {wm['device']}")
            else:
                print(f"   Worker Manager: ❌ Fallback")

        else:
            print("❌ Falha na validação")

    except Exception as e:
        print(f"❌ Erro crítico: {e}")

if __name__ == "__main__":
    demo_config_with_workers()
