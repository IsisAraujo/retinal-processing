import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from contextlib import contextmanager
import psutil
import shutil
import threading
from functools import wraps
import warnings

# Import Worker Manager se disponível
try:
    from worker_manager import get_worker_manager, GPUContext, get_device, get_cpu_workers
    WORKER_MANAGER_AVAILABLE = True
except ImportError:
    WORKER_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

def validate_image(image_path: Path, min_size: Tuple[int, int] = (100, 100),
                  max_size: Tuple[int, int] = (5000, 5000)) -> Dict[str, Any]:
    """Valida se uma imagem é adequada para processamento IQA"""
    try:
        if not image_path.exists():
            return {'valid': False, 'error': 'Arquivo não encontrado'}

        img = cv2.imread(str(image_path))
        if img is None:
            return {'valid': False, 'error': 'Não foi possível carregar a imagem'}

        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # Validações
        if width < min_size[0] or height < min_size[1]:
            return {'valid': False, 'error': f'Imagem muito pequena: {width}x{height}'}

        if width > max_size[0] or height > max_size[1]:
            return {'valid': False, 'error': f'Imagem muito grande: {width}x{height}'}

        if channels not in [1, 3]:
            return {'valid': False, 'error': f'Canais inválidos: {channels}'}

        if img.dtype != np.uint8:
            return {'valid': False, 'error': f'Tipo inválido: {img.dtype}'}

        # Estatísticas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 3 else img
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        if mean_intensity < 10:
            return {'valid': False, 'error': 'Imagem muito escura'}
        if mean_intensity > 245:
            return {'valid': False, 'error': 'Imagem muito clara'}
        if std_intensity < 5:
            return {'valid': False, 'error': 'Contraste insuficiente'}

        return {
            'valid': True,
            'width': width,
            'height': height,
            'channels': channels,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'file_size_mb': image_path.stat().st_size / (1024 * 1024),
            'dtype': str(img.dtype)
        }

    except Exception as e:
        return {'valid': False, 'error': f'Erro na validação: {str(e)}'}

def resize_image_smart(image: np.ndarray, target_size: Tuple[int, int],
                      preserve_aspect: bool = True) -> np.ndarray:
    """Redimensiona imagem preservando qualidade"""
    height, width = image.shape[:2]
    target_width, target_height = target_size

    if preserve_aspect:
        aspect_ratio = width / height
        target_aspect = target_width / target_height

        if aspect_ratio > target_aspect:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Criar canvas e centralizar
        if len(image.shape) == 3:
            canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_height, target_width), dtype=image.dtype)

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def extract_green_channel_enhanced(image: np.ndarray) -> np.ndarray:
    """Extrai canal verde com melhorias para imagens retinianas"""
    if len(image.shape) == 2:
        return image

    if image.shape[2] == 3:
        green = image[:, :, 1]
        # Realce se baixo contraste
        if np.std(green) < 20:
            green = cv2.convertScaleAbs(green, alpha=1.2, beta=10)
        return green
    else:
        raise ValueError(f"Formato não suportado: {image.shape}")

def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """Calcula métricas básicas de qualidade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Entropia
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Nitidez
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Contraste RMS
    rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))

    # Gradiente médio
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    return {
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'entropy': float(entropy),
        'laplacian_variance': float(laplacian_var),
        'rms_contrast': float(rms_contrast),
        'mean_gradient': float(mean_gradient)
    }

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager para monitorar performance com Worker Manager integrado"""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    # Obter informações do Worker Manager se disponível
    worker_info = {}
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_info = {
                'cpu_workers': worker_manager.get_cpu_workers(),
                'gpu_workers': worker_manager.get_gpu_workers(),
                'device': worker_manager.get_device(),
                'mode': worker_manager.get_worker_config().mode.value
            }
        except Exception:
            pass

    try:
        yield worker_info
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        # Log com informações do Worker Manager
        if worker_info:
            device_info = f" ({worker_info['device']}, {worker_info['cpu_workers']}w)"
        else:
            device_info = ""

        print(f"⏱️ {operation_name}{device_info}: {duration:.2f}s, Memória: {memory_delta:+.1f}MB")

        # Atualizar estatísticas do Worker Manager se disponível
        if WORKER_MANAGER_AVAILABLE:
            try:
                worker_manager = get_worker_manager()
                if hasattr(worker_manager, 'update_task_stats'):
                    worker_manager.update_task_stats(duration, success=True)
            except Exception:
                pass

def safe_save_json(data: Dict[str, Any], file_path: Path, backup: bool = True) -> bool:
    """Salva JSON com backup e conversão de tipos NumPy"""
    try:
        file_path = Path(file_path)

        # Backup se solicitado
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f'.backup_{int(time.time())}.json')
            shutil.copy2(file_path, backup_path)

        # Salvar temporário primeiro
        temp_path = file_path.with_suffix('.tmp')
        safe_data = numpy_to_python(data)

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(safe_data, f, indent=2, ensure_ascii=False)

        temp_path.replace(file_path)
        return True

    except Exception as e:
        print(f"❌ Erro ao salvar JSON {file_path}: {e}")

        # Fallback
        try:
            fallback_data = json.loads(json.dumps(data, default=str))
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
            print(f"⚠️ JSON salvo com fallback")
            return True
        except Exception as e2:
            print(f"❌ Falha no fallback: {e2}")
            return False

def load_json_safe(file_path: Path, default: Any = None) -> Any:
    """Carrega JSON de forma segura"""
    try:
        if not Path(file_path).exists():
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Erro ao carregar JSON {file_path}: {e}")
        return default

def check_system_resources() -> Dict[str, Any]:
    """Verifica recursos disponíveis do sistema com integração Worker Manager"""
    try:
        # CPU e Memória
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disco
        disk = psutil.disk_usage('.')

        # GPU
        gpu_available = False
        gpu_count = 0
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
        except ImportError:
            pass

        base_info = {
            'cpu_count': cpu_count,
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'disk_free_gb': disk.free / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'gpu_available': gpu_available,
            'gpu_count': gpu_count,
            'recommended_workers': min(cpu_count - 1, 8),
            'can_run_vit': memory.available / (1024**3) > 4.0 and (not gpu_available or gpu_count > 0)
        }

        # Integrar informações do Worker Manager se disponível
        if WORKER_MANAGER_AVAILABLE:
            try:
                worker_manager = get_worker_manager()
                worker_config = worker_manager.get_worker_config()
                worker_stats = worker_manager.get_stats()

                base_info.update({
                    'worker_manager': {
                        'available': True,
                        'cpu_workers': worker_config.cpu_workers,
                        'gpu_workers': worker_config.gpu_workers,
                        'mode': worker_config.mode.value,
                        'device': worker_config.device,
                        'memory_limit_gb': worker_config.memory_limit_gb,
                        'use_async': worker_config.use_async,
                        'batch_size': worker_config.batch_size,
                        'stats': {
                            'total_tasks': worker_stats['total_tasks'],
                            'success_rate': worker_stats['success_rate'],
                            'avg_task_time': worker_stats['avg_task_time']
                        }
                    }
                })
            except Exception as e:
                base_info['worker_manager'] = {'available': False, 'error': str(e)}
        else:
            base_info['worker_manager'] = {'available': False, 'reason': 'not_imported'}

        return base_info

    except Exception as e:
        return {
            'error': f'Erro ao verificar recursos: {e}',
            'cpu_count': 1,
            'memory_available_gb': 2.0,
            'disk_free_gb': 1.0,
            'gpu_available': False,
            'recommended_workers': 1,
            'can_run_vit': False,
            'worker_manager': {'available': False, 'error': str(e)}
        }

def format_bytes(bytes_value: int) -> str:
    """Formata bytes em unidades legíveis"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def format_duration(seconds: float) -> str:
    """Formata duração para formato legível"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def calculate_enhancement_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """Calcula estatísticas agregadas de enhancement"""
    if not metrics_list:
        return {}

    # Extrair métricas
    vessel_gains = [m.get('vessel_clarity_gain', 1.0) for m in metrics_list]
    clinical_scores = [m.get('clinical_relevance_score', 0.5) for m in metrics_list]
    confidence_scores = [m.get('confidence_score', 0.5) for m in metrics_list]
    effective_enhancements = [m.get('enhancement_effective', False) for m in metrics_list]

    stats = {
        'total_images': len(metrics_list),
        'effective_count': sum(effective_enhancements),
        'effectiveness_rate': sum(effective_enhancements) / len(effective_enhancements) * 100,

        'vessel_clarity': {
            'mean': np.mean(vessel_gains),
            'std': np.std(vessel_gains),
            'median': np.median(vessel_gains),
            'improvement_count': sum(1 for v in vessel_gains if v > 1.1)
        },

        'clinical_relevance': {
            'mean': np.mean(clinical_scores),
            'std': np.std(clinical_scores),
            'high_quality_count': sum(1 for c in clinical_scores if c > 0.7)
        },

        'confidence': {
            'mean': np.mean(confidence_scores),
            'high_confidence_count': sum(1 for c in confidence_scores if c > 0.8)
        }
    }

    # Classificação de qualidade
    if stats['effectiveness_rate'] >= 80:
        stats['quality_assessment'] = 'EXCELENTE'
    elif stats['effectiveness_rate'] >= 60:
        stats['quality_assessment'] = 'BOM'
    elif stats['effectiveness_rate'] >= 40:
        stats['quality_assessment'] = 'MODERADO'
    else:
        stats['quality_assessment'] = 'BAIXO'

    return stats

def validate_dataset_balance(training_data: List[Dict]) -> Dict[str, Any]:
    """Valida balanceamento do dataset de treinamento"""
    if not training_data:
        return {'balanced': False, 'error': 'Dataset vazio'}

    labels = [sample['ground_truth_label'] for sample in training_data]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    total_samples = len(labels)

    balance_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)
    is_balanced = balance_ratio >= 0.7
    is_sufficient = total_samples >= 50

    return {
        'balanced': is_balanced,
        'sufficient': is_sufficient,
        'total_samples': total_samples,
        'positive_samples': positive_count,
        'negative_samples': negative_count,
        'balance_ratio': balance_ratio,
        'recommendation': _get_dataset_recommendation(is_balanced, is_sufficient, balance_ratio, total_samples)
    }

def _get_dataset_recommendation(is_balanced: bool, is_sufficient: bool,
                               balance_ratio: float, total_samples: int) -> str:
    """Gera recomendação baseada no status do dataset"""
    if is_balanced and is_sufficient:
        return "Dataset adequado para treinamento"
    elif not is_sufficient:
        return f"Aumentar dataset para pelo menos 100 amostras (atual: {total_samples})"
    elif not is_balanced:
        if balance_ratio < 0.3:
            return "Balanceamento crítico - adicionar amostras da classe minoritária"
        else:
            return "Melhorar balanceamento"
    else:
        return "Dataset em boas condições"

def setup_experiment_directory(base_path: Path, experiment_name: str) -> Dict[str, Path]:
    """Configura estrutura de diretórios para experimento com informações do Worker Manager"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_path / f"{experiment_name}_{timestamp}"

    directories = {
        'experiment': experiment_dir,
        'data': experiment_dir / 'data',
        'models': experiment_dir / 'models',
        'results': experiment_dir / 'results',
        'visualizations': experiment_dir / 'visualizations',
        'logs': experiment_dir / 'logs',
        'configs': experiment_dir / 'configs',
        'worker_logs': experiment_dir / 'logs' / 'workers'  # NOVO: logs específicos de workers
    }

    # Criar diretórios
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # README do experimento com informações do Worker Manager
    worker_info = ""
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()
            worker_info = f"""
## Worker Manager Configuration
- **CPU Workers:** {worker_config.cpu_workers}
- **GPU Workers:** {worker_config.gpu_workers}
- **Mode:** {worker_config.mode.value}
- **Device:** {worker_config.device}
- **Memory Limit:** {worker_config.memory_limit_gb:.1f}GB
- **Async Processing:** {worker_config.use_async}
- **Batch Size:** {worker_config.batch_size}
"""
        except Exception as e:
            worker_info = f"\n## Worker Manager: Error - {e}\n"
    else:
        worker_info = "\n## Worker Manager: Not Available\n"

    readme_content = f"""# Experimento: {experiment_name}

## Informações Gerais
- **Data/Hora:** {time.strftime('%d/%m/%Y %H:%M:%S')}
- **Diretório:** {experiment_dir}

{worker_info}

## Estrutura
- `data/`: Dados de entrada e processados
- `models/`: Modelos treinados
- `results/`: Relatórios e métricas
- `visualizations/`: Gráficos e imagens
- `logs/`: Logs de execução
- `logs/workers/`: Logs específicos dos workers
- `configs/`: Arquivos de configuração
"""

    with open(experiment_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    return directories

def cleanup_temp_files(directories: List[Path], extensions: List[str] = ['.tmp', '.temp']) -> int:
    """Limpa arquivos temporários"""
    cleaned_count = 0

    for directory in directories:
        if not directory.exists():
            continue

        try:
            for extension in extensions:
                temp_files = list(directory.glob(f'*{extension}'))
                for temp_file in temp_files:
                    try:
                        temp_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️ Erro ao remover {temp_file}: {e}")
        except Exception as e:
            print(f"⚠️ Erro ao limpar {directory}: {e}")

    return cleaned_count

def estimate_processing_time(num_images: int, avg_time_per_image: float = 2.5) -> Dict[str, Any]:
    """Estima tempo de processamento considerando Worker Manager"""
    # Ajustar tempo baseado no Worker Manager se disponível
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Ajustar tempo baseado no número de workers
            efficiency_factor = min(worker_config.cpu_workers, 4) / 4.0  # Max efficiency com 4 workers

            # GPU acelera processamento
            if worker_config.gpu_workers > 0:
                efficiency_factor *= 1.5

            # Async melhora throughput
            if worker_config.use_async:
                efficiency_factor *= 1.2

            avg_time_per_image = avg_time_per_image / max(efficiency_factor, 0.3)

        except Exception:
            pass

    total_seconds = num_images * avg_time_per_image

    estimates = {
        'sample_analysis': total_seconds * 0.05,
        'batch_processing': total_seconds * 0.70,
        'vit_training': total_seconds * 0.20,
        'report_generation': total_seconds * 0.05,
        'total': total_seconds
    }

    formatted_estimates = {}
    for stage, seconds in estimates.items():
        formatted_estimates[stage] = {
            'seconds': seconds,
            'formatted': format_duration(seconds)
        }

    worker_info = {}
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()
            worker_info = {
                'optimization_applied': True,
                'cpu_workers': worker_config.cpu_workers,
                'gpu_workers': worker_config.gpu_workers,
                'efficiency_boost': f"{((2.5 / avg_time_per_image) - 1) * 100:.0f}%" if avg_time_per_image < 2.5 else "0%"
            }
        except Exception:
            worker_info = {'optimization_applied': False}
    else:
        worker_info = {'optimization_applied': False}

    return {
        'num_images': num_images,
        'estimates': formatted_estimates,
        'total_formatted': format_duration(total_seconds),
        'worker_optimization': worker_info
    }

def print_system_summary():
    """Imprime resumo do sistema com Worker Manager integrado"""
    print("🔧 RESUMO DO SISTEMA")
    print("-" * 30)

    info = check_system_resources()

    print(f"CPU: {info['cpu_count']} cores ({info['cpu_percent']:.1f}% uso)")
    print(f"RAM: {info['memory_available_gb']:.1f}/{info['memory_total_gb']:.1f}GB")
    print(f"Disco: {info['disk_free_gb']:.1f}GB livres")
    print(f"GPU: {'✅' if info['gpu_available'] else '❌'}")

    # Informações do Worker Manager
    wm_info = info.get('worker_manager', {})
    if wm_info.get('available'):
        print(f"\n🔧 WORKER MANAGER:")
        print(f"Workers: {wm_info['cpu_workers']} CPU / {wm_info['gpu_workers']} GPU")
        print(f"Modo: {wm_info['mode']}")
        print(f"Device: {wm_info['device']}")
        print(f"Batch Size: {wm_info['batch_size']}")
        print(f"Async: {'✅' if wm_info['use_async'] else '❌'}")

        if 'stats' in wm_info and wm_info['stats']['total_tasks'] > 0:
            stats = wm_info['stats']
            print(f"Estatísticas: {stats['success_rate']:.1f}% sucesso, {stats['avg_task_time']:.2f}s médio")
    else:
        print(f"\n🔧 WORKER MANAGER: ❌ {wm_info.get('reason', 'Não disponível')}")

    print(f"\n📋 RECOMENDAÇÕES:")
    if wm_info.get('available'):
        print(f"Workers otimizados automaticamente pelo Worker Manager")
    else:
        print(f"Workers: {info['recommended_workers']}")

    print(f"ViT: {'✅ Viável' if info['can_run_vit'] else '❌ Recursos insuficientes'}")

    if info['memory_available_gb'] < 4:
        print(f"⚠️ Memória baixa - feche outras aplicações")

    if info['disk_free_gb'] < 2:
        print(f"⚠️ Espaço baixo - libere pelo menos 2GB")

    print("-" * 30)

def numpy_to_python(obj):
    """Converte objetos NumPy para tipos nativos Python"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

# Decoradores para integração com Worker Manager
def with_worker_context(worker_id: int = 0):
    """Decorador para executar função com contexto de worker específico"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if WORKER_MANAGER_AVAILABLE:
                try:
                    with GPUContext(worker_id=worker_id) as device:
                        kwargs['device'] = device
                        return func(*args, **kwargs)
                except Exception:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def monitor_task_performance(task_name: str = None):
    """Decorador para monitorar performance de tarefas automaticamente"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = task_name or func.__name__
            with performance_monitor(name) as worker_info:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Atualizar estatísticas de falha no Worker Manager
                    if WORKER_MANAGER_AVAILABLE:
                        try:
                            worker_manager = get_worker_manager()
                            if hasattr(worker_manager, 'update_task_stats'):
                                worker_manager.update_task_stats(0, success=False)
                        except Exception:
                            pass
                    raise e
        return wrapper
    return decorator

# Funções de conveniência para Worker Manager
def get_optimal_workers() -> int:
    """Retorna número ótimo de workers baseado no Worker Manager"""
    if WORKER_MANAGER_AVAILABLE:
        try:
            return get_cpu_workers()
        except Exception:
            pass
    return min(psutil.cpu_count() - 1, 8)

def get_optimal_device(worker_id: int = 0) -> str:
    """Retorna device ótimo baseado no Worker Manager"""
    if WORKER_MANAGER_AVAILABLE:
        try:
            return get_device(worker_id)
        except Exception:
            pass

    # Fallback
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'

def create_worker_safe_environment():
    """Cria ambiente seguro para workers com locks apropriados"""
    return {
        'thread_lock': threading.Lock(),
        'worker_manager_available': WORKER_MANAGER_AVAILABLE,
        'optimal_workers': get_optimal_workers(),
        'optimal_device': get_optimal_device()
    }

# Constantes úteis atualizadas
DEFAULT_CLAHE_PARAMS = {
    'clip_limit': 2.0,
    'tile_grid_size': (8, 8)
}

SUPPORTED_IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp',
    '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.BMP'
]

IQA_QUALITY_THRESHOLDS = {
    'vessel_clarity_min': 1.1,
    'detail_preservation_min': 0.8,
    'clinical_relevance_min': 0.6,
    'artifact_max': 0.3,
    'confidence_min': 0.7
}

VIT_TRAINING_REQUIREMENTS = {
    'min_samples': 50,
    'recommended_samples': 200,
    'min_memory_gb': 4.0,
    'recommended_memory_gb': 8.0
}

# Configurações específicas do Worker Manager
WORKER_MANAGER_SETTINGS = {
    'performance_monitoring': True,
    'auto_optimization': True,
    'gpu_memory_threshold': 0.8,
    'cpu_usage_threshold': 0.9,
    'task_timeout_seconds': 300,
    'retry_failed_tasks': True,
    'max_retries': 3
}

def optimize_batch_size_for_memory(target_memory_gb: float = None) -> int:
    """Otimiza batch size baseado na memória disponível e Worker Manager"""
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Usar configuração do Worker Manager
            if worker_config.batch_size > 0:
                return worker_config.batch_size

            # Calcular baseado na memória disponível
            available_memory = worker_config.memory_limit_gb
        except Exception:
            available_memory = psutil.virtual_memory().available / (1024**3)
    else:
        available_memory = target_memory_gb or psutil.virtual_memory().available / (1024**3)

    # Heurística: ~1GB por batch item para ViT, ~0.5GB para processamento IQA
    if available_memory >= 16:
        return 8
    elif available_memory >= 8:
        return 4
    elif available_memory >= 4:
        return 2
    else:
        return 1

def estimate_gpu_memory_usage(batch_size: int, image_size: Tuple[int, int] = (224, 224)) -> float:
    """Estima uso de memória GPU para batch específico"""
    width, height = image_size

    # Estimativa para ViT: embedding + attention + MLP
    patch_size = 16
    num_patches = (width // patch_size) * (height // patch_size)
    embed_dim = 768
    num_heads = 12
    num_layers = 12

    # Memória por imagem (em MB)
    input_memory = width * height * 3 * 4 / (1024**2)  # RGB float32
    embedding_memory = num_patches * embed_dim * 4 / (1024**2)
    attention_memory = num_heads * num_patches * num_patches * 4 / (1024**2)
    mlp_memory = embed_dim * 4 * embed_dim * 4 / (1024**2)

    per_image_mb = input_memory + embedding_memory + attention_memory * num_layers + mlp_memory * num_layers
    total_gb = (per_image_mb * batch_size) / 1024

    # Adicionar overhead (gradientes, otimizador, etc.)
    return total_gb * 2.5

def create_worker_performance_report() -> Dict[str, Any]:
    """Cria relatório detalhado de performance dos workers"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_resources': check_system_resources(),
        'worker_manager_available': WORKER_MANAGER_AVAILABLE
    }

    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_stats = worker_manager.get_stats()
            worker_config = worker_manager.get_worker_config()

            report['worker_manager'] = {
                'configuration': {
                    'cpu_workers': worker_config.cpu_workers,
                    'gpu_workers': worker_config.gpu_workers,
                    'mode': worker_config.mode.value,
                    'device': worker_config.device,
                    'memory_limit_gb': worker_config.memory_limit_gb,
                    'use_async': worker_config.use_async,
                    'batch_size': worker_config.batch_size
                },
                'performance_stats': worker_stats,
                'recommendations': worker_manager.optimize_for_memory_usage()
            }

            # Calcular eficiência
            if worker_stats['total_tasks'] > 0:
                efficiency = {
                    'success_rate': worker_stats['success_rate'],
                    'avg_task_time': worker_stats['avg_task_time'],
                    'throughput_tasks_per_minute': 60 / max(worker_stats['avg_task_time'], 0.1),
                    'performance_rating': _calculate_performance_rating(worker_stats)
                }
                report['worker_manager']['efficiency'] = efficiency

        except Exception as e:
            report['worker_manager_error'] = str(e)

    return report

def _calculate_performance_rating(stats: Dict[str, Any]) -> str:
    """Calcula rating de performance baseado nas estatísticas"""
    success_rate = stats['success_rate']
    avg_time = stats['avg_task_time']

    if success_rate >= 95 and avg_time <= 2.0:
        return 'EXCELENTE'
    elif success_rate >= 90 and avg_time <= 3.0:
        return 'BOM'
    elif success_rate >= 80 and avg_time <= 5.0:
        return 'MODERADO'
    else:
        return 'PRECISA_MELHORAR'

def save_worker_performance_log(operation: str, duration: float, success: bool,
                               worker_id: int = 0, extra_info: Dict = None):
    """Salva log detalhado de performance para análise posterior"""
    log_entry = {
        'timestamp': time.time(),
        'operation': operation,
        'duration': duration,
        'success': success,
        'worker_id': worker_id,
        'extra_info': extra_info or {}
    }

    # Adicionar informações do sistema
    log_entry['system_info'] = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }

    # Adicionar informações do Worker Manager se disponível
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            device = worker_manager.get_device(worker_id)
            log_entry['worker_info'] = {
                'device': device,
                'worker_mode': worker_manager.get_worker_config().mode.value
            }
        except Exception:
            pass

    # Salvar em arquivo de log
    try:
        log_dir = Path('results/logs/workers')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"worker_performance_{time.strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, default=str)
            f.write('\n')

    except Exception as e:
        logger.warning(f"Falha ao salvar log de performance: {e}")

def analyze_worker_bottlenecks(log_file: Path = None) -> Dict[str, Any]:
    """Analisa logs de workers para identificar gargalos"""
    if not log_file:
        log_dir = Path('results/logs/workers')
        log_files = list(log_dir.glob("worker_performance_*.jsonl"))
        if not log_files:
            return {'error': 'Nenhum log de performance encontrado'}
        log_file = max(log_files, key=lambda x: x.stat().st_mtime)

    try:
        log_entries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        if not log_entries:
            return {'error': 'Nenhuma entrada de log válida'}

        # Análise de gargalos
        analysis = {
            'total_operations': len(log_entries),
            'success_rate': sum(1 for e in log_entries if e['success']) / len(log_entries) * 100,
            'avg_duration': np.mean([e['duration'] for e in log_entries]),
            'bottlenecks': []
        }

        # Identificar operações lentas
        durations = [e['duration'] for e in log_entries]
        slow_threshold = np.percentile(durations, 90)
        slow_operations = [e for e in log_entries if e['duration'] > slow_threshold]

        if slow_operations:
            analysis['bottlenecks'].append({
                'type': 'slow_operations',
                'count': len(slow_operations),
                'threshold': slow_threshold,
                'worst_operation': max(slow_operations, key=lambda x: x['duration'])
            })

        # Identificar falhas frequentes
        failed_operations = [e for e in log_entries if not e['success']]
        if len(failed_operations) > len(log_entries) * 0.1:  # >10% falhas
            analysis['bottlenecks'].append({
                'type': 'high_failure_rate',
                'failure_rate': len(failed_operations) / len(log_entries) * 100,
                'failed_operations': [e['operation'] for e in failed_operations]
            })

        # Análise por worker
        worker_analysis = {}
        for entry in log_entries:
            worker_id = entry.get('worker_id', 0)
            if worker_id not in worker_analysis:
                worker_analysis[worker_id] = {'count': 0, 'total_time': 0, 'failures': 0}

            worker_analysis[worker_id]['count'] += 1
            worker_analysis[worker_id]['total_time'] += entry['duration']
            if not entry['success']:
                worker_analysis[worker_id]['failures'] += 1

        analysis['worker_performance'] = {
            worker_id: {
                'avg_time': stats['total_time'] / stats['count'],
                'failure_rate': stats['failures'] / stats['count'] * 100,
                'efficiency_score': stats['count'] / max(stats['total_time'], 0.1)
            }
            for worker_id, stats in worker_analysis.items()
        }

        return analysis

    except Exception as e:
        return {'error': f'Erro na análise: {e}'}

def optimize_worker_configuration() -> Dict[str, Any]:
    """Otimiza configuração de workers baseado no histórico de performance"""
    if not WORKER_MANAGER_AVAILABLE:
        return {'error': 'Worker Manager não disponível'}

    try:
        # Analisar performance atual
        bottleneck_analysis = analyze_worker_bottlenecks()

        if 'error' in bottleneck_analysis:
            return bottleneck_analysis

        recommendations = []

        # Recomendar ajustes baseados na análise
        if bottleneck_analysis['success_rate'] < 90:
            recommendations.append({
                'issue': 'high_failure_rate',
                'recommendation': 'Reduzir número de workers ou batch size',
                'current_rate': bottleneck_analysis['success_rate'],
                'target_rate': 95
            })

        if bottleneck_analysis['avg_duration'] > 5.0:
            recommendations.append({
                'issue': 'slow_performance',
                'recommendation': 'Verificar uso de GPU ou reduzir complexidade',
                'current_avg': bottleneck_analysis['avg_duration'],
                'target_avg': 3.0
            })

        # Verificar se há workers desequilibrados
        worker_perf = bottleneck_analysis.get('worker_performance', {})
        if len(worker_perf) > 1:
            avg_times = [perf['avg_time'] for perf in worker_perf.values()]
            if max(avg_times) / min(avg_times) > 2.0:  # Diferença >100%
                recommendations.append({
                    'issue': 'unbalanced_workers',
                    'recommendation': 'Rebalancear carga entre workers',
                    'variation_factor': max(avg_times) / min(avg_times)
                })

        return {
            'analysis': bottleneck_analysis,
            'recommendations': recommendations,
            'optimization_applied': len(recommendations) == 0
        }

    except Exception as e:
        return {'error': f'Erro na otimização: {e}'}

@contextmanager
def worker_performance_context(operation_name: str, worker_id: int = 0):
    """Context manager avançado para monitoramento de performance com Worker Manager"""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    # Informações adicionais para o contexto
    context_info = {
        'operation': operation_name,
        'worker_id': worker_id,
        'start_time': start_time
    }

    # Obter informações do Worker Manager
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            context_info.update({
                'device': worker_manager.get_device(worker_id),
                'worker_mode': worker_manager.get_worker_config().mode.value,
                'cpu_workers': worker_manager.get_cpu_workers(),
                'gpu_workers': worker_manager.get_gpu_workers()
            })
        except Exception:
            pass

    success = False
    error_info = None

    try:
        yield context_info
        success = True

    except Exception as e:
        error_info = str(e)
        raise

    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        # Log detalhado
        extra_info = {
            'memory_delta_mb': memory_delta,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

        if error_info:
            extra_info['error'] = error_info

        # Salvar log de performance
        save_worker_performance_log(
            operation_name, duration, success, worker_id, extra_info
        )

        # Atualizar estatísticas do Worker Manager
        if WORKER_MANAGER_AVAILABLE:
            try:
                worker_manager = get_worker_manager()
                if hasattr(worker_manager, 'update_task_stats'):
                    worker_manager.update_task_stats(duration, success)
            except Exception:
                pass

        # Log console com informações enriquecidas
        status = "✅" if success else "❌"
        device_info = f" ({context_info.get('device', 'unknown')})" if 'device' in context_info else ""

        print(f"{status} {operation_name}{device_info}: {duration:.2f}s, "
              f"Mem: {memory_delta:+.1f}MB")

def demo_worker_integration():
    """Demonstra integração completa do utils.py com Worker Manager"""
    print("🧪 DEMONSTRAÇÃO UTILS.PY + WORKER MANAGER")
    print("=" * 50)

    # 1. Verificar recursos do sistema
    print("\n1️⃣ Verificando recursos do sistema...")
    resources = check_system_resources()

    print(f"CPU: {resources['cpu_count']} cores")
    print(f"RAM: {resources['memory_available_gb']:.1f}GB disponível")
    print(f"Worker Manager: {'✅' if resources['worker_manager']['available'] else '❌'}")

    if resources['worker_manager']['available']:
        wm = resources['worker_manager']
        print(f"  - Workers: {wm['cpu_workers']} CPU / {wm['gpu_workers']} GPU")
        print(f"  - Modo: {wm['mode']}")
        print(f"  - Device: {wm['device']}")

    # 2. Testar performance monitoring
    print("\n2️⃣ Testando monitoramento de performance...")

    @monitor_task_performance("demo_task")
    def demo_processing_task():
        time.sleep(0.5)  # Simular processamento
        return "resultado_processamento"

    result = demo_processing_task()
    print(f"Resultado: {result}")

    # 3. Testar contexto de worker
    print("\n3️⃣ Testando contexto de worker...")

    with worker_performance_context("demo_context_task", worker_id=0) as ctx:
        device = ctx.get('device', 'cpu')
        print(f"Executando no device: {device}")
        time.sleep(0.2)

    # 4. Estimativa de tempo otimizada
    print("\n4️⃣ Estimativa de processamento...")
    estimate = estimate_processing_time(10)

    print(f"10 imagens estimadas em: {estimate['total_formatted']}")
    if estimate['worker_optimization']['optimization_applied']:
        print(f"Boost de eficiência: {estimate['worker_optimization']['efficiency_boost']}")

    # 5. Relatório de performance
    print("\n5️⃣ Relatório de performance...")
    report = create_worker_performance_report()

    if 'worker_manager' in report:
        perf = report['worker_manager']['performance_stats']
        print(f"Tasks processadas: {perf['total_tasks']}")
        print(f"Taxa de sucesso: {perf['success_rate']:.1f}%")
        print(f"Tempo médio: {perf['avg_task_time']:.2f}s")

    # 6. Configuração otimizada
    print("\n6️⃣ Configurações otimizadas...")
    optimal_workers = get_optimal_workers()
    optimal_device = get_optimal_device()
    optimal_batch = optimize_batch_size_for_memory()

    print(f"Workers ótimos: {optimal_workers}")
    print(f"Device ótimo: {optimal_device}")
    print(f"Batch size ótimo: {optimal_batch}")

    print("\n✅ Demonstração concluída!")

if __name__ == "__main__":
    demo_worker_integration()
