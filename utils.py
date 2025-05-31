import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from contextlib import contextmanager
import psutil
import shutil
import threading
from functools import wraps

# Integração opcional com Worker Manager
WORKER_MANAGER_AVAILABLE = False
try:
    from worker_manager import GPUContext, get_worker_manager, get_device
    WORKER_MANAGER_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ==================================================
# VALIDAÇÃO E PRÉ-PROCESSAMENTO
# ==================================================

def validate_image(path: Path, min_size: Tuple[int, int] = (100, 100)) -> Dict[str, Any]:
    """Valida requisitos básicos de imagem retinal"""
    if not path.exists():
        return {'valid': False, 'error': 'File not found'}

    img = cv2.imread(str(path))
    if img is None:
        return {'valid': False, 'error': 'Invalid image format'}

    height, width = img.shape[:2]
    if width < min_size[0] or height < min_size[1]:
        return {'valid': False, 'error': f'Image too small: {width}x{height}'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mean_intensity = np.mean(gray)

    return {
        'valid': True,
        'width': width,
        'height': height,
        'mean_intensity': float(mean_intensity)
    }

def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """Extrai canal verde com ajuste para imagens retinianas"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        green = image[:, :, 1]
        return cv2.convertScaleAbs(green, alpha=1.2, beta=10) if np.std(green) < 20 else green
    return image

# ==================================================
# MONITORAMENTO DE PERFORMANCE
# ==================================================

@contextmanager
def performance_monitor(operation_name: str):
    """Monitora tempo e memória de operações"""
    start_time = time.perf_counter()
    start_mem = psutil.Process().memory_info().rss / 1024**2

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        end_mem = psutil.Process().memory_info().rss / 1024**2
        mem_delta = end_mem - start_mem
        logger.info(f"{operation_name}: {duration:.2f}s | ΔMem: {mem_delta:+.1f}MB")

# ==================================================
# MANIPULAÇÃO DE DADOS
# ==================================================

def safe_save_json(data: Dict, path: Path) -> bool:
    """Salva JSON com tratamento de erros e backup"""
    try:
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(data), f, indent=2)
        temp_path.replace(path)
        return True
    except Exception as e:
        logger.error(f"JSON save failed: {e}")
        return False

def convert_numpy_types(obj):
    """Converte tipos NumPy para nativos Python"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# ==================================================
# GERENCIAMENTO DE RECURSOS
# ==================================================

def get_system_resources() -> Dict[str, Any]:
    """Coleta informações de recursos do sistema"""
    mem = psutil.virtual_memory()
    return {
        'cpu_cores': psutil.cpu_count(logical=False),
        'memory_available_gb': mem.available / 1024**3,
        'gpu_available': torch.cuda.is_available() if WORKER_MANAGER_AVAILABLE else False
    }

def estimate_processing_time(num_images: int) -> Dict[str, Any]:
    """Estima tempo de processamento com otimização do Worker Manager"""
    base_time = num_images * 2.5  # 2.5s por imagem

    if WORKER_MANAGER_AVAILABLE:
        try:
            wm = get_worker_manager()
            workers = max(1, wm.get_cpu_workers())
            base_time /= workers
        except Exception:
            pass

    return {
        'total_seconds': base_time,
        'formatted': f"{base_time/60:.1f}min" if base_time > 60 else f"{base_time:.0f}s"
    }

# ==================================================
# DECORADORES E CONTEXTOS
# ==================================================

@contextmanager
def gpu_context(worker_id: int = 0):
    """Contexto para operações com GPU"""
    if WORKER_MANAGER_AVAILABLE:
        with GPUContext(worker_id=worker_id) as device:
            yield device
    else:
        yield 'cpu'

def monitor_performance(task_name: str):
    """Decorator para monitoramento de tarefas"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor(task_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# ==================================================
# CONSTANTES ESPECIALIZADAS
# ==================================================

DEFAULT_CLAHE_PARAMS = {
    'clip_limit': 2.0,
    'tile_grid_size': (8, 8)
}

IQA_THRESHOLDS = {
    'vessel_clarity_min': 1.1,
    'clinical_relevance_min': 0.6
}

# ==================================================
# UTILITÁRIOS DE VISUALIZAÇÃO
# ==================================================

def create_comparison_visual(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Cria visualização lado-a-lado para análise"""
    resized_orig = resize_to_match(original, enhanced.shape)
    return np.hstack((resized_orig, enhanced))

def resize_to_match(source: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """Redimensiona imagem preservando proporções"""
    height, width = target_shape[:2]
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_LANCZOS4)
