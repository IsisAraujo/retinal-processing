#!/usr/bin/env python3
"""
Worker Manager: Sistema centralizado para gerenciamento de workers CPU/GPU
com otimizações para processamento de imagens retinianas.
"""

import multiprocessing
import psutil
import torch
import logging
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, NamedTuple, List
from contextlib import contextmanager
from functools import wraps, lru_cache

# Configuração de logging
logger = logging.getLogger(__name__)

# ================================================================================================
# ESTRUTURAS DE DADOS E ENUMS
# ================================================================================================

class WorkerMode(Enum):
    """Modos de operação do Worker Manager"""
    CPU_ONLY = 1
    GPU_SHARED = 2  # 1 GPU compartilhada
    HYBRID_OPTIMIZED = 3  # CPU + GPU híbrido

@dataclass
class HardwareProfile:
    """Perfil de hardware detectado"""
    cpu_cores: int
    gpu_count: int
    total_memory_gb: float
    gpu_memory_gb: float
    gpu_name: str = "Unknown"

@dataclass
class WorkerConfig:
    """Configuração de workers"""
    cpu_workers: int
    gpu_workers: int
    mode: WorkerMode
    device: str
    memory_limit_gb: float
    batch_size: int = 32
    use_async: bool = False

class WorkerStats(NamedTuple):
    """Estatísticas de performance"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_task_time: float
    success_rate: float
    gpu_utilization: float
    memory_usage_gb: float

# ================================================================================================
# UTILITÁRIOS
# ================================================================================================

def fast_fail_check(condition: bool, message: str, exception_type: type = ValueError):
    """Verificação rápida com mensagem de erro"""
    if not condition:
        logger.error(f"Fast Fail: {message}")
        raise exception_type(message)

def safe_operation(default_value=None, log_errors=True):
    """Decorator para operações com fallback"""
    def decorator(func):
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
    """Decorator para monitoramento de performance"""
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
# ANÁLISE DE HARDWARE E CONFIGURAÇÃO
# ================================================================================================

class HardwareAnalyzer:
    """Analisador de hardware do sistema"""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_hardware_profile() -> HardwareProfile:
        """Detecta hardware disponível"""
        # CPU
        cpu_count = multiprocessing.cpu_count()

        # Memória
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)

        # GPU
        gpu_count = 0
        gpu_memory_gb = 0.0
        gpu_name = "None"

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                gpu_name = gpu_props.name

        return HardwareProfile(
            cpu_cores=cpu_count,
            gpu_count=gpu_count,
            total_memory_gb=total_memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            gpu_name=gpu_name
        )

    @staticmethod
    @safe_operation(default_value={})
    def get_current_usage() -> Dict[str, float]:
        """Retorna uso atual de recursos"""
        usage = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                usage.update({
                    'gpu_memory_allocated_gb': gpu_memory_allocated,
                    'gpu_memory_reserved_gb': gpu_memory_reserved,
                    'gpu_utilization': min(100.0, (gpu_memory_reserved / max(1.0, HardwareAnalyzer.get_hardware_profile().gpu_memory_gb)) * 100)
                })
            except Exception as e:
                logger.debug(f"Erro ao ler GPU stats: {e}")

        return usage

# ================================================================================================
# WORKER MANAGER PRINCIPAL
# ================================================================================================

class WorkerManager:
    """Gerenciador centralizado de workers CPU/GPU"""

    _instance: Optional['WorkerManager'] = None
    _lock = threading.RLock()

    def __new__(cls) -> 'WorkerManager':
        """Implementação Singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        logger.info("Inicializando WorkerManager...")

        # Análise de hardware e configuração
        self.hardware_profile = HardwareAnalyzer.get_hardware_profile()
        self.config = self._determine_optimal_config()

        # Controles de recursos
        self._gpu_locks = {}
        if self.hardware_profile.gpu_count > 0:
            self._gpu_locks = {f'gpu_{i}': threading.RLock() for i in range(self.hardware_profile.gpu_count)}

        self._resource_semaphore = threading.Semaphore(self.config.cpu_workers)

        # Estatísticas
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }
        self._stats_lock = threading.Lock()

        self._initialized = True
        logger.info(f"WorkerManager inicializado: {self.config.mode.name}, {self.config.cpu_workers} CPU workers, "
                    f"{self.config.gpu_workers} GPU workers, device={self.config.device}")

    def _determine_optimal_config(self) -> WorkerConfig:
        """Determina configuração ótima para o hardware disponível"""
        hw = self.hardware_profile
        current_usage = HardwareAnalyzer.get_current_usage()

        # CPU workers - deixar ao menos 1-2 cores livres
        cpu_workers = max(1, min(hw.cpu_cores - 2, 10))

        # GPU config
        gpu_workers = 0
        mode = WorkerMode.CPU_ONLY
        device = 'cpu'

        if hw.gpu_count > 0 and current_usage.get('gpu_utilization', 0) < 90:
            gpu_workers = 1
            mode = WorkerMode.GPU_SHARED
            device = 'cuda'

            # Se tiver muitos cores CPU e GPU, habilitar modo híbrido
            if hw.cpu_cores >= 8:
                mode = WorkerMode.HYBRID_OPTIMIZED

        # Limite de memória - 80% do disponível
        memory_limit = min(25.0, hw.total_memory_gb * 0.8)

        # Batch size baseado na GPU
        batch_size = 32
        if hw.gpu_memory_gb > 0:
            # ~150MB por imagem para ViT
            max_batch = int(hw.gpu_memory_gb * 0.8 * 1024 / 150)
            batch_size = min(64, max_batch)

        # Async processing para sistemas com muitos cores
        use_async = hw.cpu_cores >= 8 and hw.total_memory_gb >= 16.0

        return WorkerConfig(
            cpu_workers=cpu_workers,
            gpu_workers=gpu_workers,
            mode=mode,
            device=device,
            memory_limit_gb=memory_limit,
            batch_size=batch_size,
            use_async=use_async
        )

    # ================================================================================================
    # API PÚBLICA
    # ================================================================================================

    def get_cpu_workers(self) -> int:
        """Retorna número de workers CPU"""
        return self.config.cpu_workers

    def get_gpu_workers(self) -> int:
        """Retorna número de workers GPU"""
        return self.config.gpu_workers

    def get_device(self, worker_id: int = 0) -> str:
        """Retorna device para worker específico"""
        if self.config.mode == WorkerMode.CPU_ONLY:
            return 'cpu'

        # Multi-GPU support
        if self.config.gpu_workers > 1:
            gpu_id = worker_id % self.config.gpu_workers
            return f'cuda:{gpu_id}'

        return self.config.device

    def get_worker_config(self) -> WorkerConfig:
        """Retorna configuração de workers"""
        return self.config

    def should_use_async(self) -> bool:
        """Verifica se deve usar processamento assíncrono"""
        return self.config.use_async

    def get_batch_size(self) -> int:
        """Retorna tamanho do batch"""
        return self.config.batch_size

    def update_task_stats(self, task_time: float, success: bool = True):
        """Atualiza estatísticas de tarefas"""
        with self._stats_lock:
            self._stats['total_tasks'] += 1
            self._stats['total_time'] += task_time

            if success:
                self._stats['completed_tasks'] += 1
            else:
                self._stats['failed_tasks'] += 1

    def get_stats(self) -> WorkerStats:
        """Retorna estatísticas"""
        with self._stats_lock:
            stats = self._stats.copy()

        # Calcular métricas derivadas
        success_rate = 0.0
        avg_task_time = 0.0

        if stats['total_tasks'] > 0:
            success_rate = (stats['completed_tasks'] / stats['total_tasks']) * 100
            avg_task_time = stats['total_time'] / stats['total_tasks']

        # Uso atual
        current_usage = HardwareAnalyzer.get_current_usage()

        return WorkerStats(
            total_tasks=stats['total_tasks'],
            completed_tasks=stats['completed_tasks'],
            failed_tasks=stats['failed_tasks'],
            avg_task_time=avg_task_time,
            success_rate=success_rate,
            gpu_utilization=current_usage.get('gpu_utilization', 0.0),
            memory_usage_gb=current_usage.get('memory_available_gb', 0.0)
        )

    def optimize_for_memory_usage(self) -> Dict[str, Any]:
        """Recomendações baseadas no uso de memória"""
        current_usage = HardwareAnalyzer.get_current_usage()
        available_memory = current_usage.get('memory_available_gb', 0)

        recommendations = {
            'action': 'maintain_current',
            'suggested_workers': self.config.cpu_workers,
            'suggested_batch_size': self.config.batch_size
        }

        if available_memory < 4.0:
            # Memória crítica
            recommendations.update({
                'action': 'emergency_reduction',
                'suggested_workers': 1,
                'suggested_batch_size': 4
            })
        elif available_memory < 8.0:
            # Memória baixa
            recommendations.update({
                'action': 'moderate_reduction',
                'suggested_workers': max(1, self.config.cpu_workers // 2),
                'suggested_batch_size': max(8, self.config.batch_size // 2)
            })
        elif available_memory > 20.0:
            # Memória abundante
            recommendations.update({
                'action': 'scale_up',
                'suggested_workers': min(10, self.config.cpu_workers + 2),
                'suggested_batch_size': min(64, self.config.batch_size * 2)
            })

        return recommendations

    def force_cpu_mode(self):
        """Força modo CPU"""
        with self._lock:
            self.config = WorkerConfig(
                cpu_workers=self.config.cpu_workers,
                gpu_workers=0,
                mode=WorkerMode.CPU_ONLY,
                device='cpu',
                memory_limit_gb=self.config.memory_limit_gb,
                batch_size=max(8, self.config.batch_size // 2),
                use_async=False
            )
            logger.info("Modo CPU ativado")

    def print_configuration(self):
        """Imprime configuração atual"""
        hw = self.hardware_profile

        print("\n" + "="*50)
        print("🔧 CONFIGURAÇÃO WORKER MANAGER")
        print("="*50)

        print(f"Hardware:")
        print(f"  CPU: {hw.cpu_cores} cores")
        print(f"  RAM: {hw.total_memory_gb:.1f}GB")
        if hw.gpu_count > 0:
            print(f"  GPU: {hw.gpu_name} ({hw.gpu_memory_gb:.1f}GB VRAM)")
        else:
            print("  GPU: Não disponível")

        print(f"\nWorkers:")
        print(f"  CPU Workers: {self.config.cpu_workers}")
        print(f"  GPU Workers: {self.config.gpu_workers}")
        print(f"  Modo: {self.config.mode.name}")
        print(f"  Device: {self.config.device}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Async: {'✅' if self.config.use_async else '❌'}")
        print(f"  Memory Limit: {self.config.memory_limit_gb:.1f}GB")

        # Uso atual
        usage = HardwareAnalyzer.get_current_usage()
        print(f"\nUso Atual:")
        print(f"  CPU: {usage.get('cpu_percent', 0):.1f}%")
        print(f"  RAM: {usage.get('memory_percent', 0):.1f}% ({usage.get('memory_available_gb', 0):.1f}GB disponível)")
        if 'gpu_utilization' in usage:
            print(f"  GPU: {usage['gpu_utilization']:.1f}% ({usage.get('gpu_memory_reserved_gb', 0):.1f}GB)")

        # Estatísticas
        stats = self.get_stats()
        if stats.total_tasks > 0:
            print(f"\nEstatísticas:")
            print(f"  Tarefas: {stats.completed_tasks}/{stats.total_tasks} ({stats.success_rate:.1f}% sucesso)")
            print(f"  Tempo médio: {stats.avg_task_time:.3f}s")

        print("="*50)

# ================================================================================================
# CONTEXT MANAGERS
# ================================================================================================

class GPUContext:
    """Context manager para uso seguro de GPU"""

    def __init__(self, worker_id: int = 0, timeout: float = 30.0):
        self.worker_id = worker_id
        self.timeout = timeout
        self.manager = get_worker_manager()
        self.device = self.manager.get_device(worker_id)
        self.lock_acquired = False
        self._original_device = None

    def __enter__(self) -> str:
        """Configuração do context"""
        if 'cuda' not in self.device:
            return self.device

        # Verificar disponibilidade CUDA
        if not torch.cuda.is_available():
            return 'cpu'

        gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0

        # Verificar GPU ID
        if gpu_id >= torch.cuda.device_count():
            return 'cpu'

        # Tentar adquirir lock
        lock_key = f'gpu_{gpu_id}'
        if lock_key in self.manager._gpu_locks:
            self.lock_acquired = self.manager._gpu_locks[lock_key].acquire(timeout=self.timeout)

            if not self.lock_acquired:
                logger.warning(f"Timeout ao adquirir GPU {gpu_id}")
                return 'cpu'

        # Configurar GPU
        try:
            self._original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            return self.device
        except Exception as e:
            logger.error(f"Erro ao configurar GPU {gpu_id}: {e}")
            self._cleanup()
            return 'cpu'

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpeza do context"""
        self._cleanup()

    def _cleanup(self):
        """Libera recursos"""
        try:
            # Restaurar dispositivo original
            if self._original_device is not None and torch.cuda.is_available():
                torch.cuda.set_device(self._original_device)

            # Liberar lock
            if self.lock_acquired:
                gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                lock_key = f'gpu_{gpu_id}'
                if lock_key in self.manager._gpu_locks:
                    self.manager._gpu_locks[lock_key].release()

        except Exception as e:
            logger.warning(f"Erro na limpeza do GPUContext: {e}")

@contextmanager
def worker_performance_context(operation_name: str, worker_id: int = 0):
    """Context manager para monitoramento de performance"""
    manager = get_worker_manager()
    start_time = time.perf_counter()
    success = False

    try:
        yield {
            'worker_id': worker_id,
            'operation': operation_name,
            'device': manager.get_device(worker_id),
            'batch_size': manager.get_batch_size()
        }
        success = True

    except Exception as e:
        logger.error(f"Operação {operation_name} falhou: {e}")
        raise

    finally:
        elapsed_time = time.perf_counter() - start_time
        manager.update_task_stats(elapsed_time, success)

# ================================================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ================================================================================================

def get_worker_manager() -> WorkerManager:
    """Obtém instância do WorkerManager"""
    try:
        return WorkerManager()
    except Exception as e:
        logger.error(f"Falha ao obter WorkerManager: {e}")
        raise RuntimeError("WorkerManager não disponível") from e

def get_cpu_workers() -> int:
    """Retorna número de workers CPU"""
    return get_worker_manager().get_cpu_workers()

def get_gpu_workers() -> int:
    """Retorna número de workers GPU"""
    return get_worker_manager().get_gpu_workers()

def get_device(worker_id: int = 0) -> str:
    """Retorna device para worker específico"""
    return get_worker_manager().get_device(worker_id)

def should_use_async() -> bool:
    """Verifica se deve usar processamento assíncrono"""
    return get_worker_manager().should_use_async()

def print_worker_config():
    """Imprime configuração"""
    get_worker_manager().print_configuration()

# ================================================================================================
# PONTO DE ENTRADA
# ================================================================================================

def run_demo():
    """Demonstração simples do sistema"""
    print("🔧 DEMONSTRAÇÃO WORKER MANAGER")
    print("="*50)

    try:
        manager = get_worker_manager()
        manager.print_configuration()

        # Testar context
        print("\nTestando GPUContext...")
        with GPUContext(worker_id=0) as device:
            print(f"  Device ativo: {device}")

        # Testar estatísticas
        for i in range(3):
            with worker_performance_context(f"demo_task_{i}") as ctx:
                print(f"  Tarefa {i+1}: {ctx['device']}")
                time.sleep(0.1)

        # Mostrar estatísticas
        stats = manager.get_stats()
        print("\nEstatísticas:")
        print(f"  Total de tarefas: {stats.total_tasks}")
        print(f"  Taxa de sucesso: {stats.success_rate:.1f}%")
        print(f"  Tempo médio: {stats.avg_task_time:.3f}s")

    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_demo()
