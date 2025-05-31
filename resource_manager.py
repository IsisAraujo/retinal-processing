from typing import Dict, Any, Optional
import numpy as np

class ResourceManager:
    """Gerencia recursos computacionais para processamento eficiente"""

    def __init__(self):
        self.task_times = []
        self.current_workers = 4
        self.max_memory = None
        self.worker_manager = None

    def setup(self):
        """Configura o gerenciador de recursos"""
        try:
            from worker_manager import get_worker_manager
            self.worker_manager = get_worker_manager()
        except ImportError:
            print("⚠️ WorkerManager não disponível - usando configuração padrão")

    def get_optimal_workers(self) -> int:
        """Retorna número ótimo de workers"""
        if self.worker_manager:
            return self.worker_manager.get_cpu_workers()
        return self.current_workers

    def should_use_async(self) -> bool:
        """Determina se deve usar processamento assíncrono"""
        if self.worker_manager:
            return self.worker_manager.should_use_async()
        return False

    def get_batch_size(self) -> int:
        """Retorna tamanho ideal de lote"""
        if self.worker_manager:
            return self.worker_manager.get_batch_size()
        return 1

    def update_task_stats(self, processing_time: float, success: bool):
        """Atualiza estatísticas de processamento"""
        self.task_times.append(processing_time)
        if self.worker_manager:
            self.worker_manager.update_task_stats(processing_time, success)

    def get_resource_advice(self) -> Dict[str, Any]:
        """Obtém recomendações para otimização de recursos"""
        if not self.worker_manager:
            return {'action': 'maintain_current', 'suggested_workers': self.current_workers}

        if self.task_times and np.mean(self.task_times) > 5.0:
            return self.worker_manager.optimize_for_memory_usage()

        return {'action': 'maintain_current', 'suggested_workers': self.current_workers}

    def print_recommendations(self):
        """Exibe recomendações de recursos"""
        resource_advice = self.get_resource_advice()
        if resource_advice['action'] != 'maintain_current':
            print(f"⚠️ Recomendação: {resource_advice['action']} - "
                  f"Ajustar para {resource_advice['suggested_workers']} workers")
