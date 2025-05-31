import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import logging
from contextlib import contextmanager
import threading
from functools import wraps

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Import Worker Manager se disponível
try:
    from worker_manager import (
        get_worker_manager,
        GPUContext,
        get_device,
        get_cpu_workers,
        should_use_async
    )
    WORKER_MANAGER_AVAILABLE = True
except ImportError:
    WORKER_MANAGER_AVAILABLE = False
    logger.warning("Worker Manager não disponível para ViT - usando configuração manual")

# Import utils para performance monitoring
try:
    from utils import (
        performance_monitor,
        worker_performance_context,
        optimize_batch_size_for_memory,
        estimate_gpu_memory_usage
    )
    UTILS_MONITORING_AVAILABLE = True
except ImportError:
    UTILS_MONITORING_AVAILABLE = False

class WorkerAwareViTMixin:
    """Mixin para integração de componentes ViT com Worker Manager"""

    def __init__(self):
        self.worker_manager_available = WORKER_MANAGER_AVAILABLE
        self.device_lock = threading.Lock()
        self._setup_worker_integration()

    def _setup_worker_integration(self):
        """Configura integração com Worker Manager"""
        if self.worker_manager_available:
            try:
                self.worker_manager = get_worker_manager()
                self.optimal_device = self.worker_manager.get_device()
                self.cpu_workers = self.worker_manager.get_cpu_workers()
                self.gpu_workers = self.worker_manager.get_gpu_workers()
                self.use_async = self.worker_manager.should_use_async()

                logger.info(f"ViT integrado com Worker Manager: {self.optimal_device}, "
                           f"{self.cpu_workers} CPU workers, {self.gpu_workers} GPU workers")
            except Exception as e:
                logger.warning(f"Falha na integração Worker Manager: {e}")
                self.worker_manager_available = False
                self._setup_fallback()
        else:
            self._setup_fallback()

    def _setup_fallback(self):
        """Configuração fallback sem Worker Manager"""
        self.worker_manager = None
        self.optimal_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu_workers = 4
        self.gpu_workers = 1 if torch.cuda.is_available() else 0
        self.use_async = False

        logger.info(f"ViT usando fallback: {self.optimal_device}")

    @contextmanager
    def worker_context(self, worker_id: int = 0):
        """Context manager para uso seguro de workers"""
        if self.worker_manager_available:
            with GPUContext(worker_id=worker_id) as device:
                yield device
        else:
            yield self.optimal_device

    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Determina batch size ótimo baseado no Worker Manager"""
        if self.worker_manager_available:
            try:
                worker_config = self.worker_manager.get_worker_config()

                # Ajustar baseado na configuração do worker
                if worker_config.mode.value == 'multi_gpu':
                    return base_batch_size * worker_config.gpu_workers
                elif worker_config.memory_limit_gb >= 16:
                    return min(64, base_batch_size * 2)
                elif worker_config.memory_limit_gb >= 8:
                    return base_batch_size
                else:
                    return max(8, base_batch_size // 2)
            except Exception:
                pass

        # Fallback baseado na memória GPU
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 16:
                return min(64, base_batch_size * 2)
            elif gpu_memory_gb >= 8:
                return base_batch_size
            else:
                return max(8, base_batch_size // 2)

        return 16  # CPU fallback

class PatchEmbedding(nn.Module, WorkerAwareViTMixin):
    """Converte imagem em sequência de patches embeddings com Worker Manager"""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 1, embed_dim: int = 768):
        nn.Module.__init__(self)
        WorkerAwareViTMixin.__init__(self)

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (batch, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)        # (batch, embed_dim, n_patches)
        return x.transpose(1, 2)  # (batch, n_patches, embed_dim)

class MultiHeadAttention(nn.Module, WorkerAwareViTMixin):
    """Multi-Head Attention para ViT com otimizações de Worker Manager"""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        nn.Module.__init__(self)
        WorkerAwareViTMixin.__init__(self)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention com otimizações para GPU se disponível
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Usar torch.backends.cudnn para otimização se GPU disponível
        if torch.cuda.is_available() and x.device.type == 'cuda':
            with torch.backends.cudnn.flags(enabled=True, benchmark=True):
                attention_probs = F.softmax(attention_scores, dim=-1)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, embed_dim)

        return self.projection(context), attention_probs

class TransformerBlock(nn.Module, WorkerAwareViTMixin):
    """Bloco Transformer individual com Worker Manager"""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        nn.Module.__init__(self)
        WorkerAwareViTMixin.__init__(self)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attention_out, attention_weights = self.attention(norm_x)
        x = x + attention_out

        # MLP
        norm_x = self.norm2(x)
        x = x + self.mlp(norm_x)

        return x, attention_weights

class RetinalIQAViT(nn.Module, WorkerAwareViTMixin):
    """Vision Transformer especializado para IQA retinal com Worker Manager"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 num_classes: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.1):
        nn.Module.__init__(self)
        WorkerAwareViTMixin.__init__(self)

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Class token e position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.patch_embedding.n_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Inicialização de pesos otimizada"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x)

        # Class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position embedding
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer blocks
        attention_weights_list = []
        for block in self.transformer_blocks:
            x, attention_weights = block(x)
            if return_attention:
                attention_weights_list.append(attention_weights)

        # Classificação
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.classifier(cls_token_final)

        if return_attention:
            return logits, attention_weights_list
        return logits

    def estimate_memory_usage(self, batch_size: int) -> Dict[str, float]:
        """Estima uso de memória para batch específico"""
        if UTILS_MONITORING_AVAILABLE:
            try:
                return estimate_gpu_memory_usage(batch_size, (224, 224))
            except Exception:
                pass

        # Estimativa simplificada
        base_memory_mb = 500  # Modelo base
        per_batch_mb = 50     # Por imagem no batch
        return {
            'estimated_memory_gb': (base_memory_mb + per_batch_mb * batch_size) / 1024,
            'recommendation': 'Reduza batch_size se out of memory' if batch_size > 32 else 'OK'
        }

class RetinalIQADataset(Dataset):
    """Dataset para treinamento ViT com dados IQA retinal otimizado"""
    def __init__(self, training_data: List[Dict], img_size: int = 224,
                 transform=None, augment: bool = True, worker_manager_available: bool = False):
        self.training_data = training_data
        self.img_size = img_size
        self.transform = transform
        self.augment = augment
        self.worker_manager_available = worker_manager_available

        if self.transform is None:
            self._setup_transforms()

    def _setup_transforms(self):
        """Configura transformações otimizadas"""
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]

        if self.augment:
            # Augmentations mais conservadoras para dados médicos
            transforms_list.extend([
                transforms.RandomRotation(degrees=10),  # Reduzido de 15
                transforms.RandomHorizontalFlip(p=0.3),  # Reduzido de 0.5
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduzido
            ])

        transforms_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        sample = self.training_data[idx]

        enhanced_image = sample['enhanced_image']
        label = sample['ground_truth_label']

        # Garantir uint8
        if enhanced_image.dtype != np.uint8:
            enhanced_image = (enhanced_image * 255).astype(np.uint8)

        # Adicionar canal se necessário
        if len(enhanced_image.shape) == 2:
            enhanced_image = np.expand_dims(enhanced_image, axis=2)

        if self.transform:
            enhanced_image = self.transform(enhanced_image)

        return enhanced_image, torch.tensor(label, dtype=torch.long)

class RetinalIQAViTTrainer:
    """Trainer para modelo ViT de IQA retinal com Worker Manager integrado"""
    def __init__(self, config: Dict, device: str = None):
        self.config = config
        self.worker_manager_available = WORKER_MANAGER_AVAILABLE

        # Setup device com Worker Manager
        self._setup_device_and_workers(device)

        # Inicializar modelo
        self._setup_model()

        # Configurações de treinamento adaptadas
        self._setup_training_config()

        # Histórico de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Performance tracking
        self.training_stats = {
            'total_epochs': 0,
            'best_val_accuracy': 0.0,
            'total_training_time': 0.0,
            'avg_epoch_time': 0.0,
            'worker_performance': {}
        }

    def _setup_device_and_workers(self, device: str = None):
        """Configura device e workers com Worker Manager"""
        if self.worker_manager_available:
            try:
                self.worker_manager = get_worker_manager()
                self.device = device or self.worker_manager.get_device()
                self.cpu_workers = self.worker_manager.get_cpu_workers()
                self.gpu_workers = self.worker_manager.get_gpu_workers()

                logger.info(f"ViT Trainer usando Worker Manager: {self.device}")

                # Verificar se device está disponível
                if 'cuda' in self.device and not torch.cuda.is_available():
                    logger.warning("CUDA solicitado mas não disponível, usando CPU")
                    self.device = 'cpu'

            except Exception as e:
                logger.warning(f"Erro no Worker Manager: {e}")
                self.worker_manager_available = False
                self._setup_fallback_device(device)
        else:
            self._setup_fallback_device(device)

    def _setup_fallback_device(self, device: str = None):
        """Configuração fallback para device"""
        self.worker_manager = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_workers = 4
        self.gpu_workers = 1 if torch.cuda.is_available() else 0

        logger.info(f"ViT Trainer usando fallback: {self.device}")

    def _setup_model(self):
        """Inicializa modelo ViT"""
        self.model = RetinalIQAViT(
            img_size=224,
            patch_size=self.config.get('patch_size', 16),
            embed_dim=self.config.get('embed_dim', 768),
            num_heads=self.config.get('num_heads', 12),
            num_layers=self.config.get('num_layers', 12),
            mlp_ratio=self.config.get('mlp_ratio', 4.0),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)

    def _setup_training_config(self):
        """Configura parâmetros de treinamento adaptativos"""
        # Learning rate adaptado ao Worker Manager
        base_lr = self.config.get('learning_rate', 1e-4)
        if self.worker_manager_available and self.gpu_workers > 1:
            # Ajustar learning rate para multi-GPU
            self.learning_rate = base_lr * np.sqrt(self.gpu_workers)
        else:
            self.learning_rate = base_lr

        self.weight_decay = self.config.get('weight_decay', 0.05)

        # Batch size otimizado
        base_batch_size = self.config.get('batch_size', 32)
        if hasattr(self.model, 'get_optimal_batch_size'):
            self.batch_size = self.model.get_optimal_batch_size(base_batch_size)
        else:
            self.batch_size = base_batch_size

        self.epochs = self.config.get('epochs', 100)

        # Otimizador e scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Configuração ViT: batch_size={self.batch_size}, lr={self.learning_rate:.2e}")

    def prepare_data(self, training_samples: List[Dict],
                    validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepara dados para treinamento e validação com Worker Manager"""
        print(f"📊 Preparando: {len(training_samples)} amostras")

        train_data, val_data = train_test_split(
            training_samples, test_size=validation_split, random_state=42,
            stratify=[s['ground_truth_label'] for s in training_samples]
        )

        print(f"   • Treino: {len(train_data)}, Validação: {len(val_data)}")

        # Datasets com informação do Worker Manager
        train_dataset = RetinalIQADataset(
            train_data, augment=True,
            worker_manager_available=self.worker_manager_available
        )
        val_dataset = RetinalIQADataset(
            val_data, augment=False,
            worker_manager_available=self.worker_manager_available
        )

        # DataLoaders otimizados para Worker Manager
        num_workers = min(self.cpu_workers, 8) if self.worker_manager_available else 4
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Treina por uma época com monitoramento de Worker Manager"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Context manager para performance se disponível
        context_manager = (worker_performance_context(f"ViT_epoch_{epoch}")
                          if UTILS_MONITORING_AVAILABLE else
                          performance_monitor(f"ViT_epoch_{epoch}"))

        with context_manager:
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Usar Worker Manager context se disponível
                if self.worker_manager_available:
                    with GPUContext(worker_id=batch_idx % self.gpu_workers) as device:
                        if device != 'cpu':
                            images, labels = images.to(device), labels.to(device)
                        else:
                            images, labels = images.to(self.device), labels.to(self.device)
                        loss, predicted = self._train_batch(images, labels)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)
                    loss, predicted = self._train_batch(images, labels)

                total_loss += loss
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 10 == 0:
                    print(f"   Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss:.4f}, Acc: {100.*correct/total:.2f}%")

                # Atualizar estatísticas do Worker Manager
                if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
                    batch_time = 0.1  # Estimativa simplificada
                    self.worker_manager.update_task_stats(batch_time, success=True)

        return total_loss / len(train_loader), 100. * correct / total

    def _train_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Treina um batch individual"""
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        _, predicted = outputs.max(1)
        return loss.item(), predicted

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Valida por uma época"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                # Usar Worker Manager context se disponível para validação
                if self.worker_manager_available:
                    with GPUContext(worker_id=batch_idx % max(self.gpu_workers, 1)) as device:
                        if device != 'cpu':
                            images, labels = images.to(device), labels.to(device)
                        else:
                            images, labels = images.to(self.device), labels.to(self.device)
                else:
                    images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(val_loader), 100. * correct / total

    def train(self, training_samples: List[Dict], save_path: Path) -> Dict[str, Any]:
        """Treinamento completo do modelo com Worker Manager"""
        print("🚀 INICIANDO TREINAMENTO ViT IQA COM WORKER MANAGER")
        print("-" * 50)

        # Verificar memória estimada
        if hasattr(self.model, 'estimate_memory_usage'):
            memory_est = self.model.estimate_memory_usage(self.batch_size)
            print(f"💾 Memória estimada: {memory_est['estimated_memory_gb']:.1f}GB")
            print(f"💡 {memory_est['recommendation']}")

        train_loader, val_loader = self.prepare_data(training_samples)

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 10

        start_time = time.time()

        for epoch in range(self.epochs):
            print(f"\n📅 Época {epoch+1}/{self.epochs}")

            # Treinar e validar
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader)

            self.scheduler.step()

            # Salvar histórico
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Log com informações do Worker Manager
            device_info = f" ({self.device})" if hasattr(self, 'device') else ""
            print(f"   Treino{device_info}: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"   Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Worker Manager stats se disponível
            if self.worker_manager_available and hasattr(self.worker_manager, 'get_stats'):
                try:
                    stats = self.worker_manager.get_stats()
                    if stats['total_tasks'] > 0:
                        print(f"   Workers: {stats['success_rate']:.1f}% sucesso, {stats['avg_task_time']:.2f}s médio")
                except Exception:
                    pass

            # Best model e early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"   ✅ Novo melhor: {val_acc:.2f}%")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   ⏹️ Early stopping após {patience} épocas")
                break

        # Restaurar melhor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        # Atualizar estatísticas de treinamento
        self.training_stats.update({
            'total_epochs': epoch + 1,
            'best_val_accuracy': best_val_acc,
            'total_training_time': training_time,
            'avg_epoch_time': training_time / (epoch + 1)
        })

        # Salvar modelo com informações do Worker Manager
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / "retinal_iqa_vit.pth"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_accuracy': best_val_acc,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            },
            'training_stats': self.training_stats,
            'worker_manager_info': {
                'available': self.worker_manager_available,
                'device': self.device,
                'cpu_workers': self.cpu_workers,
                'gpu_workers': self.gpu_workers,
                'batch_size_used': self.batch_size,
                'learning_rate_used': self.learning_rate
            }
        }

        torch.save(checkpoint, model_path)

        print(f"\n✅ TREINAMENTO CONCLUÍDO!")
        print(f"   • Tempo: {training_time/60:.1f} min")
        print(f"   • Melhor acurácia: {best_val_acc:.2f}%")
        print(f"   • Device: {self.device}")
        print(f"   • Workers: {self.cpu_workers} CPU / {self.gpu_workers} GPU")
        print(f"   • Modelo: {model_path}")

        # Gerar visualizações
        self._plot_training_history(save_path)

        return {
            'best_accuracy': best_val_acc,
            'training_time': training_time,
            'model_path': model_path,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            },
            'worker_manager_info': checkpoint['worker_manager_info'],
            'training_stats': self.training_stats
        }

    def _plot_training_history(self, save_path: Path):
        """Plota histórico de treinamento com informações do Worker Manager"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss
        ax1.plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val', linewidth=2)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Adicionar informações do Worker Manager como subtítulo
        worker_info = f"Device: {self.device} | Workers: {self.cpu_workers} CPU, {self.gpu_workers} GPU | Batch: {self.batch_size}"
        fig.suptitle(f'ViT Training History\n{worker_info}', fontsize=12)

        plt.tight_layout()

        plot_path = save_path / "training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   📊 Histórico: {plot_path}")

class RetinalIQAViTPredictor:
    """Preditor para modelo ViT treinado com Worker Manager integrado"""
    def __init__(self, model_path: Path, device: str = None):
        self.worker_manager_available = WORKER_MANAGER_AVAILABLE
        self._setup_device_and_workers(device)
        self._load_model(model_path)
        self._setup_transforms()

        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'avg_prediction_time': 0.0,
            'device_used': self.device
        }

    def _setup_device_and_workers(self, device: str = None):
        """Configura device com Worker Manager"""
        if self.worker_manager_available:
            try:
                self.worker_manager = get_worker_manager()
                self.device = device or self.worker_manager.get_device()
                self.cpu_workers = self.worker_manager.get_cpu_workers()
                self.gpu_workers = self.worker_manager.get_gpu_workers()

                logger.info(f"ViT Predictor usando Worker Manager: {self.device}")
            except Exception as e:
                logger.warning(f"Erro no Worker Manager: {e}")
                self.worker_manager_available = False
                self._setup_fallback_device(device)
        else:
            self._setup_fallback_device(device)

    def _setup_fallback_device(self, device: str = None):
        """Configuração fallback para device"""
        self.worker_manager = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_workers = 4
        self.gpu_workers = 1 if torch.cuda.is_available() else 0

    def _load_model(self, model_path: Path):
        """Carrega modelo com informações do Worker Manager"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']

        # Carregar informações do Worker Manager se disponível no checkpoint
        self.training_worker_info = checkpoint.get('worker_manager_info', {})

        self.model = RetinalIQAViT(
            img_size=224,
            patch_size=self.config.get('patch_size', 16),
            embed_dim=self.config.get('embed_dim', 768),
            num_heads=self.config.get('num_heads', 12),
            num_layers=self.config.get('num_layers', 12),
            mlp_ratio=self.config.get('mlp_ratio', 4.0),
            dropout=0.0  # Sem dropout na inferência
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Log informações do treinamento e inferência
        training_device = self.training_worker_info.get('device', 'unknown')
        training_workers = f"{self.training_worker_info.get('cpu_workers', 'unknown')} CPU / {self.training_worker_info.get('gpu_workers', 'unknown')} GPU"

        print(f"✅ Modelo ViT carregado: {model_path}")
        print(f"   • Acurácia: {checkpoint['best_val_accuracy']:.2f}%")
        print(f"   • Treinado em: {training_device} ({training_workers} workers)")
        print(f"   • Inferência em: {self.device}")

    def _setup_transforms(self):
        """Configura transformações para inferência"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict_enhancement_quality(self, original: np.ndarray,
                                  enhanced: np.ndarray) -> Dict[str, Any]:
        """Prediz qualidade do enhancement usando ViT com Worker Manager"""
        start_time = time.time()

        # Context manager para performance se disponível
        context_manager = (worker_performance_context("ViT_prediction")
                          if UTILS_MONITORING_AVAILABLE else
                          performance_monitor("ViT_prediction"))

        with context_manager:
            # Preparar imagem
            if enhanced.dtype != np.uint8:
                enhanced = (enhanced * 255).astype(np.uint8)

            if len(enhanced.shape) == 2:
                enhanced = np.expand_dims(enhanced, axis=2)

            enhanced_tensor = self.transform(enhanced).unsqueeze(0)

            # Usar Worker Manager context se disponível
            if self.worker_manager_available:
                with GPUContext(worker_id=0) as device:
                    if device != 'cpu':
                        enhanced_tensor = enhanced_tensor.to(device)
                        # Temporariamente mover modelo para o device correto
                        current_device = next(self.model.parameters()).device
                        if str(current_device) != device:
                            self.model = self.model.to(device)
                    prediction_result = self._predict_on_device(enhanced_tensor)
            else:
                enhanced_tensor = enhanced_tensor.to(self.device)
                prediction_result = self._predict_on_device(enhanced_tensor)

        # Atualizar estatísticas
        prediction_time = time.time() - start_time
        self._update_prediction_stats(prediction_time)

        # Adicionar informações do Worker Manager ao resultado
        prediction_result.update({
            'prediction_time': prediction_time,
            'device_used': self.device,
            'worker_manager_available': self.worker_manager_available
        })

        return prediction_result

    def _predict_on_device(self, enhanced_tensor: torch.Tensor) -> Dict[str, Any]:
        """Executa predição no device especificado"""
        with torch.no_grad():
            logits, attention_weights = self.model(enhanced_tensor, return_attention=True)
            probabilities = F.softmax(logits, dim=1)

            prob_not_effective = probabilities[0, 0].item()
            prob_effective = probabilities[0, 1].item()

            predicted_class = torch.argmax(probabilities, dim=1).item()
            enhancement_effective = bool(predicted_class)
            confidence = max(prob_effective, prob_not_effective)

            return {
                'enhancement_effective': enhancement_effective,
                'confidence_score': confidence,
                'probability_effective': prob_effective,
                'probability_not_effective': prob_not_effective,
                'vit_prediction': True,
                'attention_weights': attention_weights
            }

    def _update_prediction_stats(self, prediction_time: float):
        """Atualiza estatísticas de predição"""
        self.prediction_stats['total_predictions'] += 1

        # Média móvel do tempo de predição
        current_avg = self.prediction_stats['avg_prediction_time']
        total = self.prediction_stats['total_predictions']
        self.prediction_stats['avg_prediction_time'] = ((current_avg * (total - 1)) + prediction_time) / total

        # Atualizar Worker Manager se disponível
        if self.worker_manager_available and hasattr(self.worker_manager, 'update_task_stats'):
            self.worker_manager.update_task_stats(prediction_time, success=True)

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de predição"""
        stats = self.prediction_stats.copy()

        # Adicionar informações do Worker Manager se disponível
        if self.worker_manager_available and hasattr(self.worker_manager, 'get_stats'):
            try:
                worker_stats = self.worker_manager.get_stats()
                stats['worker_manager_stats'] = worker_stats
            except Exception:
                pass

        return stats

    def visualize_attention(self, enhanced: np.ndarray,
                          save_path: Path, image_name: str) -> None:
        """Visualiza mapas de atenção do ViT com informações do Worker Manager"""
        # Preparar imagem
        if enhanced.dtype != np.uint8:
            enhanced = (enhanced * 255).astype(np.uint8)

        if len(enhanced.shape) == 2:
            enhanced = np.expand_dims(enhanced, axis=2)

        enhanced_tensor = self.transform(enhanced).unsqueeze(0)

        # Context manager para visualização
        context_manager = (worker_performance_context("ViT_attention_visualization")
                          if UTILS_MONITORING_AVAILABLE else
                          performance_monitor("ViT_attention_visualization"))

        with context_manager:
            # Usar Worker Manager context se disponível
            if self.worker_manager_available:
                with GPUContext(worker_id=0) as device:
                    if device != 'cpu':
                        enhanced_tensor = enhanced_tensor.to(device)
                        current_device = next(self.model.parameters()).device
                        if str(current_device) != device:
                            self.model = self.model.to(device)
                    attention_weights = self._get_attention_weights(enhanced_tensor)
            else:
                enhanced_tensor = enhanced_tensor.to(self.device)
                attention_weights = self._get_attention_weights(enhanced_tensor)

        # Processar atenção
        last_attention = attention_weights[-1][0]  # Remove batch dimension
        avg_attention = last_attention.mean(0)     # Média sobre heads
        cls_attention = avg_attention[0, 1:]       # Atenção do class token para patches

        # Reshape para imagem
        grid_size = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        attention_map = attention_map.detach().cpu().numpy()

        # Redimensionar
        attention_resized = cv2.resize(attention_map, (224, 224))

        # Visualizar com informações do Worker Manager
        self._create_attention_plot(enhanced, attention_resized, save_path, image_name)

    def _get_attention_weights(self, enhanced_tensor: torch.Tensor):
        """Obtém pesos de atenção"""
        with torch.no_grad():
            _, attention_weights = self.model(enhanced_tensor, return_attention=True)
        return attention_weights

    def _create_attention_plot(self, enhanced: np.ndarray, attention_resized: np.ndarray,
                              save_path: Path, image_name: str):
        """Cria plot de atenção com informações do Worker Manager"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Imagem original
        axes[0].imshow(enhanced.squeeze(), cmap='gray')
        axes[0].set_title('Enhanced Image')
        axes[0].axis('off')

        # Mapa de atenção
        im1 = axes[1].imshow(attention_resized, cmap='hot')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Sobreposição
        axes[2].imshow(enhanced.squeeze(), cmap='gray', alpha=0.7)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.3)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        # Título com informações do Worker Manager
        device_info = f" (Device: {self.device})"
        if self.worker_manager_available:
            worker_info = f", Workers: {self.cpu_workers} CPU/{self.gpu_workers} GPU"
            device_info += worker_info

        plt.suptitle(f'ViT Attention - {image_name}{device_info}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Salvar
        attention_path = save_path / f"{image_name}_vit_attention.png"
        plt.savefig(attention_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   📊 Atenção: {attention_path}")

def integrate_vit_with_iqa_pipeline(config, training_dataset: List[Dict]) -> RetinalIQAViTPredictor:
    """Integra ViT ao pipeline IQA com Worker Manager"""
    print("🔗 INTEGRANDO ViT AO PIPELINE COM WORKER MANAGER")
    print("-" * 50)

    if len(training_dataset) < 50:
        print(f"⚠️ Dataset pequeno ({len(training_dataset)}). Recomendado: >200")

    # Configurações ViT com informações do Worker Manager
    vit_config = config.vit_config['model_architecture'].copy()
    vit_config.update(config.vit_config['training_params'])

    # Verificar se há configurações do Worker Manager no config
    if 'worker_integration' in config.vit_config:
        worker_integration = config.vit_config['worker_integration']
        print(f"🔧 Worker Manager detectado: {worker_integration['mode']}")
        print(f"   Device: {worker_integration['device']}")
        print(f"   Workers: {worker_integration['cpu_workers']} CPU / {worker_integration['gpu_workers']} GPU")

    # Ajustar para dataset pequeno
    if len(training_dataset) < 100:
        vit_config['batch_size'] = min(16, len(training_dataset) // 4)
        vit_config['epochs'] = min(20, vit_config['epochs'])
        print(f"⚙️ Ajustado para dataset pequeno: batch_size={vit_config['batch_size']}, epochs={vit_config['epochs']}")

    # Verificar Worker Manager
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            device = worker_manager.get_device()
            print(f"✅ Worker Manager ativo para ViT: {device}")
        except Exception as e:
            print(f"⚠️ Worker Manager com problemas: {e}")

    # Treinar
    trainer = RetinalIQAViTTrainer(vit_config)
    models_dir = config.paths['models']
    training_results = trainer.train(training_dataset, models_dir)

    # Criar predictor
    predictor = RetinalIQAViTPredictor(training_results['model_path'])

    # Resumo da integração
    worker_info = training_results.get('worker_manager_info', {})
    print(f"\n✅ ViT integrado!")
    print(f"   • Acurácia: {training_results['best_accuracy']:.2f}%")
    print(f"   • Tempo: {training_results['training_time']/60:.1f} min")
    print(f"   • Device: {worker_info.get('device', 'unknown')}")
    print(f"   • Workers: {worker_info.get('cpu_workers', 'unknown')} CPU / {worker_info.get('gpu_workers', 'unknown')} GPU")
    print(f"   • Batch Size: {worker_info.get('batch_size_used', 'unknown')}")

    return predictor

def demonstrate_vit_iqa():
    """Demonstra ViT para IQA com Worker Manager"""
    print("🤖 DEMONSTRAÇÃO ViT IQA COM WORKER MANAGER")
    print("-" * 50)

    # Verificar Worker Manager
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            print(f"✅ Worker Manager disponível: {worker_manager.get_device()}")
            worker_manager.print_configuration()
        except Exception as e:
            print(f"❌ Worker Manager com erro: {e}")
    else:
        print("❌ Worker Manager não disponível")

    # Dados sintéticos
    training_samples = []
    for i in range(100):
        base_image = np.random.randint(50, 200, (224, 224), dtype=np.uint8)

        # Estruturas vasculares simuladas
        center = (112, 112)
        for angle in np.linspace(0, 2*np.pi, 6):
            end = (int(center[0] + 80 * np.cos(angle)),
                   int(center[1] + 80 * np.sin(angle)))
            cv2.line(base_image, center, end, int(np.random.randint(30, 70)), 2)

        if i % 2 == 0:
            # Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0 + np.random.normal(0, 0.3),
                                  tileGridSize=(8, 8))
            enhanced = clahe.apply(base_image)
            label = 1
            desc = f"beneficial_{i}"
        else:
            # Degradação
            enhanced = cv2.GaussianBlur(base_image, (5, 5), 0)
            enhanced = np.clip(enhanced * 0.7, 0, 255).astype(np.uint8)
            label = 0
            desc = f"degradation_{i}"

        training_samples.append({
            'image_id': f"demo_{i:03d}",
            'enhanced_image': enhanced,
            'ground_truth_label': label,
            'description': desc,
            'confidence': 1.0
        })

    print(f"   ✅ {len(training_samples)} amostras geradas")

    # Configuração demo adaptada ao Worker Manager
    demo_config = {
        'patch_size': 16,
        'embed_dim': 384,
        'num_heads': 6,
        'num_layers': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 0.05,
        'batch_size': 16,
        'epochs': 10
    }

    # Ajustar baseado no Worker Manager se disponível
    if WORKER_MANAGER_AVAILABLE:
        try:
            worker_manager = get_worker_manager()
            worker_config = worker_manager.get_worker_config()

            # Ajustar batch size
            if worker_config.memory_limit_gb >= 8:
                demo_config['batch_size'] = 32
            elif worker_config.memory_limit_gb < 4:
                demo_config['batch_size'] = 8

            print(f"🎛️ Configuração ajustada pelo Worker Manager: batch_size={demo_config['batch_size']}")
        except Exception as e:
            print(f"⚠️ Erro no ajuste Worker Manager: {e}")

    # Treinar
    trainer = RetinalIQAViTTrainer(demo_config)
    save_path = Path("demo_vit_models")
    training_results = trainer.train(training_samples, save_path)

    # Testar
    predictor = RetinalIQAViTPredictor(training_results['model_path'])

    test_image = training_samples[0]['enhanced_image']
    test_original = np.random.randint(50, 200, (224, 224), dtype=np.uint8)

    prediction = predictor.predict_enhancement_quality(test_original, test_image)

    print(f"\n📊 Predição:")
    print(f"   • Enhancement: {prediction['enhancement_effective']}")
    print(f"   • Confiança: {prediction['confidence_score']:.3f}")
    print(f"   • Prob. efetivo: {prediction['probability_effective']:.3f}")
    print(f"   • Device usado: {prediction['device_used']}")
    print(f"   • Tempo: {prediction['prediction_time']:.3f}s")

    # Visualizar atenção
    vis_path = Path("demo_vit_visualizations")
    vis_path.mkdir(exist_ok=True)
    predictor.visualize_attention(test_image, vis_path, "demo_sample")

    # Estatísticas finais
    pred_stats = predictor.get_prediction_stats()
    training_info = training_results['worker_manager_info']

    print(f"\n✅ Demonstração concluída!")
    print(f"   • Modelo: {training_results['model_path']}")
    print(f"   • Acurácia: {training_results['best_accuracy']:.2f}%")
    print(f"   • Treinamento: {training_info['device']} ({training_info['cpu_workers']} CPU/{training_info['gpu_workers']} GPU)")
    print(f"   • Predições: {pred_stats['total_predictions']} realizadas")
    print(f"   • Tempo médio predição: {pred_stats['avg_prediction_time']:.3f}s")

if __name__ == "__main__":
    demonstrate_vit_iqa()
