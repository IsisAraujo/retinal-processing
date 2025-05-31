import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch

from enhanced_metrics import RetinalIQAMetrics, GroundTruthGenerator
from utils import convert_numpy_types, safe_save_json
from clahe_optimizer import CLAHEOptimizer
from resource_manager import ResourceManager
from batch_processor import BatchProcessor

class EnhancedRetinalProcessor:
    """Processador retinal com IQA integrado"""

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iqa_metrics = RetinalIQAMetrics(device=self.device)
        self.gt_generator = GroundTruthGenerator()

        # Inicializar componentes
        self.optimizer = CLAHEOptimizer(config, self.iqa_metrics)
        self.resource_manager = ResourceManager()
        self.resource_manager.setup()
        self.batch_processor = BatchProcessor(self, self.resource_manager)

    def process_image_with_iqa(self, path: Path) -> Dict[str, Any]:
        """Processa imagem com avaliação IQA completa"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Falha ao carregar: {path}")

        green = self._extract_green(img)
        best_result = self.optimizer.find_optimal_params(green, path.stem)
        iqa_metrics = self.iqa_metrics.calculate_comprehensive_metrics(
            green, best_result['enhanced']
        )

        # Gerar dados de treinamento se necessário
        training_data = None
        if hasattr(self.config, 'generate_training_data') and self.config.generate_training_data:
            training_data = self._generate_training_data(green, path.stem)

        return {
            'path': path,
            'original': green,
            'enhanced': best_result['enhanced'],
            'optimal_params': best_result['params'],
            'iqa_metrics': iqa_metrics,
            'enhancement_effective': iqa_metrics['enhancement_effective'],
            'confidence_score': iqa_metrics['confidence_score'],
            'training_data': training_data
        }

    def _generate_training_data(self, image: np.ndarray, image_id: str) -> Dict[str, Any]:
        """Gera dados de treinamento para ViT"""
        training_pairs = self.gt_generator.generate_training_pairs(
            image, n_positive=8, n_negative=8
        )

        labeled_data = []
        for enhanced_img, label, description in training_pairs:
            metrics = self.iqa_metrics.calculate_comprehensive_metrics(image, enhanced_img)
            labeled_data.append({
                'image_id': f"{image_id}_{description}",
                'enhanced_image': enhanced_img,
                'ground_truth_label': label,
                'description': description,
                'iqa_metrics': metrics,
                'enhancement_predicted': metrics['enhancement_effective'],
                'confidence': metrics['confidence_score']
            })

        positive_samples = sum(1 for item in labeled_data if item['ground_truth_label'] == 1)

        return {
            'samples': labeled_data,
            'statistics': {
                'total_samples': len(labeled_data),
                'positive_samples': positive_samples,
                'negative_samples': len(labeled_data) - positive_samples,
                'balance_ratio': positive_samples / max(len(labeled_data) - positive_samples, 1)
            }
        }

    def process_batch_with_iqa(self, max_workers: int = 4, batch_size: int = 1, use_async: bool = False) -> Dict[str, Any]:
        """Processamento em lote com IQA usando gerenciamento de recursos

        Args:
            max_workers: Número máximo de workers paralelos
            batch_size: Tamanho do lote para processamento
            use_async: Usar processamento assíncrono

        Returns:
            Dicionário com resultados do processamento em lote
        """
        # Configurar gerenciador de recursos com valores iniciais
        if max_workers != 4:  # Se não for o valor padrão
            self.resource_manager.current_workers = max_workers

        # Encontrar imagens
        images = self._find_images()

        # Delegar processamento para BatchProcessor
        return self.batch_processor.process_batch(images)

    def _save_processed_image(self, result: Dict[str, Any]) -> None:
        """Salva imagem processada com metadados"""
        output_dir = self.config.paths['output']
        output_dir.mkdir(exist_ok=True)

        # Salvar imagem
        output_path = output_dir / result['path'].name
        cv2.imwrite(str(output_path), result['enhanced'])

        # Salvar metadados
        metadata_path = output_dir / f"{result['path'].stem}_iqa_metadata.json"
        metadata = {
            'original_file': str(result['path']),
            'processed_file': str(output_path),
            'optimal_parameters': result['optimal_params'],
            'iqa_assessment': {
                'enhancement_effective': result['enhancement_effective'],
                'confidence_score': result['confidence_score'],
                'clinical_relevance': result['iqa_metrics']['clinical_relevance_score']
            },
            'key_metrics': {
                k: v for k, v in result['iqa_metrics'].items()
                if k in ['vessel_clarity_gain', 'detail_preservation_score',
                        'perceptual_quality_score', 'total_artifact_score']
            }
        }

        safe_save_json(convert_numpy_types(metadata), metadata_path)

    def _extract_green(self, image: np.ndarray) -> np.ndarray:
        """Extrai canal verde com validação"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Imagem deve estar no formato BGR")
        return image[:, :, 1]

    def _find_images(self) -> List[Path]:
        """Busca imagens válidas"""
        exts = {ext.lower() for ext in self.config.clahe_params['extensions']}
        return [f for f in self.config.paths['input'].iterdir()
                if f.is_file() and f.suffix.lower() in exts]

class IQAReportGenerator:
    """Gerador de relatórios IQA"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def generate_iqa_report(self, batch_results: Dict[str, Any],
                           config) -> Tuple[Path, Path]:
        """Gera relatório científico focado em IQA"""
        report_path = self.output_dir / f"iqa_report_{self.timestamp}.txt"
        json_path = self.output_dir / f"iqa_data_{self.timestamp}.json"

        # Relatório em texto
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_report(batch_results, config))

        # Dados estruturados
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self._prepare_json_data(batch_results)), f, indent=2)

        print(f"📄 Relatório: {report_path}")
        return report_path, json_path

    def _generate_text_report(self, results: Dict[str, Any], config) -> str:
        """Gera relatório em texto"""
        sections = [
            "=" * 80,
            "RELATÓRIO DE IMAGE QUALITY ASSESSMENT (IQA)",
            "=" * 80,
            f"Data/Hora: {time.strftime('%d/%m/%Y %H:%M:%S')}",
            f"Dataset: {config.paths['input']}",
            "",
            "RESUMO EXECUTIVO",
            "-" * 40,
            f"Imagens processadas: {results['successful_processing']}",
            f"Enhancements efetivos: {results['effective_enhancements']}",
            f"Taxa de efetividade: {results['effective_enhancements']/max(results['successful_processing'], 1)*100:.1f}%",
            f"Dados de treinamento: {len(results.get('training_dataset', []))} amostras",
            ""
        ]

        # Estatísticas detalhadas
        if 'summary_statistics' in results and 'metrics_statistics' in results['summary_statistics']:
            sections.append("MÉTRICAS AGREGADAS")
            sections.append("-" * 40)

            stats = results['summary_statistics']['metrics_statistics']
            for metric_name, metric_stats in stats.items():
                sections.extend([
                    f"{metric_name.replace('_', ' ').title()}:",
                    f"  Média: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}",
                    f"  Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]",
                    ""
                ])

        sections.extend([
            "=" * 80,
            f"Relatório gerado em {time.strftime('%d/%m/%Y %H:%M:%S')}",
            "=" * 80
        ])

        return "\n".join(sections)

    def _prepare_json_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara dados para JSON"""
        serializable_results = {}

        for key, value in results.items():
            if key == 'training_dataset':
                serializable_training = []
                for sample in value:
                    serializable_sample = sample.copy()
                    # Remover imagens numpy (muito grandes)
                    if 'enhanced_image' in serializable_sample:
                        del serializable_sample['enhanced_image']
                    serializable_training.append(serializable_sample)
                serializable_results[key] = serializable_training
            else:
                serializable_results[key] = value

        return {
            'timestamp': self.timestamp,
            'report_type': 'Image Quality Assessment',
            'methodology': 'Enhanced CLAHE with automated IQA validation',
            'results': serializable_results
        }
