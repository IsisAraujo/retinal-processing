#!/usr/bin/env python3
"""
Multi-Image Parameter Aggregation and Statistical Validation
Sistema para consolidação de análises paramétricas individuais
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from utils import numpy_to_python
warnings.filterwarnings('ignore')

class ParameterAggregator:
    """Agregador de análises paramétricas multi-imagem"""

    def __init__(self, analysis_dir: Path):
        self.analysis_dir = Path(analysis_dir)
        self.results_dir = self.analysis_dir / "aggregated_results"
        self.results_dir.mkdir(exist_ok=True)

        # Configurações estatísticas
        self.confidence_level = 0.95
        self.significance_threshold = 0.05

    def aggregate_parameter_analyses(self) -> Dict[str, Any]:
        """Agrega todas as análises individuais"""
        print("🔄 AGREGANDO ANÁLISES PARAMÉTRICAS")
        print("-" * 50)

        # Carregar todos os arquivos de análise
        analysis_files = list(self.analysis_dir.glob("*_parameter_analysis.json"))

        if not analysis_files:
            raise ValueError(f"Nenhuma análise encontrada em {self.analysis_dir}")

        print(f"📁 Encontrados: {len(analysis_files)} arquivos")

        aggregated_data = {
            'images': [],
            'all_results': [],
            'best_configs_per_image': [],
            'parameter_statistics': {},
            'correlation_matrix': None,
            'clustering_results': None
        }

        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                image_id = data['image_id']
                aggregated_data['images'].append(image_id)

                # Melhor configuração para esta imagem
                best_config = data['results'][0]  # Já ordenado por score
                aggregated_data['best_configs_per_image'].append({
                    'image_id': image_id,
                    'best_params': best_config['params'],
                    'best_score': best_config['composite_score'],
                    'best_metrics': best_config['key_metrics']
                })

                # Todos os resultados para análise estatística
                for result in data['results']:
                    result_entry = {
                        'image_id': image_id,
                        'clip_limit': result['params']['clip_limit'],
                        'tile_grid_x': result['params']['tile_grid'][0],
                        'tile_grid_y': result['params']['tile_grid'][1],
                        'composite_score': result['composite_score'],
                        'clinical_relevance': result['key_metrics']['clinical_relevance_score'],
                        'vessel_clarity_gain': result['key_metrics']['vessel_clarity_gain'],
                        'confidence_score': result['key_metrics']['confidence_score'],
                        'enhancement_effective': result['key_metrics']['enhancement_effective']
                    }
                    aggregated_data['all_results'].append(result_entry)

                print(f"  ✅ {image_id}: {len(data['results'])} configurações")

            except Exception as e:
                print(f"  ❌ Erro em {file_path.name}: {e}")

        # Análises estatísticas
        aggregated_data['parameter_statistics'] = self._analyze_parameter_statistics(
            aggregated_data['all_results']
        )
        aggregated_data['correlation_matrix'] = self._calculate_correlations(
            aggregated_data['all_results']
        )
        aggregated_data['clustering_results'] = self._perform_parameter_clustering(
            aggregated_data['best_configs_per_image']
        )

        # Salvar resultados
        self._save_aggregated_results(aggregated_data)

        print(f"\n✅ Agregação concluída: {len(aggregated_data['images'])} imagens")
        return aggregated_data

    def _analyze_parameter_statistics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Análise estatística dos parâmetros"""
        df = pd.DataFrame(all_results)

        stats_analysis = {
            'global_statistics': {},
            'parameter_distributions': {},
            'optimal_ranges': {},
            'significance_tests': {}
        }

        # Estatísticas globais
        numeric_cols = ['clip_limit', 'tile_grid_x', 'composite_score',
                       'vessel_clarity_gain', 'confidence_score']

        for col in numeric_cols:
            stats_analysis['global_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }

        # Distribuições por parâmetro
        for param in ['clip_limit', 'tile_grid_x']:
            param_groups = df.groupby(param)['composite_score']
            stats_analysis['parameter_distributions'][param] = {
                'means': param_groups.mean().to_dict(),
                'stds': param_groups.std().fillna(0).to_dict(),
                'counts': param_groups.count().to_dict()
            }

        # Testes de significância (ANOVA)
        try:
            # Teste se clip_limit afeta significativamente o composite_score
            clip_groups = [group['composite_score'].values
                          for name, group in df.groupby('clip_limit')]
            f_stat_clip, p_val_clip = stats.f_oneway(*clip_groups)

            tile_groups = [group['composite_score'].values
                          for name, group in df.groupby('tile_grid_x')]
            f_stat_tile, p_val_tile = stats.f_oneway(*tile_groups)

            stats_analysis['significance_tests'] = {
                'clip_limit_effect': {
                    'f_statistic': float(f_stat_clip),
                    'p_value': float(p_val_clip),
                    'significant': p_val_clip < self.significance_threshold
                },
                'tile_grid_effect': {
                    'f_statistic': float(f_stat_tile),
                    'p_value': float(p_val_tile),
                    'significant': p_val_tile < self.significance_threshold
                }
            }
        except Exception as e:
            print(f"⚠️ Erro nos testes estatísticos: {e}")
            stats_analysis['significance_tests'] = {}

        # Faixas ótimas (top 20% dos scores)
        top_20_percent = df.nlargest(int(len(df) * 0.2), 'composite_score')
        stats_analysis['optimal_ranges'] = {
            'clip_limit': {
                'min': float(top_20_percent['clip_limit'].min()),
                'max': float(top_20_percent['clip_limit'].max()),
                'mean': float(top_20_percent['clip_limit'].mean()),
                'most_frequent': float(top_20_percent['clip_limit'].mode().iloc[0])
            },
            'tile_grid': {
                'min': int(top_20_percent['tile_grid_x'].min()),
                'max': int(top_20_percent['tile_grid_x'].max()),
                'mean': float(top_20_percent['tile_grid_x'].mean()),
                'most_frequent': int(top_20_percent['tile_grid_x'].mode().iloc[0])
            }
        }

        return stats_analysis

    def _calculate_correlations(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Calcula matriz de correlações"""
        df = pd.DataFrame(all_results)

        # Variáveis numéricas para correlação
        numeric_vars = ['clip_limit', 'tile_grid_x', 'composite_score',
                       'clinical_relevance', 'vessel_clarity_gain', 'confidence_score']

        correlation_matrix = df[numeric_vars].corr()

        # Identificar correlações significativas
        significant_correlations = []
        for i in range(len(numeric_vars)):
            for j in range(i+1, len(numeric_vars)):
                var1, var2 = numeric_vars[i], numeric_vars[j]
                corr_value = correlation_matrix.loc[var1, var2]

                if abs(corr_value) > 0.3:  # Correlação moderada ou forte
                    significant_correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': float(corr_value),
                        'strength': self._classify_correlation_strength(abs(corr_value))
                    })

        return {
            'matrix': correlation_matrix.to_dict(),
            'significant_correlations': significant_correlations
        }

    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classifica força da correlação"""
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'negligible'

    def _perform_parameter_clustering(self, best_configs: List[Dict]) -> Dict[str, Any]:
        """Clustering dos parâmetros ótimos"""
        if len(best_configs) < 3:
            return {'error': 'Insufficient data for clustering'}

        # Preparar dados para clustering
        features = []
        image_ids = []

        for config in best_configs:
            features.append([
                config['best_params']['clip_limit'],
                config['best_params']['tile_grid'][0],  # tile_grid_x
                config['best_score']
            ])
            image_ids.append(config['image_id'])

        features_array = np.array(features)

        # Normalizar features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)

        # K-means clustering
        optimal_k = min(3, len(best_configs) // 2)  # Número sensato de clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_normalized)

        # Analisar clusters
        clusters_analysis = {}
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_features = features_array[cluster_mask]
            cluster_images = [image_ids[j] for j in range(len(image_ids)) if cluster_mask[j]]

            clusters_analysis[f'cluster_{i}'] = {
                'images': cluster_images,
                'size': int(np.sum(cluster_mask)),
                'centroid': {
                    'clip_limit': float(np.mean(cluster_features[:, 0])),
                    'tile_grid': float(np.mean(cluster_features[:, 1])),
                    'score': float(np.mean(cluster_features[:, 2]))
                },
                'characteristics': self._characterize_cluster(cluster_features)
            }

        return {
            'n_clusters': optimal_k,
            'clusters': clusters_analysis,
            'silhouette_score': self._calculate_silhouette_score(features_normalized, cluster_labels)
        }

    def _characterize_cluster(self, cluster_features: np.ndarray) -> str:
        """Caracteriza um cluster baseado nos valores médios"""
        mean_clip = np.mean(cluster_features[:, 0])
        mean_tile = np.mean(cluster_features[:, 1])
        mean_score = np.mean(cluster_features[:, 2])

        if mean_clip <= 1.5 and mean_tile >= 12:
            return "Conservative (baixo clip, grid grande)"
        elif mean_clip >= 3.0 and mean_tile <= 8:
            return "Aggressive (alto clip, grid pequeno)"
        else:
            return "Moderate (parâmetros intermediários)"

    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calcula silhouette score para validar clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(features, labels))
        except:
            return -1.0  # Indica erro no cálculo

    def _save_aggregated_results(self, aggregated_data: Dict[str, Any]) -> None:
        """Salva resultados agregados"""
        # Salvar dados JSON
        json_path = self.results_dir / "aggregated_analysis.json"



        # Usar a função já existente no utils
        serializable_data = numpy_to_python(aggregated_data)

        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"📊 Dados agregados salvos: {json_path}")

        # Salvar relatório em texto
        self._generate_text_report(aggregated_data)

        # Gerar visualizações
        self._create_aggregate_visualizations(aggregated_data)

    def _generate_text_report(self, data: Dict[str, Any]) -> None:
        """Gera relatório científico em texto"""
        report_path = self.results_dir / "statistical_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO ESTATÍSTICO DE OTIMIZAÇÃO PARAMÉTRICA\n")
            f.write("=" * 60 + "\n\n")

            # Resumo executivo
            n_images = len(data['images'])
            n_configs = len(data['all_results'])

            f.write(f"RESUMO EXECUTIVO\n")
            f.write(f"Imagens analisadas: {n_images}\n")
            f.write(f"Configurações testadas: {n_configs}\n")
            f.write(f"Configurações por imagem: {n_configs // n_images}\n\n")

            # Estatísticas dos parâmetros ótimos
            stats = data['parameter_statistics']
            optimal = stats['optimal_ranges']

            f.write("PARÂMETROS ÓTIMOS (TOP 20%)\n")
            f.write("-" * 30 + "\n")
            f.write(f"Clip Limit: {optimal['clip_limit']['min']:.1f} - {optimal['clip_limit']['max']:.1f} ")
            f.write(f"(média: {optimal['clip_limit']['mean']:.2f}, ")
            f.write(f"mais frequente: {optimal['clip_limit']['most_frequent']:.1f})\n")

            f.write(f"Tile Grid: {optimal['tile_grid']['min']} - {optimal['tile_grid']['max']} ")
            f.write(f"(média: {optimal['tile_grid']['mean']:.1f}, ")
            f.write(f"mais frequente: {optimal['tile_grid']['most_frequent']})\n\n")

            # Testes de significância
            if 'significance_tests' in stats and stats['significance_tests']:
                f.write("TESTES DE SIGNIFICÂNCIA ESTATÍSTICA\n")
                f.write("-" * 40 + "\n")

                clip_test = stats['significance_tests']['clip_limit_effect']
                f.write(f"Efeito do Clip Limit: F={clip_test['f_statistic']:.3f}, ")
                f.write(f"p={clip_test['p_value']:.6f} ")
                f.write(f"({'SIGNIFICATIVO' if clip_test['significant'] else 'NÃO SIGNIFICATIVO'})\n")

                tile_test = stats['significance_tests']['tile_grid_effect']
                f.write(f"Efeito do Tile Grid: F={tile_test['f_statistic']:.3f}, ")
                f.write(f"p={tile_test['p_value']:.6f} ")
                f.write(f"({'SIGNIFICATIVO' if tile_test['significant'] else 'NÃO SIGNIFICATIVO'})\n\n")

            # Correlações significativas
            if data['correlation_matrix']['significant_correlations']:
                f.write("CORRELAÇÕES SIGNIFICATIVAS\n")
                f.write("-" * 30 + "\n")
                for corr in data['correlation_matrix']['significant_correlations']:
                    f.write(f"{corr['variable_1']} ↔ {corr['variable_2']}: ")
                    f.write(f"r={corr['correlation']:.3f} ({corr['strength']})\n")
                f.write("\n")

            # Clustering
            if 'clusters' in data['clustering_results']:
                f.write("ANÁLISE DE CLUSTERING\n")
                f.write("-" * 25 + "\n")
                clusters = data['clustering_results']['clusters']
                for cluster_id, cluster_info in clusters.items():
                    f.write(f"{cluster_id.upper()}: {cluster_info['characteristics']}\n")
                    f.write(f"  Imagens: {cluster_info['size']} ({cluster_info['images']})\n")
                    f.write(f"  Centroide: clip={cluster_info['centroid']['clip_limit']:.2f}, ")
                    f.write(f"tile={cluster_info['centroid']['tile_grid']:.1f}\n\n")

        print(f"📄 Relatório estatístico: {report_path}")

    def _create_aggregate_visualizations(self, data: Dict[str, Any]) -> None:
        """Cria visualizações agregadas"""
        # Plot 1: Distribuição dos parâmetros ótimos
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise Agregada de Parâmetros CLAHE', fontsize=16, fontweight='bold')

        # Preparar dados
        df = pd.DataFrame(data['all_results'])
        best_configs = data['best_configs_per_image']

        # Distribuição de clip_limit nos top performers
        top_clips = [config['best_params']['clip_limit'] for config in best_configs]
        axes[0, 0].hist(top_clips, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribuição de Clip Limit Ótimo')
        axes[0, 0].set_xlabel('Clip Limit')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].grid(True, alpha=0.3)

        # Distribuição de tile_grid nos top performers
        top_tiles = [config['best_params']['tile_grid'][0] for config in best_configs]
        axes[0, 1].hist(top_tiles, bins=6, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribuição de Tile Grid Ótimo')
        axes[0, 1].set_xlabel('Tile Grid Size')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].grid(True, alpha=0.3)

        # Heatmap de scores médios por parâmetros
        pivot_table = df.pivot_table(values='composite_score',
                                   index='clip_limit',
                                   columns='tile_grid_x',
                                   aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[1, 0], cbar_kws={'label': 'Composite Score'})
        axes[1, 0].set_title('Heatmap: Score Médio por Parâmetros')

        # Scatter plot: Clip vs Score com clustering
        for i, config in enumerate(best_configs):
            axes[1, 1].scatter(config['best_params']['clip_limit'],
                             config['best_score'],
                             alpha=0.7, s=80)

        axes[1, 1].set_title('Relação Clip Limit vs Score Ótimo')
        axes[1, 1].set_xlabel('Clip Limit')
        axes[1, 1].set_ylabel('Composite Score')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.results_dir / "aggregate_parameter_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Visualizações agregadas: {plot_path}")

def main():
    """Função principal para teste"""
    # Simular execução
    analysis_dir = Path("results/parameter_analysis")

    if not analysis_dir.exists():
        print("⚠️ Diretório de análises não encontrado")
        print("Execute primeiro o processamento das imagens individuais")
        return

    aggregator = ParameterAggregator(analysis_dir)
    results = aggregator.aggregate_parameter_analyses()

    print("\n" + "="*60)
    print("AGREGAÇÃO MULTI-IMAGEM CONCLUÍDA")
    print("="*60)
    print(f"📊 Imagens analisadas: {len(results['images'])}")
    print(f"📈 Resultados salvos em: {aggregator.results_dir}")

if __name__ == "__main__":
    main()
