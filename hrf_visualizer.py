#!/usr/bin/env python3
"""
Academic Visualization Generator for Fundoscopic Image Enhancement Methods
Generates publication-ready figures for scientific paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Academic publication settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

class AcademicVisualizationGenerator:
    """
    Generates publication-ready visualizations for medical imaging research
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("results/figures")  # Ajuste: saída para 'figures conforme hrf_experiment.py
        self.output_dir.mkdir(exist_ok=True)

        # Academic color palette (colorblind-friendly)
        self.colors = {
            'CLAHE': '#1f77b4',    # Blue - efficiency leader
            'SSR': '#ff7f0e',      # Orange - baseline Retinex
            'MSR': '#2ca02c',      # Green - uniformity leader
            'MSRCR': '#d62728'     # Red - clinical detection leader
        }

        self.load_data()

    def load_data(self):
        """Load experimental data from CSV and JSON files"""
        try:
            # Ajuste: caminhos para subpasta 'results/data' conforme sua estrutura
            self.df = pd.read_csv(self.data_dir / "results" / "data" / "enhanced_results_dataframe.csv")
            with open(self.data_dir / "results" / "data" / "enhanced_statistical_analysis.json", 'r') as f:
                self.stats = json.load(f)
            self.stats_summary = pd.read_csv(self.data_dir / "results" / "tables" / "enhanced_statistical_summary.csv")
            self.power_analysis = pd.read_csv(self.data_dir / "results" / "power_analysis" / "power_analysis_report.csv")

            print("✓ Data loaded successfully")
            print(f"  - {len(self.df)} observations across {len(self.df['method'].unique())} methods")
            print(f"  - {len(self.df['image_id'].unique())} images processed")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_figure_1_method_comparison(self):
        """
        Figure 1: Comprehensive method comparison across all metrics
        Key insight: CLAHE vs MSRCR trade-offs for different clinical needs
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Comparative Performance Analysis of Illumination Correction Methods',
                    fontsize=16, fontweight='bold')

        # Metrics for visualization
        metrics = [
            ('psnr', 'PSNR (dB)', 'Higher is Better'),
            ('ssim', 'SSIM Index', 'Higher is Better'),
            ('contrast_ratio', 'Contrast Ratio', 'Higher is Better'),
            ('vessel_clarity_index', 'Vessel Clarity Index', 'Higher is Better'),
            ('illumination_uniformity', 'Illumination Uniformity', 'Higher is Better'),
            ('microaneurysm_visibility', 'Microaneurysm Visibility', 'Higher is Better')
        ]

        for idx, (metric, title, subtitle) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            # Create boxplot with custom colors
            method_order = ['CLAHE', 'SSR', 'MSR', 'MSRCR']
            box_data = [self.df[self.df['method'] == method][metric].dropna()
                       for method in method_order]

            bp = ax.boxplot(box_data, labels=method_order, patch_artist=True,
                           boxprops=dict(alpha=0.7), medianprops=dict(color='black', linewidth=2))

            # Color boxes
            for patch, method in zip(bp['boxes'], method_order):
                patch.set_facecolor(self.colors[method])

            ax.set_title(f'{title}\n({subtitle})', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            ax.set_xlabel('Method', fontsize=14, fontweight='bold')
            ax.set_ylabel(title, fontsize=14, fontweight='bold')

            # Add mean values as text
            for i, method in enumerate(method_order):
                mean_val = self.df[self.df['method'] == method][metric].mean()
                ax.text(i+1, ax.get_ylim()[0], f'{mean_val:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        # plt.savefig(self.output_dir / 'figure_1_method_comparison.pdf', dpi=300, bbox_inches='tight')  # Removido
        plt.savefig(self.output_dir / 'figure_1_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        return "Figure 1: Comprehensive method comparison generated"

    def create_figure_2_efficiency_analysis(self):
        """
        Figure 2: Computational efficiency vs Quality trade-off
        Key insight: CLAHE exceptional efficiency, MSRCR quality at computational cost
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Computational Efficiency vs Clinical Quality Trade-off Analysis',
                     fontsize=16, fontweight='bold')

        # Calculate composite quality score
        quality_metrics = ['psnr', 'ssim', 'contrast_ratio', 'vessel_clarity_index',
                          'illumination_uniformity', 'microaneurysm_visibility']

        # Normalize metrics to 0-1 scale for fair comparison
        df_norm = self.df.copy()
        for metric in quality_metrics:
            min_val = df_norm[metric].min()
            max_val = df_norm[metric].max()
            df_norm[f'{metric}_norm'] = (df_norm[metric] - min_val) / (max_val - min_val)

        # Calculate composite score
        norm_cols = [f'{m}_norm' for m in quality_metrics if f'{m}_norm' in df_norm.columns]
        df_norm['quality_score'] = df_norm[norm_cols].mean(axis=1)

        # Scatter plot: Processing Time vs Quality
        for method in ['CLAHE', 'SSR', 'MSR', 'MSRCR']:
            method_data = df_norm[df_norm['method'] == method]
            ax.scatter(method_data['processing_time_ms'], method_data['quality_score'],
                       label=method, color=self.colors[method], s=100, alpha=0.7, edgecolors='black')

        ax.set_xlabel('Processing Time (ms)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Composite Quality Score (0-1)', fontweight='bold', fontsize=14)
        ax.set_title('Quality vs Computational Cost\n(Bottom-right is optimal)', fontweight='bold', fontsize=16)
        ax.set_xscale('log')
        ax.legend(fontsize=13, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        # Add efficiency annotations
        clahe_data = df_norm[df_norm['method'] == 'CLAHE']
        msrcr_data = df_norm[df_norm['method'] == 'MSRCR']

        clahe_time = clahe_data['processing_time_ms'].mean()
        msrcr_time = msrcr_data['processing_time_ms'].mean()
        efficiency_ratio = msrcr_time / clahe_time

        # Ajuste: seta e texto um pouco mais à direita e maior para caber em uma linha só
        ax.annotate(
            f'CLAHE: {efficiency_ratio:.0f}x faster than MSRCR',
            xy=(clahe_time, clahe_data['quality_score'].mean()),
            xytext=(clahe_time * 2, 0.72),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2, shrinkA=8, shrinkB=8),  # shrinkA/B deixam a seta menor
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.85)
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure_2_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        return "Figure 2: Efficiency vs Quality analysis generated"

    def generate_all_figures(self):
        """Generate all academic figures for the paper"""
        print("\n" + "="*60)
        print("GENERATING ACADEMIC FIGURES")
        print("="*60)

        results = []

        try:
            results.append(self.create_figure_1_method_comparison())
            print("✓ Figure 1 completed")

            results.append(self.create_figure_2_efficiency_analysis())
            print("✓ Figure 2 completed")

            # Remover ou comentar as linhas abaixo:
            # results.append(self.create_summary_table_figure())
            # print("✓ Summary table completed")

            print(f"\n✓ All figures saved to: {self.output_dir.absolute()}")

        except Exception as e:
            print(f"Error generating figures: {e}")
            raise

        return results

if __name__ == "__main__":
    # Ajuste: use "." para data_dir, pois os arquivos estão em subpastas do diretório atual
    generator = AcademicVisualizationGenerator(data_dir=".")
    generator.generate_all_figures()

