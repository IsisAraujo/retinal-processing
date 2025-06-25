#!/usr/bin/env python3
"""
Academic Visualization Generator for Fundoscopic Image Enhancement Methods
Generates publication-ready figures and tables for scientific paper
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
    Generates publication-ready visualizations and tables for medical imaging research.
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.output_figures_dir = Path("results/figures") # Saída para figuras
        self.output_tables_dir = Path("results/tables")   # Saída para tabelas
        self.output_figures_dir.mkdir(exist_ok=True)
        self.output_tables_dir.mkdir(exist_ok=True)

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
        plt.savefig(self.output_figures_dir / 'figure_1_method_comparison.png', dpi=300, bbox_inches='tight')
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
        ax.set_title('Quality vs Computational Cost\n(Lower Time and Higher Quality Score are Optimal)', fontweight='bold', fontsize=16)
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

        ax.annotate(
            f'CLAHE: {efficiency_ratio:.0f}x faster than MSRCR',
            xy=(clahe_time, clahe_data['quality_score'].mean()),
            xytext=(clahe_time * 2, 0.72),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2, shrinkA=8, shrinkB=8),
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.85)
        )

        plt.tight_layout()
        plt.savefig(self.output_figures_dir / 'figure_2_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        return "Figure 2: Efficiency vs Quality analysis generated"

    def create_figure_3_p_value_table(self):
        """
        Generates a publication-ready p-value table with robust error handling
        and symmetric design for academic papers.
        """
        metrics = sorted(self.stats['comparisons'].keys())
        data = []

        # Color coding scheme for significance levels
        def get_color(p):
            if np.isnan(p):
                return 'white'
            elif p <= 0.001:
                return '#006400'  # Dark green (highly significant)
            elif p <= 0.01:
                return '#32CD32'  # Medium green (very significant)
            elif p <= 0.05:
                return '#FFD700'  # Gold (significant)
            else:
                return '#D3D3D3'  # Light gray (not significant)

        # Prepare table data with compact metric names
        for metric in metrics:
            result = self.stats['comparisons'][metric]
            bonf = result.get('bonferroni_corrected_p', np.nan)
            fdr = result.get('fdr_corrected_p', np.nan)

            # Scientific abbreviations for retinal metrics
            metric_name = metric.replace("_", " ").title()
            metric_name = (metric_name.replace("Peak Signal To Noise Ratio", "PSNR")
                        .replace("Structural Similarity Index", "SSIM")
                        .replace("Contrast Ratio", "Contrast")
                        .replace("Vessel Clarity Index", "Vessel Clarity")
                        .replace("Illumination Uniformity", "Illum. Unif.")
                        .replace("Microaneurysm Visibility", "MA Visibility"))

            # Use English terms for international submission
            bonf_sig = "YES" if not np.isnan(bonf) and bonf < 0.05 else "NO"
            fdr_sig = "YES" if not np.isnan(fdr) and fdr < 0.05 else "NO"

            data.append([
                metric_name,
                f"{bonf:.3e}" if not np.isnan(bonf) else "N/A",
                bonf_sig,
                f"{fdr:.3e}" if not np.isnan(fdr) else "N/A",
                fdr_sig,
                get_color(bonf),
                get_color(fdr)
            ])

        # Create symmetric dataframe
        df_vis = pd.DataFrame(data, columns=[
            "Metric", "Bonf. p", "Sig_Bonf", "FDR p", "Sig_FDR", "BonfColor", "FDRColor"
        ])

        # Tamanho aumentado da figura
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        # === CRITICAL FIX: Robust column width handling ===
        num_columns = 5  # Explicitly define expected columns
        # Ajustado para dar mais espaço para as colunas de significância
        symmetric_widths = [0.30, 0.18, 0.12, 0.18, 0.12]

        # Validate dimensions before table creation
        if len(symmetric_widths) != num_columns:
            symmetric_widths = [0.25] * num_columns  # Safe fallback

        # Adicionando verificação para dados
        print("Verifying table data:")
        for row in df_vis[["Metric", "Bonf. p", "Sig_Bonf", "FDR p", "Sig_FDR"]].values:
            print(row)  # Diagnostic information for console

        # Create table with dimension safeguards
        try:
            table = ax.table(
                cellText=df_vis[["Metric", "Bonf. p", "Sig_Bonf", "FDR p", "Sig_FDR"]].values,
                colLabels=[
                    "Ophthalmic Metric",
                    "Bonferroni p-value",
                    "Significant",  # Changed to English
                    "FDR p-value",
                    "Significant"   # Changed to English
                ],
                loc='center',
                cellLoc='center',
                colLoc='center',
                colWidths=symmetric_widths
            )

            # Increase header row height to accommodate larger text
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_height(0.10)  # Increased from 0.08 to 0.10
        except ValueError as e:
            print(f"Error creating table: {e}")
            # Emergency recovery for dimension mismatch
            table = ax.table(
                cellText=df_vis.iloc[:, :num_columns].values,
                colLabels=df_vis.columns[:num_columns],
                loc='center'
            )

        # ===== ACADEMIC STYLING =====
        # 1. Header styling with LARGER FONT SIZE
        for i in range(5):
            try:
                cell = table.get_celld()[(0, i)]
                # Increased font size for headers (14 pt)
                cell.set_text_props(weight='bold', color='white', fontsize=14)
                cell.set_facecolor('#2F5496')  # Academic blue
            except KeyError:
                continue  # Skip if cell doesn't exist

        # 2. Significance highlighting
        for i in range(len(df_vis)):
            try:
                # Bonferroni column
                table.get_celld()[(i+1, 1)].set_facecolor(df_vis.iloc[i]['BonfColor'])
                # FDR column
                table.get_celld()[(i+1, 3)].set_facecolor(df_vis.iloc[i]['FDRColor'])

                # Color significance columns based on value
                bonf_sig = df_vis.iloc[i]['Sig_Bonf']
                fdr_sig = df_vis.iloc[i]['Sig_FDR']

                if bonf_sig == "YES":
                    table.get_celld()[(i+1, 2)].set_facecolor('#90EE90')  # light green
                else:
                    table.get_celld()[(i+1, 2)].set_facecolor('#FFCCCB')  # light red

                if fdr_sig == "YES":
                    table.get_celld()[(i+1, 4)].set_facecolor('#90EE90')  # light green
                else:
                    table.get_celld()[(i+1, 4)].set_facecolor('#FFCCCB')  # light red

                # Zebra striping for metrics
                if i % 2 == 0:
                    table.get_celld()[(i+1, 0)].set_facecolor('#F8F8F8')
            except (KeyError, IndexError) as e:
                print(f"Error styling cell [{i}]: {e}")
                continue

        # 3. Typography refinement - INCREASED FONT SIZE
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # Increased from 11 to 12 for cell text
        table.scale(1, 1.7)     # Improved spacing for taller cells

        # ===== STATISTICAL CONTEXT =====
        # 1. Significance legend
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, color='#006400', label='p ≤ 0.001'),
            plt.Rectangle((0,0), 1, 1, color='#32CD32', label='p ≤ 0.01'),
            plt.Rectangle((0,0), 1, 1, color='#FFD700', label='p ≤ 0.05'),
            plt.Rectangle((0,0), 1, 1, color='#D3D3D3', label='p > 0.05')
        ]

        ax.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=4,
            frameon=False,
            title="Significance Level:",
            title_fontproperties={'weight': 'bold', 'size': 12},  # Increased legend title size
            fontsize=12  # Increased legend text size
        )

        # 2. Statistical recommendation (in English)
        fig.text(
            0.5, -0.05,
            "Interpretation recommendation: FDR-corrected values should be prioritized for clinical conclusions",
            ha='center',
            fontsize=11,  # Increased from 10 to 11
            style='italic'
        )

        # 3. Table title
        plt.suptitle(
            "Statistical Significance of Illumination Correction Methods",
            fontsize=18,  # Increased from 16 to 18
            fontweight='bold',
            y=0.98
        )

        # Save publication-ready figure
        out_path = self.output_figures_dir / "figure3_statistical_significance_table.png"
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()

        return f"Figure 3: Statistical significance table saved to {out_path}"

    def generate_all_figures(self):
        """Generate all academic figures and tables for the paper"""
        print("\n" + "="*60)
        print("GENERATING ACADEMIC FIGURES AND TABLES")
        print("="*60)

        results = []

        try:
            results.append(self.create_figure_1_method_comparison())
            print("✓ Figure 1 completed")

            results.append(self.create_figure_2_efficiency_analysis())
            print("✓ Figure 2 completed")

            # CHAMA AGORA O NOVO MÉTODO QUE GERA A TABELA AO INVÉS DO PLOT
            results.append(self.create_figure_3_p_value_table())
            print("✓ Figure 3 (Table) completed")

            print(f"\n✓ All figures and tables saved to: {self.output_figures_dir.absolute()} and {self.output_tables_dir.absolute()}")

        except Exception as e:
            print(f"Error generating figures/tables: {e}")
            raise

        return results

if __name__ == "__main__":
    # Ajuste: use "." para data_dir, pois os arquivos estão em subpastas do diretório atual
    generator = AcademicVisualizationGenerator(data_dir=".")
    generator.generate_all_figures()
