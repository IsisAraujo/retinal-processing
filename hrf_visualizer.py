"""
HRF Results Visualization Module

Generates publication-quality visualizations from experimental results
for inclusion in LaTeX documents and academic papers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats

# Usando as mesmas configurações de estilo do hrf_analysis.py
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class HRFVisualizer:
    """
    Generates high-quality visualizations for HRF illumination correction results
    following academic publication standards.
    """

    def __init__(self, results_path: str, output_dir: str = "figures"):
        """
        Initialize the visualizer with data path and output directory.

        Parameters:
        -----------
        results_path : str
            Path to the results_dataframe.csv file
        output_dir : str
            Directory where visualizations will be saved
        """
        self.df = pd.read_csv(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Define consistent color palette for methods, similar to default seaborn colors
        self.method_colors = {
            'CLAHE': '#1f77b4',  # Blue
            'SSR': '#ff7f0e',    # Orange
            'MSR': '#2ca02c',    # Green
            'MSRCR': '#d62728'   # Red
        }

        # Define metrics to analyze (excluding processing time)
        self.metrics = [
            'contrast_ratio',
            'vessel_clarity_index',
            'illumination_uniformity',
            'edge_preservation_index',
            'microaneurysm_visibility'
        ]

        # Human-readable labels for metrics
        self.metric_labels = {
            'contrast_ratio': 'Contrast Ratio',
            'vessel_clarity_index': 'Vessel Clarity Index',
            'illumination_uniformity': 'Illumination Uniformity',
            'edge_preservation_index': 'Edge Preservation Index',
            'microaneurysm_visibility': 'Microaneurysm Visibility',
            'processing_time_ms': 'Processing Time (ms)'
        }

    def _add_statistical_annotations(self, ax, data: pd.DataFrame, metric: str) -> None:
        """
        Add statistical significance annotations to box plots.
        Uses non-parametric Kruskal-Wallis test followed by post-hoc
        Mann-Whitney U tests with Bonferroni correction.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes object to add annotations to
        data : pd.DataFrame
            Data to analyze
        metric : str
            Metric to test for statistical significance
        """
        # Perform Kruskal-Wallis test
        groups = [data[data['method'] == method][metric].values
                 for method in data['method'].unique()]

        # Skip if any group has less than 2 observations
        if any(len(group) < 2 for group in groups):
            return

        h_stat, p_value = stats.kruskal(*groups)

        # Only proceed with post-hoc tests if overall significance
        if p_value < 0.05:
            methods = data['method'].unique()
            y_max = data[metric].max() * 1.05
            y_step = data[metric].std() * 0.2

            # Add overall significance with background box for readability
            ax.text(0.5, 0.92, f"Kruskal-Wallis: p = {p_value:.4f}" if p_value >= 0.0001
                    else "Kruskal-Wallis: p < 0.0001",
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=8, style='italic',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

            # Perform post-hoc Mann-Whitney U tests with Bonferroni correction
            num_comparisons = len(methods) * (len(methods) - 1) // 2
            alpha_corrected = 0.05 / num_comparisons

            # Limit to a reasonable number of annotations to avoid clutter
            significant_pairs = []

            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    if i < j:
                        group1 = data[data['method'] == m1][metric].values
                        group2 = data[data['method'] == m2][metric].values

                        # Skip if either group has less than 2 observations
                        if len(group1) < 2 or len(group2) < 2:
                            continue

                        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                        if p_val < alpha_corrected:
                            significant_pairs.append((m1, m2, p_val))

            # Add up to 3 most significant comparisons to avoid cluttering
            significant_pairs.sort(key=lambda x: x[2])
            for idx, (m1, m2, p_val) in enumerate(significant_pairs[:3]):
                i1, i2 = list(methods).index(m1), list(methods).index(m2)
                y_pos = y_max + (idx + 1) * y_step

                # Draw significance bar
                ax.plot([i1, i2], [y_pos, y_pos], 'k-', linewidth=1)
                ax.plot([i1, i1], [y_max * 0.99, y_pos], 'k-', linewidth=1)
                ax.plot([i2, i2], [y_max * 0.99, y_pos], 'k-', linewidth=1)

                # Add p-value
                p_text = f"p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
                ax.text((i1 + i2) / 2, y_pos + y_step * 0.2, p_text,
                        ha='center', va='bottom', fontsize=7)

    def create_boxplots(self, show_points: bool = True) -> str:
        """
        Create box plots for all quality metrics, comparing different methods.
        Similar to create_metrics_boxplots in hrf_analysis.py

        Parameters:
        -----------
        show_points : bool
            Whether to show individual data points on box plots

        Returns:
        --------
        str
            Path to saved figure
        """
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # For each metric, create a box plot
        for i, metric in enumerate(self.metrics):
            if i < len(axes):
                ax = axes[i]

                # Create box plot with seaborn for better appearance
                sns.boxplot(
                    x='method',
                    y=metric,
                    hue='method',
                    data=self.df,
                    palette=self.method_colors,
                    legend=False,
                    ax=ax
                )

                # Add individual points
                if show_points:
                    sns.stripplot(
                        x='method',
                        y=metric,
                        data=self.df,
                        color='black',
                        size=3,
                        alpha=0.4,
                        jitter=True,
                        ax=ax
                    )

                # Set labels and title - similar to hrf_analysis.py style
                ax.set_title(self.metric_labels[metric])
                ax.set_xlabel('Method')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)

                # Format y-axis
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

                # Add statistical annotations
                self._add_statistical_annotations(ax, self.df, metric)

        # Create a box plot for processing time in a separate subplot
        if len(self.metrics) < len(axes):
            ax = axes[len(self.metrics)]

            # Box plot for processing time (log scale)
            sns.boxplot(
                x='method',
                y='processing_time_ms',
                data=self.df,
                palette=self.method_colors,
                ax=ax
            )

            # Add individual points
            if show_points:
                sns.stripplot(
                    x='method',
                    y='processing_time_ms',
                    data=self.df,
                    color='black',
                    size=3,
                    alpha=0.4,
                    jitter=True,
                    ax=ax
                )

            # Set logarithmic scale for processing time
            ax.set_yscale('log')

            # Set labels and title
            ax.set_title('Processing Time')
            ax.set_xlabel('Method')
            ax.set_ylabel('Processing Time (ms, log scale)')
            ax.grid(True, alpha=0.3)

            # Add statistical annotations
            self._add_statistical_annotations(ax, self.df, 'processing_time_ms')

        # Remove empty subplot
        if len(self.metrics) + 1 < len(axes):
            fig.delaxes(axes[-1])

        # Adjust layout - similar to hrf_analysis.py
        plt.suptitle('Distribution of Evaluation Metrics by Method', fontsize=12)
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'quality_metrics_boxplots.pdf'
        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def create_processing_time_barchart(self) -> str:
        """
        Create a bar chart for processing time comparison.

        Returns:
        --------
        str
            Path to saved figure
        """
        # Calculate mean and std of processing time for each method
        time_stats = self.df.groupby('method')['processing_time_ms'].agg(['mean', 'std']).reset_index()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create bar chart (sem barras de erro)
        bars = ax.bar(
            time_stats['method'],
            time_stats['mean'],
            color=[self.method_colors[m] for m in time_stats['method']],
            alpha=0.8,
            width=0.7
        )

        # Add value labels 0.5 acima da barra
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.6,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # Set labels and title
        ax.set_xlabel('Method')
        ax.set_ylabel('Processing Time (ms)')
        ax.set_title('Average Processing Time by Method')

        # Add grid for readability
        ax.grid(axis='y', alpha=0.3)

        # Improve y-axis formatting based on data scale
        if time_stats['mean'].max() > 1000:
            ax.set_yscale('log')
            ax.set_ylabel('Processing Time (ms, log scale)')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,.0f}'.format(y)))
            plt.subplots_adjust(left=0.15)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'processing_time_analysis.pdf'
        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def create_radar_chart(self) -> str:
        """
        Create a radar chart comparing methods across all metrics with improved symmetry
        and matching font style with processing_time_analysis.pdf.
        """
        radar_data = self.df.groupby('method')[self.metrics].mean().reset_index()
        num_metrics = len(self.metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each method
        for method in radar_data['method']:
            values = radar_data[radar_data['method'] == method][self.metrics].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=method, color=self.method_colors[method])
            ax.fill(angles, values, alpha=0.1, color=self.method_colors[method])

        # Set metric labels using built-in positioning
        ax.set_xticks(angles[:-1])

        # Ajuste da fonte para corresponder ao processing_time_analysis.pdf
        ax.set_xticklabels([self.metric_labels[m] for m in self.metrics],
                           fontsize=10,
                           fontfamily='serif')

        # Improve radial axis with matching font style
        max_value = max([radar_data[metric].max() for metric in self.metrics]) * 1.2
        ax.set_ylim(0, max_value)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                           color='gray',
                           fontsize=8,
                           fontfamily='serif')

        ax.set_rlabel_position(0)  # Radial labels position

        # Add grid and legend with matching font style
        ax.grid(True, alpha=0.3)
        legend = ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.15))

        # Ajuste da fonte da legenda
        plt.setp(legend.get_texts(), fontsize=9, fontfamily='serif')

        # Título com a mesma fonte
        plt.title('Performance Comparison Across All Metrics',
                  y=1.05,
                  fontsize=11,
                  fontfamily='serif')

        plt.tight_layout()

        output_path = self.output_dir / 'radar_chart_comparison.pdf'
        plt.savefig(output_path)
        plt.close()
        return str(output_path)

    def create_correlation_heatmap(self) -> str:
        """
        Create correlation heatmap of metrics, similar to the one in hrf_analysis.py

        Returns:
        --------
        str
            Path to saved figure
        """
        # Prepare data
        metrics_df = self.df[self.metrics]
        correlation_matrix = metrics_df.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap - exatamente como no hrf_analysis.py
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'correlation_heatmap.pdf'
        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def create_summary_grid(self) -> str:
        """
        Create a comprehensive summary grid with all key visualizations.

        Returns:
        --------
        str
            Path to saved figure
        """
        # Create figure with grid layout
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)

        # Top left: Processing time
        ax_time = fig.add_subplot(gs[0, 0])

        # Top right: Radar chart
        ax_radar = fig.add_subplot(gs[0, 1], polar=True)

        # Bottom left: Correlation heatmap
        ax_corr = fig.add_subplot(gs[1, 0])

        # Bottom right: Quality metrics
        ax_quality = fig.add_subplot(gs[1, 1])

        # 1. Processing time (logarithmic scale)
        time_stats = self.df.groupby('method')['processing_time_ms'].agg(['mean', 'std']).reset_index()
        bars = ax_time.bar(
            time_stats['method'],
            time_stats['mean'],
            yerr=time_stats['std'],
            color=[self.method_colors[m] for m in time_stats['method']],
            capsize=5,
            alpha=0.8,
            ecolor='black'
        )

        # Log scale for processing time
        ax_time.set_yscale('log')
        ax_time.set_ylabel('Processing Time (ms, log scale)')
        ax_time.set_xlabel('Method')
        ax_time.set_title('Processing Time Comparison')
        ax_time.grid(axis='y', alpha=0.3)

        # 2. Radar chart
        radar_data = self.df.groupby('method')[self.metrics].mean().reset_index()
        angles = np.linspace(0, 2 * np.pi, len(self.metrics), endpoint=False).tolist()
        angles += angles[:1]

        for method in radar_data['method']:
            values = radar_data[radar_data['method'] == method][self.metrics].values.flatten().tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=method, color=self.method_colors[method])
            ax_radar.fill(angles, values, alpha=0.1, color=self.method_colors[method])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels([m.split('_')[0].capitalize() for m in self.metrics], fontsize=8)

        # Adjust labels position
        for label, angle in zip(ax_radar.get_xticklabels(), angles[:-1]):
            if angle < np.pi/2 or angle > 3*np.pi/2:
                label.set_horizontalalignment('left')
                label.set_position((1.3*np.cos(angle), 1.3*np.sin(angle)))
            else:
                label.set_horizontalalignment('right')
                label.set_position((1.3*np.cos(angle), 1.3*np.sin(angle)))

        ax_radar.set_title('Multi-Metric Performance')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax_radar.grid(True, alpha=0.3)

        # 3. Correlation heatmap
        correlation_matrix = self.df[self.metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax_corr)
        ax_corr.set_title('Correlation Matrix')

        # 4. Quality composite score
        # Create a composite quality score
        quality_metrics = ['contrast_ratio', 'vessel_clarity_index', 'microaneurysm_visibility']
        self.df['quality_score'] = self.df[quality_metrics].mean(axis=1)

        # Create boxplot of quality score
        sns.boxplot(
            x='method',
            y='quality_score',
            data=self.df,
            palette=self.method_colors,
            hue='method',  # Adicionado o parâmetro hue
            legend=False,  # Desativa a legenda para evitar duplicação
            ax=ax_quality
        )

        # Add points
        sns.stripplot(
            x='method',
            y='quality_score',
            data=self.df,
            color='black',
            size=3,
            alpha=0.4,
            jitter=True,
            ax=ax_quality
        )

        ax_quality.set_title('Composite Quality Score')
        ax_quality.set_xlabel('Method')
        ax_quality.set_ylabel('Score')
        ax_quality.grid(True, alpha=0.3)

        # Add statistical annotations
        self._add_statistical_annotations(ax_quality, self.df, 'quality_score')

        # Main title
        fig.suptitle('Comprehensive Analysis of HRF Illumination Correction Methods',
                    fontsize=12)

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)

        # Save figure
        output_path = self.output_dir / 'comprehensive_analysis_grid.pdf'
        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate selected visualizations and return their file paths.
        """
        # print("Generating box plots for quality metrics...")
        # boxplots_path = self.create_boxplots()  # Removido

        print("Generating processing time bar chart...")
        time_chart_path = self.create_processing_time_barchart()

        print("Generating correlation heatmap...")
        heatmap_path = self.create_correlation_heatmap()

        print("Generating radar chart comparison...")
        radar_chart_path = self.create_radar_chart()

        return {
            # "quality_metrics_boxplots": boxplots_path,  # Removido
            "processing_time_chart": time_chart_path,
            "correlation_heatmap": heatmap_path,
            "radar_chart": radar_chart_path
        }


def main():
    """Main function to run the visualizer"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication-quality visualizations for HRF results")
    parser.add_argument("--results", type=str, default="results/data/results_dataframe.csv",
                        help="Path to results_dataframe.csv file")
    parser.add_argument("--output", type=str, default="results/figures",
                        help="Directory to save visualizations")

    args = parser.parse_args()

    print(f"Loading results from: {args.results}")
    print(f"Saving visualizations to: {args.output}")

    visualizer = HRFVisualizer(args.results, args.output)
    paths = visualizer.generate_all_visualizations()

    print("\nVisualization complete. Files saved:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
