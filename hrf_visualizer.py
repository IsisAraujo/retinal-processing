"""
HRF Results Visualization Module

Generates publication-quality visualizations from experimental results
for inclusion in LaTeX documents and academic papers.

Author: [Isis Araaujo]
Affiliation: [Universidade Federal de Sergipe, Brazil]
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

# Configurações para visualizações adequadas para publicação
plt.rcParams.update({
    'font.family': 'sans',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 11,
    'text.usetex': False,  # Mudar para True se tiver LaTeX instalado
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
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

        # Define consistent color palette for methods
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

            # Add overall significance
            ax.text(0.5, 0.99, f"Kruskal-Wallis: p = {p_value:.4f}" if p_value >= 0.0001
                    else "Kruskal-Wallis: p < 0.0001",
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=8, style='italic')

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
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        axes = axes.flatten()

        # For each metric, create a box plot
        for i, metric in enumerate(self.metrics):
            if i < len(axes):
                ax = axes[i]

                # Create box plot
                sns.boxplot(
                    x='method',
                    y=metric,
                    data=self.df,
                    palette=self.method_colors,
                    width=0.6,
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

                # Set labels and title
                ax.set_xlabel('')
                ax.set_ylabel(self.metric_labels[metric])
                ax.set_title(self.metric_labels[metric], fontweight='bold')

                # Format y-axis
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

                # Add statistical annotations
                self._add_statistical_annotations(ax, self.df, metric)

        # Create a box plot for processing time in a separate subplot
        if len(self.metrics) < len(axes):
            ax = axes[len(self.metrics)]

            # Convert to log scale for better visualization
            self.df['processing_time_log'] = np.log10(self.df['processing_time_ms'])

            # Create box plot for processing time (log scale)
            sns.boxplot(
                x='method',
                y='processing_time_log',
                data=self.df,
                palette=self.method_colors,
                width=0.6,
                ax=ax
            )

            # Add individual points
            if show_points:
                sns.stripplot(
                    x='method',
                    y='processing_time_log',
                    data=self.df,
                    color='black',
                    size=3,
                    alpha=0.4,
                    jitter=True,
                    ax=ax
                )

            # Create custom y-tick labels (convert from log back to original scale)
            log_ticks = ax.get_yticks()
            ax.set_yticklabels([f"{10**y:.0f}" for y in log_ticks])

            # Set labels and title
            ax.set_xlabel('')
            ax.set_ylabel('Processing Time (ms, log scale)')
            ax.set_title('Processing Time', fontweight='bold')

            # Add statistical annotations
            self._add_statistical_annotations(ax, self.df, 'processing_time_log')

        # Remove any unused subplots
        for i in range(len(self.metrics) + 1, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.suptitle('Comparison of Illumination Correction Methods', fontsize=14, fontweight='bold')

        # Save figure
        output_path = self.output_dir / 'quality_metrics_boxplots.pdf'
        plt.savefig(output_path, bbox_inches='tight')
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

        # Create bar chart
        bars = ax.bar(
            time_stats['method'],
            time_stats['mean'],
            yerr=time_stats['std'],
            color=[self.method_colors[m] for m in time_stats['method']],
            capsize=5,
            alpha=0.8,
            ecolor='black'
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + time_stats['std'].max() * 0.2,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # Set labels and title
        ax.set_xlabel('Illumination Correction Method', fontsize=11)
        ax.set_ylabel('Processing Time (ms)', fontsize=11)
        ax.set_title('Average Processing Time by Method', fontsize=12, fontweight='bold')

        # Add grid for readability
        ax.grid(axis='y', alpha=0.3)

        # Improve y-axis formatting based on data scale
        if time_stats['mean'].max() > 1000:
            # Use logarithmic scale for large time differences
            ax.set_yscale('log')
            ax.set_ylabel('Processing Time (ms, log scale)', fontsize=11)

        # Perform statistical test
        if len(self.df['method'].unique()) >= 2:
            f_stat, p_value = stats.f_oneway(
                *[self.df[self.df['method'] == method]['processing_time_ms'].values
                  for method in self.df['method'].unique()]
            )

            p_text = f"p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"
            ax.text(0.5, 0.02, f"ANOVA: {p_text}", transform=ax.transAxes,
                    ha='center', fontsize=9, style='italic')

        # Save figure
        output_path = self.output_dir / 'processing_time_comparison.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_radar_chart(self) -> str:
        """
        Create a radar chart comparing methods across all metrics.

        Returns:
        --------
        str
            Path to saved figure
        """
        # Calculate mean of each metric for each method
        radar_data = self.df.groupby('method')[self.metrics].mean().reset_index()

        # Number of metrics
        num_metrics = len(self.metrics)

        # Angle for each metric
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Setup figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Add each method
        for method in radar_data['method']:
            values = radar_data[radar_data['method'] == method][self.metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop

            # Plot method
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=self.method_colors[method])
            ax.fill(angles, values, alpha=0.1, color=self.method_colors[method])

        # Set metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.metric_labels[m] for m in self.metrics])

        # Style radar chart
        ax.grid(True, alpha=0.3)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Title
        plt.title('Performance Comparison Across All Metrics', size=12, fontweight='bold', y=1.08)

        # Save figure
        output_path = self.output_dir / 'radar_chart_comparison.pdf'
        plt.savefig(output_path, bbox_inches='tight')
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
        fig = plt.figure(figsize=(12, 14))
        gs = GridSpec(3, 2, figure=fig)

        # Top row: Processing time and radar chart
        ax_time = fig.add_subplot(gs[0, 0])
        ax_radar = fig.add_subplot(gs[0, 1], polar=True)

        # Middle and bottom rows: Quality metrics
        axes_metrics = [
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1])
        ]

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
        ax_time.set_title('Processing Time Comparison', fontweight='bold')

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
        ax_radar.set_title('Multi-Metric Performance', fontweight='bold')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # 3. Quality metrics box plots (select the 4 most important metrics)
        key_metrics = self.metrics[:4]  # First 4 metrics

        for i, metric in enumerate(key_metrics):
            ax = axes_metrics[i]

            # Box plot
            sns.boxplot(
                x='method',
                y=metric,
                data=self.df,
                palette=self.method_colors,
                width=0.6,
                ax=ax
            )

            # Add individual points
            sns.stripplot(
                x='method',
                y=metric,
                data=self.df,
                color='black',
                size=2,
                alpha=0.3,
                jitter=True,
                ax=ax
            )

            # Set labels
            ax.set_xlabel('')
            ax.set_ylabel(self.metric_labels[metric])
            ax.set_title(self.metric_labels[metric], fontweight='bold')

            # Add statistical annotations if applicable
            self._add_statistical_annotations(ax, self.df, metric)

        # Main title
        fig.suptitle('Comprehensive Analysis of HRF Illumination Correction Methods',
                    fontsize=14, fontweight='bold', y=0.98)

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.94)

        # Save figure
        output_path = self.output_dir / 'comprehensive_analysis_grid.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualizations and return their file paths.

        Returns:
        --------
        Dict[str, str]
            Dictionary mapping visualization names to file paths
        """
        print("Generating box plots for quality metrics...")
        boxplots_path = self.create_boxplots()

        print("Generating processing time bar chart...")
        time_chart_path = self.create_processing_time_barchart()

        print("Generating radar chart comparison...")
        radar_chart_path = self.create_radar_chart()

        print("Generating comprehensive summary grid...")
        summary_grid_path = self.create_summary_grid()

        return {
            "quality_metrics_boxplots": boxplots_path,
            "processing_time_chart": time_chart_path,
            "radar_chart": radar_chart_path,
            "summary_grid": summary_grid_path
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
