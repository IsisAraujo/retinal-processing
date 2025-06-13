"""
Statistical analysis and visualization for academic publication
Implements rigorous statistical tests and publication-ready figures
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

# Academic publication settings
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

class StatisticalAnalyzer:
    """
    Performs rigorous statistical analysis following academic standards
    """

    @staticmethod
    def create_performance_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create scatter plot of quality vs computational performance"""
        # Calculate composite quality score
        quality_metrics = ['contrast_ratio', 'vessel_clarity_index', 'microaneurysm_visibility']
        df['quality_score'] = df[quality_metrics].mean(axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))

        methods = df['method'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            ax.scatter(method_data['processing_time_ms'],
                      method_data['quality_score'],
                      label=method, color=color, alpha=0.6, s=50)

        ax.set_xlabel('Processing Time (ms)')
        ax.set_ylabel('Composite Quality Score')
        ax.set_title('Quality-Performance Trade-off Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / 'performance_analysis.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def create_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create correlation heatmap of metrics"""
        metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity',
                  'edge_preservation_index', 'microaneurysm_visibility']

        correlation_matrix = df[metrics].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()

        output_path = output_dir / 'correlation_heatmap.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def results_dataframe(results: List[Dict]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        for result in results:
            row = {
                'image_id': result['image_id'],
                'method': result['method'],
                'processing_time_ms': result['processing_time_ms']
            }
            # Add metrics
            for metric, value in result['metrics'].items():
                row[metric] = value
            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def perform_normality_tests(df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Shapiro-Wilk test for normality
        Reference: Shapiro & Wilk (1965)
        """
        results = {}
        for metric in metrics:
            results[metric] = {}
            for method in df['method'].unique():
                data = df[df['method'] == method][metric].values
                if len(data) >= 3:
                    statistic, p_value = stats.shapiro(data)
                    results[metric][method] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'normal': p_value > 0.05
                    }
        return results

    @staticmethod
    def perform_homoscedasticity_test(df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Levene's test for homogeneity of variances
        Reference: Levene (1960)
        """
        results = {}
        for metric in metrics:
            groups = [df[df['method'] == method][metric].values
                     for method in df['method'].unique()]
            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                statistic, p_value = stats.levene(*groups)
                results[metric] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'equal_variances': p_value > 0.05
                }
        return results

    @staticmethod
    def perform_comparison_tests(
        df: pd.DataFrame, metrics: List[str], normality_results: Dict,
        homoscedasticity_results: Dict) -> Dict:
        """
        Perform appropriate comparison tests based on assumptions
        ANOVA for normal data with equal variances, Kruskal-Wallis otherwise
        """
        results = {}

        for metric in metrics:
            groups = [df[df['method'] == method][metric].values
                     for method in df['method'].unique()]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                continue

            # Check assumptions
            all_normal = all(normality_results.get(metric, {}).get(method, {}).get('normal', False)
                           for method in df['method'].unique())
            equal_variances = homoscedasticity_results.get(metric, {}).get('equal_variances', False)

            if all_normal and equal_variances:
                # Parametric test (ANOVA)
                statistic, p_value = stats.f_oneway(*groups)
                test_name = 'One-way ANOVA'
            else:
                # Non-parametric test (Kruskal-Wallis)
                statistic, p_value = stats.kruskal(*groups)
                test_name = 'Kruskal-Wallis H-test'

            results[metric] = {
                'test': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # Post-hoc tests if significant
            if p_value < 0.05:
                results[metric]['post_hoc'] = StatisticalAnalyzer._perform_post_hoc_tests(
                    df, metric, all_normal and equal_variances
                )

        return results

    @staticmethod
    def _perform_post_hoc_tests(df: pd.DataFrame, metric: str,
                               parametric: bool) -> Dict:
        """Perform pairwise post-hoc tests with Bonferroni correction"""
        methods = df['method'].unique()
        n_comparisons = len(methods) * (len(methods) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction

        results = {}
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = df[df['method'] == method1][metric].values
                data2 = df[df['method'] == method2][metric].values

                if parametric:
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    test = 't-test'
                else:
                    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    test = 'Mann-Whitney U'

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) +
                                    (len(data2)-1)*np.var(data2, ddof=1)) /
                                   (len(data1) + len(data2) - 2))
                if pooled_std > 0:
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                else:
                    cohens_d = 0.0

                results[f"{method1}_vs_{method2}"] = {
                    'test': test,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha_corrected,
                    'cohens_d': cohens_d
                }

        return results

class AcademicVisualizer:
    """
    Creates publication-ready figures following academic standards
    """

    @staticmethod
    def create_comparison_grid(results: List[Dict], output_dir: Path) -> Path:
        """Create visual comparison grid of methods"""
        # Select representative images
        image_ids = list(set(r['image_id'] for r in results))[:3]
        methods = ['CLAHE', 'SSR', 'MSR', 'MSRCR']

        fig, axes = plt.subplots(len(image_ids), len(methods) + 1,
                                figsize=(15, 3 * len(image_ids)))

        if len(image_ids) == 1:
            axes = axes.reshape(1, -1)

        for i, image_id in enumerate(image_ids):
            # Original image
            original_result = next((r for r in results
                                  if r['image_id'] == image_id and r['method'] == 'original'), None)
            if original_result and 'image' in original_result:
                axes[i, 0].imshow(cv2.cvtColor(original_result['image'], cv2.COLOR_BGR2RGB))
                axes[i, 0].set_title('Original' if i == 0 else '')
                axes[i, 0].axis('off')

            # Processed images
            for j, method in enumerate(methods):
                method_result = next((r for r in results
                                    if r['image_id'] == image_id and r['method'] == method), None)
                if method_result and 'image' in method_result:
                    axes[i, j+1].imshow(cv2.cvtColor(method_result['image'], cv2.COLOR_BGR2RGB))
                    axes[i, j+1].set_title(method if i == 0 else '')
                    axes[i, j+1].axis('off')

        plt.suptitle('Visual Comparison of Illumination Correction Methods', fontsize=12)
        plt.tight_layout()

        output_path = output_dir / 'comparison_grid.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def create_metrics_boxplots(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create boxplots for metric distributions"""
        metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity',
                  'edge_preservation_index', 'microaneurysm_visibility']

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                df.boxplot(column=metric, by='method', ax=axes[i])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_xlabel('Method')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)

        # Remove empty subplot
        if len(metrics) < 6:
            fig.delaxes(axes[-1])

        plt.suptitle('Distribution of Evaluation Metrics by Method', fontsize=12)
        plt.tight_layout()

        output_path = output_dir / 'metrics_boxplots.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def create_statistical_summary_table(comparison_results: Dict,
                                       output_dir: Path) -> Path:
        """Create LaTeX table of statistical results"""
        rows = []
        for metric, results in comparison_results.items():
            row = {
                'Metric': metric.replace('_', ' ').title(),
                'Test': results['test'],
                'Statistic': f"{results['statistic']:.3f}",
                'p-value': f"{results['p_value']:.4f}",
                'Significant': 'Yes' if results['significant'] else 'No'
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save as CSV
        csv_path = output_dir / 'statistical_summary.csv'
        df.to_csv(csv_path, index=False)

        # Save as LaTeX
        latex_path = output_dir / 'statistical_summary.tex'
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))

        return csv_path

    @staticmethod
    def create_method_comparison_table(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create LaTeX table comparing methods across metrics"""
        metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity',
                  'edge_preservation_index', 'microaneurysm_visibility']

        # Aggregate metrics by method
        summary = df.groupby('method')[metrics].mean().reset_index()

        # Save as CSV
        csv_path = output_dir / 'method_comparison.csv'
        summary.to_csv(csv_path, index=False)

        # Save as LaTeX
        latex_path = output_dir / 'method_comparison.tex'
        with open(latex_path, 'w') as f:
            f.write(summary.to_latex(index=False, escape=False))

        return csv_path

    @staticmethod
    def create_performance_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
        """
        Create performance scatter plot comparing metrics across methods
        """
        # Calculate composite quality score
        quality_metrics = ['contrast_ratio', 'vessel_clarity_index', 'microaneurysm_visibility']
        df['quality_score'] = df[quality_metrics].mean(axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))

        methods = df['method'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            ax.scatter(method_data['processing_time_ms'],
                      method_data['quality_score'],
                      label=method, color=color, alpha=0.6, s=50)

        ax.set_xlabel('Processing Time (ms)')
        ax.set_ylabel('Composite Quality Score')
        ax.set_title('Quality-Performance Trade-off Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / 'performance_analysis.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path
