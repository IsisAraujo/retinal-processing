"""
Enhanced statistical analysis and visualization for academic publication
Implements rigorous statistical tests with FDR correction and power analysis
Following current standards for medical imaging research (2024-2025)

References:
[1] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B (Methodological), 57(1), 289-300.
[2] Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    Lawrence Erlbaum Associates.
[3] Editage Insights (2024). Statistical approaches for analyzing imaging data: An overview.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import cv2

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

class EnhancedStatisticalAnalyzer:
    """
    Performs rigorous statistical analysis following current academic standards
    with FDR correction and power analysis
    """

    @staticmethod
    def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Cohen's d effect size and interpretation

        Reference: Cohen, J. (1988). Statistical power analysis for the behavioral sciences.

        Returns:
            Tuple of (effect_size, interpretation)
        """
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                             (len(group2) - 1) * np.var(group2, ddof=1)) /
                            (len(group1) + len(group2) - 2))

        if pooled_std == 0:
            return 0.0, "No effect"

        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Cohen's conventions for effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "Negligible"
        elif abs_d < 0.5:
            interpretation = "Small"
        elif abs_d < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"

        return cohens_d, interpretation

    @staticmethod
    def calculate_statistical_power(effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
        """
        Calculate observed statistical power for two-sample t-test

        Reference: Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
        """
        try:
            power = ttest_power(effect_size, sample_size, alpha, alternative='two-sided')
            return float(power)
        except:
            return np.nan

    @staticmethod
    def perform_fdr_correction(p_values: List[float], method: str = 'bh') -> np.ndarray:
        """
        Perform False Discovery Rate correction using Benjamini-Hochberg method

        Reference: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate

        Args:
            p_values: List of p-values to correct
            method: 'bh' for Benjamini-Hochberg, 'by' for Benjamini-Yekutieli

        Returns:
            Array of FDR-corrected p-values
        """
        if len(p_values) == 0:
            return np.array([])

        # Use scipy's implementation (recommended over statsmodels for consistency)
        corrected_p = false_discovery_control(p_values, method=method)
        return corrected_p

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
        Enhanced Shapiro-Wilk test for normality with better handling
        """
        results = {}
        for metric in metrics:
            results[metric] = {}
            for method in df['method'].unique():
                data = df[df['method'] == method][metric].values
                if len(data) >= 3:
                    try:
                        statistic, p_value = stats.shapiro(data)
                        results[metric][method] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'normal': bool(p_value > 0.05),
                            'sample_size': len(data)
                        }
                    except:
                        results[metric][method] = {
                            'statistic': np.nan,
                            'p_value': np.nan,
                            'normal': False,
                            'sample_size': len(data)
                        }
        return results

    @staticmethod
    def perform_homoscedasticity_test(df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Enhanced Levene's test for homogeneity of variances
        """
        results = {}
        for metric in metrics:
            groups = [df[df['method'] == method][metric].values
                     for method in df['method'].unique()]
            # Remove empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                try:
                    statistic, p_value = stats.levene(*groups, center='median')  # More robust
                    results[metric] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'equal_variances': bool(p_value > 0.05),
                        'groups_tested': len(groups)
                    }
                except:
                    results[metric] = {
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'equal_variances': False,
                        'groups_tested': len(groups)
                    }
        return results

    @staticmethod
    def perform_enhanced_comparison_tests(
        df: pd.DataFrame, metrics: List[str], normality_results: Dict,
        homoscedasticity_results: Dict) -> Dict:
        """
        Enhanced comparison tests with FDR correction and power analysis
        """
        results = {}
        all_p_values = []
        test_info = []

        for metric in metrics:
            groups = [df[df['method'] == method][metric].values
                     for method in df['method'].unique()]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                continue

            # Check assumptions
            normal_methods = [method for method in df['method'].unique()
                            if normality_results.get(metric, {}).get(method, {}).get('normal', False)]
            all_normal = len(normal_methods) == len(df['method'].unique())
            equal_variances = homoscedasticity_results.get(metric, {}).get('equal_variances', False)

            if all_normal and equal_variances and len(groups) > 2:
                # Parametric test (ANOVA)
                try:
                    statistic, p_value = stats.f_oneway(*groups)
                    test_name = 'One-way ANOVA'
                except:
                    statistic, p_value = stats.kruskal(*groups)
                    test_name = 'Kruskal-Wallis H-test (fallback)'
            else:
                # Non-parametric test (Kruskal-Wallis)
                try:
                    statistic, p_value = stats.kruskal(*groups)
                    test_name = 'Kruskal-Wallis H-test'
                except:
                    statistic, p_value = np.nan, 1.0
                    test_name = 'Test failed'

            # Store for FDR correction
            all_p_values.append(p_value)
            test_info.append((metric, test_name, statistic, p_value))

            # Calculate effect size and power for first two groups
            if len(groups) >= 2:
                effect_size, effect_interpretation = EnhancedStatisticalAnalyzer.calculate_effect_size(
                    groups[0], groups[1]
                )
                power = EnhancedStatisticalAnalyzer.calculate_statistical_power(
                    effect_size, min(len(groups[0]), len(groups[1]))
                )
            else:
                effect_size, effect_interpretation = 0.0, "Not calculated"
                power = np.nan

            results[metric] = {
                'test': test_name,
                'statistic': float(statistic) if not np.isnan(statistic) else np.nan,
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
                'significant_uncorrected': bool(p_value < 0.05) if not np.isnan(p_value) else False,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpretation,
                'statistical_power': power,
                'assumptions_met': all_normal and equal_variances
            }

        # Apply FDR correction
        if all_p_values:
            try:
                fdr_corrected_p = EnhancedStatisticalAnalyzer.perform_fdr_correction(all_p_values)
                bonferroni_corrected_p = [min(1.0, p * len(all_p_values)) for p in all_p_values]

                for i, (metric, _, _, _) in enumerate(test_info):
                    results[metric]['bonferroni_corrected_p'] = float(bonferroni_corrected_p[i])
                    results[metric]['fdr_corrected_p'] = float(fdr_corrected_p[i])
                    results[metric]['significant_bonferroni'] = bool(bonferroni_corrected_p[i] < 0.05)
                    results[metric]['significant_fdr'] = bool(fdr_corrected_p[i] < 0.05)
            except Exception as e:
                # Fallback if correction fails
                for metric in results:
                    results[metric]['bonferroni_corrected_p'] = results[metric]['p_value']
                    results[metric]['fdr_corrected_p'] = results[metric]['p_value']
                    results[metric]['significant_bonferroni'] = results[metric]['significant_uncorrected']
                    results[metric]['significant_fdr'] = results[metric]['significant_uncorrected']

            # Post-hoc tests if significant after FDR correction
            for metric in results:
                if results[metric]['significant_fdr']:
                    results[metric]['post_hoc'] = EnhancedStatisticalAnalyzer._perform_enhanced_post_hoc_tests(
                        df, metric, results[metric]['assumptions_met']
                    )

        return results

    @staticmethod
    def _perform_enhanced_post_hoc_tests(df: pd.DataFrame, metric: str,
                                       parametric: bool) -> Dict:
        """Perform pairwise post-hoc tests with enhanced corrections"""
        methods = df['method'].unique()
        n_comparisons = len(methods) * (len(methods) - 1) // 2

        results = {}
        pairwise_p_values = []
        pairwise_info = []

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = df[df['method'] == method1][metric].values
                data2 = df[df['method'] == method2][metric].values

                if len(data1) < 2 or len(data2) < 2:
                    continue

                if parametric:
                    try:
                        statistic, p_value = stats.ttest_ind(data1, data2)
                        test = 't-test'
                    except:
                        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test = 'Mann-Whitney U (fallback)'
                else:
                    try:
                        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test = 'Mann-Whitney U'
                    except:
                        statistic, p_value = np.nan, 1.0
                        test = 'Test failed'

                # Calculate effect size
                effect_size, effect_interpretation = EnhancedStatisticalAnalyzer.calculate_effect_size(
                    data1, data2
                )

                pairwise_p_values.append(p_value)
                pairwise_info.append((method1, method2, test, statistic, effect_size, effect_interpretation))

        # Apply FDR correction to post-hoc tests
        if pairwise_p_values:
            try:
                fdr_corrected = EnhancedStatisticalAnalyzer.perform_fdr_correction(pairwise_p_values)
                bonferroni_corrected = [min(1.0, p * len(pairwise_p_values)) for p in pairwise_p_values]

                for i, (m1, m2, test, stat, eff_size, eff_interp) in enumerate(pairwise_info):
                    results[f"{m1}_vs_{m2}"] = {
                        'test': test,
                        'statistic': float(stat) if not np.isnan(stat) else np.nan,
                        'p_value': float(pairwise_p_values[i]),
                        'bonferroni_corrected_p': float(bonferroni_corrected[i]),
                        'fdr_corrected_p': float(fdr_corrected[i]),
                        'significant_fdr': bool(fdr_corrected[i] < 0.05),
                        'significant_bonferroni': bool(bonferroni_corrected[i] < 0.05),
                        'effect_size': eff_size,
                        'effect_interpretation': eff_interp
                    }
            except:
                # Fallback
                for i, (m1, m2, test, stat, eff_size, eff_interp) in enumerate(pairwise_info):
                    results[f"{m1}_vs_{m2}"] = {
                        'test': test,
                        'statistic': float(stat) if not np.isnan(stat) else np.nan,
                        'p_value': float(pairwise_p_values[i]),
                        'bonferroni_corrected_p': float(pairwise_p_values[i]),
                        'fdr_corrected_p': float(pairwise_p_values[i]),
                        'significant_fdr': bool(pairwise_p_values[i] < 0.05),
                        'significant_bonferroni': bool(pairwise_p_values[i] < 0.05),
                        'effect_size': eff_size,
                        'effect_interpretation': eff_interp
                    }

        return results

    @staticmethod
    def create_performance_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create enhanced scatter plot of quality vs computational performance"""
        # Calculate composite quality score including new metrics
        quality_metrics = ['psnr', 'ssim', 'contrast_ratio', 'vessel_clarity_index']
        available_metrics = [m for m in quality_metrics if m in df.columns]

        if available_metrics:
            df['quality_score'] = df[available_metrics].mean(axis=1)
        else:
            # Fallback to original metrics
            quality_metrics = ['contrast_ratio', 'vessel_clarity_index', 'microaneurysm_visibility']
            df['quality_score'] = df[quality_metrics].mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 7))

        methods = df['method'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for method, color in zip(methods, colors):
            method_data = df[df['method'] == method]
            scatter = ax.scatter(method_data['processing_time_ms'],
                               method_data['quality_score'],
                               label=method, color=color, alpha=0.7, s=60)

        ax.set_xlabel('Processing Time (ms)')
        ax.set_ylabel('Composite Quality Score')
        ax.set_title('Quality-Performance Trade-off Analysis\n(Higher quality score and lower time is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add logarithmic scale if time differences are large
        time_range = df['processing_time_ms'].max() / df['processing_time_ms'].min()
        if time_range > 100:
            ax.set_xscale('log')

        output_path = output_dir / 'enhanced_performance_analysis.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def create_enhanced_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create enhanced correlation heatmap including new metrics"""
        # Include all available metrics
        all_metrics = ['psnr', 'ssim', 'contrast_ratio', 'vessel_clarity_index',
                      'illumination_uniformity', 'edge_preservation_index', 'microaneurysm_visibility']
        available_metrics = [m for m in all_metrics if m in df.columns]

        correlation_matrix = df[available_metrics].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show only lower triangle

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   mask=mask, ax=ax, fmt='.3f')

        ax.set_title('Enhanced Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()

        output_path = output_dir / 'enhanced_correlation_heatmap.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def generate_power_analysis_report(comparison_results: Dict, output_dir: Path) -> Path:
        """Generate statistical power analysis report"""
        report_data = []

        for metric, results in comparison_results.items():
            if 'statistical_power' in results:
                report_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Statistical_Power': results.get('statistical_power', np.nan),
                    'Effect_Size': results.get('effect_size', np.nan),
                    'Effect_Interpretation': results.get('effect_interpretation', 'Unknown'),
                    'Sample_Adequate': 'Yes' if results.get('statistical_power', 0) >= 0.8 else 'No'
                })

        if report_data:
            df_power = pd.DataFrame(report_data)

            # Save as CSV
            csv_path = output_dir / 'power_analysis_report.csv'
            df_power.to_csv(csv_path, index=False)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Usar as mesmas cores do enhanced_performance_analysis.pdf
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            # Power analysis plot com a cor do primeiro método
            ax1.bar(range(len(df_power)), df_power['Statistical_Power'], color=colors[0], alpha=0.7)
            ax1.axhline(y=0.8, color='red', linestyle='--', label='Adequate Power (0.8)')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Statistical Power')
            ax1.set_title('Statistical Power by Metric')
            ax1.set_xticks(range(len(df_power)))
            ax1.set_xticklabels(df_power['Metric'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Effect size plot com a cor do segundo método
            effect_sizes = pd.to_numeric(df_power['Effect_Size'], errors='coerce')
            ax2.bar(range(len(df_power)), np.abs(effect_sizes), color=colors[1], alpha=0.7)
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('|Effect Size| (Cohen\'s d)')
            ax2.set_title('Effect Sizes by Metric')
            ax2.set_xticks(range(len(df_power)))
            ax2.set_xticklabels(df_power['Metric'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = output_dir / 'power_analysis_plot.pdf'
            plt.savefig(plot_path, format='pdf')
            plt.close()

            return csv_path
        else:
            return None

class AcademicVisualizerEnhanced:
    """
    Enhanced visualizer with additional plots and improved statistics
    """

    @staticmethod
    def create_enhanced_comparison_grid(results: List[Dict], output_dir: Path) -> Path:
        """Create enhanced visual comparison grid with metrics overlay"""
        # Select representative images
        image_ids = list(set(r['image_id'] for r in results))[:3]
        methods = ['CLAHE', 'SSR', 'MSR', 'MSRCR']

        fig, axes = plt.subplots(len(image_ids), len(methods) + 1,
                                figsize=(18, 4 * len(image_ids)))

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

            # Processed images with metrics
            for j, method in enumerate(methods):
                method_result = next((r for r in results
                                    if r['image_id'] == image_id and r['method'] == method), None)
                if method_result and 'image' in method_result:
                    axes[i, j+1].imshow(cv2.cvtColor(method_result['image'], cv2.COLOR_BGR2RGB))

                    # Add metrics as text overlay
                    if 'metrics' in method_result:
                        metrics = method_result['metrics']
                        if 'psnr' in metrics and 'ssim' in metrics:
                            text = f"PSNR: {metrics['psnr']:.1f}\nSSIM: {metrics['ssim']:.3f}"
                        else:
                            text = f"Contrast: {metrics.get('contrast_ratio', 0):.3f}"

                        axes[i, j+1].text(0.02, 0.98, text, transform=axes[i, j+1].transAxes,
                                         fontsize=8, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    axes[i, j+1].set_title(method if i == 0 else '')
                    axes[i, j+1].axis('off')

        plt.suptitle('Enhanced Visual Comparison of Illumination Correction Methods', fontsize=16)
        plt.tight_layout()

        output_path = output_dir / 'enhanced_comparison_grid.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path

    @staticmethod
    def create_method_ranking_chart(df: pd.DataFrame, output_dir: Path) -> Path:
        """Create ranking chart showing method performance across metrics"""
        methods = df['method'].unique()
        metrics = ['psnr', 'ssim', 'contrast_ratio', 'vessel_clarity_index',
                  'illumination_uniformity', 'edge_preservation_index', 'microaneurysm_visibility']
        available_metrics = [m for m in metrics if m in df.columns]

        # Calculate rankings (1 = best, 4 = worst for each metric)
        rankings = pd.DataFrame(index=methods, columns=available_metrics)

        for metric in available_metrics:
            metric_means = df.groupby('method')[metric].mean()
            # Higher is better for all these metrics
            rankings[metric] = metric_means.rank(ascending=False).astype(int)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(rankings, annot=True, cmap='coolwarm', center=2.5,
           cbar_kws={'label': 'Ranking (1=Best, 4=Worst)'}, fmt='d', ax=ax)

        ax.set_title('Method Performance Rankings Across All Metrics', fontsize=12)
        ax.set_xlabel('Evaluation Metrics')
        ax.set_ylabel('Methods')

        plt.tight_layout()

        output_path = output_dir / 'method_ranking_heatmap.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()

        return output_path
