"""

Módulo de visualização científica para análise comparativa de métodos
de correção de iluminação em imagens de fundo de olho (Dataset HRF).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Configuração do matplotlib para publicação científica
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif'
})

class HRFVisualizationAnalyzer:
    """
    Classe para análise visual comparativa rigorosa dos métodos de correção de iluminação
    para imagens de fundo de olho do dataset HRF.
    """

    def __init__(self):
        self.methods = ['original', 'clahe', 'ssr', 'msr', 'msrcr']
        self.method_names = {
            'original': 'Original',
            'clahe': 'CLAHE',
            'ssr': 'SSR',
            'msr': 'MSR',
            'msrcr': 'MSRCR'
        }
        self.colors = {
            'original': '#2E3440',
            'clahe': '#5E81AC',
            'ssr': '#88C0D0',
            'msr': '#81A1C1',
            'msrcr': '#8FBCBB'
        }

    def plot_comparative_grid(self, results, image_name, save_path=None):
        """
        Cria grid comparativo 2x3 para análise visual side-by-side
        Critical for fast fail assessment
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Análise Comparativa: {image_name}', fontsize=16, fontweight='bold')

        methods_to_plot = self.methods
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]

        for i, method in enumerate(methods_to_plot):
            if i < len(positions):
                row, col = positions[i]
                ax = axes[row, col]

                # Converter BGR para RGB para matplotlib
                img = cv2.cvtColor(results[image_name][method], cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f'{self.method_names[method]}', fontweight='bold')
                ax.axis('off')

                # Adicionar métricas na imagem
                if method != 'original':
                    metrics = results[image_name]['metrics'][method]
                    text = f"PSNR: {metrics['PSNR']:.2f}\nSSIM: {metrics['SSIM']:.3f}\nContrast: {metrics['Contrast']:.1f}"
                    ax.text(0.02, 0.98, text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                           facecolor='white', alpha=0.8), fontsize=8)

        # Remover subplot vazio
        axes[1, 2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_metrics_comparison(self, results, save_path=None):
        """
        Gráfico de barras comparativo das métricas - decisão rápida
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Agregar métricas de todas as imagens
        metrics_data = {'PSNR': [], 'SSIM': [], 'Contrast': [], 'Method': []}

        for img_name, data in results.items():
            for method in ['clahe', 'ssr', 'msr', 'msrcr']:
                metrics = data['metrics'][method]
                metrics_data['PSNR'].append(metrics['PSNR'])
                metrics_data['SSIM'].append(metrics['SSIM'])
                metrics_data['Contrast'].append(metrics['Contrast'])
                metrics_data['Method'].append(self.method_names[method])

        df = pd.DataFrame(metrics_data)

        # PSNR
        sns.boxplot(data=df, x='Method', y='PSNR', ax=axes[0], palette=[self.colors[k] for k in ['clahe', 'ssr', 'msr', 'msrcr']])
        axes[0].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].tick_params(axis='x', rotation=45)

        # SSIM
        sns.boxplot(data=df, x='Method', y='SSIM', ax=axes[1], palette=[self.colors[k] for k in ['clahe', 'ssr', 'msr', 'msrcr']])
        axes[1].set_title('Structural Similarity Index', fontweight='bold')
        axes[1].set_ylabel('SSIM')
        axes[1].tick_params(axis='x', rotation=45)

        # Contrast
        sns.boxplot(data=df, x='Method', y='Contrast', ax=axes[2], palette=[self.colors[k] for k in ['clahe', 'ssr', 'msr', 'msrcr']])
        axes[2].set_title('RMS Contrast Enhancement', fontweight='bold')
        axes[2].set_ylabel('Contrast (RMS)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return df

    def plot_radar_comparison(self, results, save_path=None):
        """
        Radar chart para análise multi-dimensional - visão holística
        """
        # Calcular métricas médias
        method_averages = {}
        for method in ['clahe', 'ssr', 'msr', 'msrcr']:
            psnr_vals = []
            ssim_vals = []
            contrast_vals = []

            for img_name, data in results.items():
                metrics = data['metrics'][method]
                psnr_vals.append(metrics['PSNR'])
                ssim_vals.append(metrics['SSIM'])
                contrast_vals.append(metrics['Contrast'])

            method_averages[method] = {
                'PSNR': np.mean(psnr_vals),
                'SSIM': np.mean(ssim_vals),
                'Contrast': np.mean(contrast_vals)
            }

        # Normalizar métricas para radar (0-1)
        categories = ['PSNR', 'SSIM', 'Contrast']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

        for method in ['clahe', 'ssr', 'msr', 'msrcr']:
            values = [method_averages[method][cat] for cat in categories]
            # Normalizar individualmente cada métrica
            values_norm = []
            values_norm.append((values[0] - 15) / (35 - 15))  # PSNR typical range
            values_norm.append(values[1])  # SSIM já está 0-1
            values_norm.append((values[2] - 20) / (60 - 20))  # Contrast typical range

            # Garantir que valores estão no range 0-1
            values_norm = [max(0, min(1, v)) for v in values_norm]
            values_norm = np.concatenate((values_norm, [values_norm[0]]))  # Complete the circle

            ax.plot(angles, values_norm, 'o-', linewidth=2,
                   label=self.method_names[method], color=self.colors[method])
            ax.fill(angles, values_norm, alpha=0.25, color=self.colors[method])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Análise Comparativa Multi-dimensional\n(Valores Normalizados)',
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_histogram_analysis(self, results, image_name, save_path=None):
        """
        Análise de histogramas - crítico para avaliar distribuição de intensidade
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Análise de Histogramas: {image_name}', fontsize=16, fontweight='bold')

        methods_to_plot = self.methods
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]

        for i, method in enumerate(methods_to_plot):
            if i < len(positions):
                row, col = positions[i]
                ax = axes[row, col]

                img = results[image_name][method]
                # Converter para grayscale para análise de histograma
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Calcular histograma
                hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

                ax.plot(hist, color=self.colors[method], linewidth=2)
                ax.set_title(f'{self.method_names[method]}', fontweight='bold')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

                # Adicionar estatísticas
                mean_intensity = np.mean(gray_img)
                std_intensity = np.std(gray_img)
                ax.axvline(mean_intensity, color='red', linestyle='--', alpha=0.7)
                ax.text(0.7, 0.9, f'μ: {mean_intensity:.1f}\nσ: {std_intensity:.1f}',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Remover subplot vazio
        axes[1, 2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_statistical_significance(self, results, save_path=None):
        """
        Teste de significância estatística entre métodos
        """
        # Coletar dados para teste estatístico
        methods_data = {}
        for method in ['clahe', 'ssr', 'msr', 'msrcr']:
            methods_data[method] = {
                'PSNR': [],
                'SSIM': [],
                'Contrast': []
            }

            for img_name, data in results.items():
                metrics = data['metrics'][method]
                methods_data[method]['PSNR'].append(metrics['PSNR'])
                methods_data[method]['SSIM'].append(metrics['SSIM'])
                methods_data[method]['Contrast'].append(metrics['Contrast'])

        # Realizar testes de Kruskal-Wallis (não-paramétrico)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        metrics = ['PSNR', 'SSIM', 'Contrast']
        for i, metric in enumerate(metrics):
            # Coletar dados para o teste
            data_for_test = [methods_data[method][metric] for method in ['clahe', 'ssr', 'msr', 'msrcr']]

            # Teste de Kruskal-Wallis
            h_stat, p_value = stats.kruskal(*data_for_test)

            # Boxplot com anotação estatística
            bp = axes[i].boxplot(data_for_test, labels=[self.method_names[m] for m in ['clahe', 'ssr', 'msr', 'msrcr']],
                               patch_artist=True)

            # Colorir boxes
            colors_list = [self.colors[m] for m in ['clahe', 'ssr', 'msr', 'msrcr']]
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[i].set_title(f'{metric}\nKruskal-Wallis: H={h_stat:.3f}, p={p_value:.4f}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

            # Adicionar significância
            if p_value < 0.05:
                axes[i].text(0.5, 0.95, '***SIGNIFICATIVO***', transform=axes[i].transAxes,
                           ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return methods_data

    def generate_summary_table(self, results):
        """
        Tabela resumo para decisão rápida - formato publicação
        """
        summary_data = []

        for method in ['clahe', 'ssr', 'msr', 'msrcr']:
            psnr_vals = []
            ssim_vals = []
            contrast_vals = []

            for img_name, data in results.items():
                metrics = data['metrics'][method]
                psnr_vals.append(metrics['PSNR'])
                ssim_vals.append(metrics['SSIM'])
                contrast_vals.append(metrics['Contrast'])

            summary_data.append({
                'Method': self.method_names[method],
                'PSNR (dB)': f"{np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f}",
                'SSIM': f"{np.mean(ssim_vals):.3f} ± {np.std(ssim_vals):.3f}",
                'Contrast': f"{np.mean(contrast_vals):.1f} ± {np.std(contrast_vals):.1f}",
                'Rank Score': np.mean(psnr_vals) * 0.4 + np.mean(ssim_vals) * 100 * 0.3 + np.mean(contrast_vals) * 0.3
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Rank Score', ascending=False)
        df_summary['Ranking'] = range(1, len(df_summary) + 1)

        print("=" * 80)
        print("RESUMO EXECUTIVO - RANKING DOS MÉTODOS")
        print("=" * 80)
        print(df_summary[['Ranking', 'Method', 'PSNR (dB)', 'SSIM', 'Contrast']].to_string(index=False))
        print("=" * 80)

        return df_summary

def run_comprehensive_analysis(results):
    """
    Função principal para executar análise completa
    """
    analyzer = HRFVisualizationAnalyzer()

    print("🔬 INICIANDO ANÁLISE COMPARATIVA RIGOROSA...")

    # 1. Análise de uma imagem específica (grid comparativo)
    first_image = list(results.keys())[0]
    print(f"📊 Análise visual comparativa: {first_image}")
    analyzer.plot_comparative_grid(results, first_image)

    # 2. Comparação de métricas
    print("📈 Análise estatística das métricas...")
    df_metrics = analyzer.plot_metrics_comparison(results)

    # 3. Análise radar multi-dimensional
    print("🎯 Análise multi-dimensional (Radar)...")
    analyzer.plot_radar_comparison(results)

    # 4. Análise de histogramas
    print("📊 Análise de distribuição de intensidade...")
    analyzer.plot_histogram_analysis(results, first_image)

    # 5. Teste de significância estatística
    print("🧮 Teste de significância estatística...")
    statistical_data = analyzer.plot_statistical_significance(results)

    # 6. Tabela resumo final
    print("📋 Gerando resumo executivo...")
    summary_table = analyzer.generate_summary_table(results)

    print("\n✅ ANÁLISE CONCLUÍDA!")
    print("💡 RECOMENDAÇÃO: Examine os resultados visuais e estatísticos para decisão final.")

    return {
        'metrics_df': df_metrics,
        'statistical_data': statistical_data,
        'summary_table': summary_table,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    print("⚠️  Este módulo deve ser importado e usado com dados do hrf_preprocessing.py")
    print("🚀 Execute: python main_hrf_analysis.py")
