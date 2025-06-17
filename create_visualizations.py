import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Carregar dados
df = pd.read_csv('results/data/results_dataframe.csv')

# 1. Gráfico de barras com erro padrão para todas as métricas
metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity',
           'edge_preservation_index', 'microaneurysm_visibility']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    # Calcular média e erro padrão
    grouped = df.groupby('method')[metric].agg(['mean', 'std']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(45)  # 45 imagens por método

    # Criar gráfico de barras
    bars = ax.bar(grouped['method'], grouped['mean'],
                  yerr=grouped['se'], capsize=5, alpha=0.8)

    # Personalizar
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor da Métrica', fontsize=12)
    ax.set_xlabel('Método', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # Adicionar valores nas barras
    for bar, mean_val in zip(bars, grouped['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + grouped['se'].max()*0.1,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=10)

# Remover subplot extra
axes[5].remove()

plt.tight_layout()
plt.savefig('results/figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Boxplot para visualizar distribuições
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    # Criar boxplot
    df.boxplot(column=metric, by='method', ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor da Métrica', fontsize=12)
    ax.set_xlabel('Método', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

# Remover subplot extra
axes[5].remove()

plt.suptitle('')  # Remove título automático do pandas
plt.tight_layout()
plt.savefig('results/data/metrics_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Gráfico de tempo de processamento
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

time_data = df.groupby('method')['processing_time_ms'].agg(['mean', 'std']).reset_index()
time_data['se'] = time_data['std'] / np.sqrt(45)

bars = ax.bar(time_data['method'], time_data['mean'],
              yerr=time_data['se'], capsize=5, alpha=0.8, color='skyblue')

ax.set_title('Tempo de Processamento por Método', fontsize=16, fontweight='bold')
ax.set_ylabel('Tempo (ms)', fontsize=12)
ax.set_xlabel('Método', fontsize=12)
ax.set_yscale('log')  # Escala logarítmica devido à grande diferença

# Adicionar valores nas barras
for bar, mean_val in zip(bars, time_data['mean']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
            f'{mean_val:.0f} ms', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/data/processing_time.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Heatmap de correlação entre métricas
correlation_matrix = df[metrics].corr()

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlação'})
ax.set_title('Matriz de Correlação entre Métricas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Radar chart para comparação geral
from math import pi

# Normalizar métricas para escala 0-1
df_norm = df.copy()
for metric in metrics:
    if metric in ['illumination_uniformity', 'contrast_ratio', 'edge_preservation_index']:
        # Métricas onde maior é melhor
        df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    else:
        # Para vessel_clarity_index e microaneurysm_visibility, maior também é melhor
        df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Calcular médias normalizadas por método
radar_data = df_norm.groupby('method')[metrics].mean()

# Configurar radar chart
angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]  # Fechar o círculo

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))

colors = ['red', 'blue', 'green', 'orange']
methods = radar_data.index

for i, method in enumerate(methods):
    values = radar_data.loc[method].values.tolist()
    values += values[:1]  # Fechar o círculo

    ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

# Personalizar
ax.set_xticks(angles[:-1])
ax.set_xticklabels([metric.replace('_', '\n').title() for metric in metrics])
ax.set_ylim(0, 1)
ax.set_title('Overall Performance Comparison\n(Normalized Metrics)',
             fontsize=16, fontweight='normal', pad=20)  # Not bold
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('results/data/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Todos os gráficos foram gerados com sucesso!")
print("Arquivos criados:")
print("- metrics_comparison.png")
print("- metrics_boxplot.png")
print("- processing_time.png")
print("- correlation_heatmap.png")
print("- radar_chart.png")

