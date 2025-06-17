import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Carregar dados
df = pd.read_csv('/home/ubuntu/upload/results_dataframe.csv')

# Carregar análise estatística
with open('/home/ubuntu/upload/statistical_analysis.json', 'r') as f:
    stats_data = json.load(f)

print("Dados carregados com sucesso!")
print(f"Shape do dataframe: {df.shape}")
print(f"Métodos únicos: {df['method'].unique()}")
print(f"Métricas disponíveis: {[col for col in df.columns if col not in ['image_id', 'method', 'processing_time_ms']]}")

# Estatísticas descritivas por método
print("\n=== ESTATÍSTICAS DESCRITIVAS POR MÉTODO ===")
metrics = ['contrast_ratio', 'vessel_clarity_index', 'illumination_uniformity', 
           'edge_preservation_index', 'microaneurysm_visibility', 'processing_time_ms']

for metric in metrics:
    print(f"\n{metric.upper()}:")
    desc_stats = df.groupby('method')[metric].agg(['mean', 'std', 'min', 'max']).round(6)
    print(desc_stats)

