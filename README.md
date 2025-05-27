# HRF Dataset Processing Pipeline

## 📋 Visão Geral

Pipeline científica para processamento e análise de imagens do dataset HRF (High-Resolution Fundus), especializada em extração do canal verde e aplicação de CLAHE (Contrast Limited Adaptive Histogram Equalization) seguindo práticas acadêmicas rigorosas.

## 🎯 Objetivos

- **Processamento**: Extração do canal verde + aplicação de CLAHE
- **Análise**: Estatísticas científicas rigorosas
- **Visualização**: Gráficos prontos para publicação
- **Documentação**: Relatórios detalhados e reprodutíveis

## 🏗️ Arquitetura Modular

```
hrf_pipeline/
├── image_processor.py    # Processamento de imagens
├── visualizer.py        # Visualizações científicas
├── statistics.py        # Análise estatística
├── main.py             # Pipeline principal
├── requirements.txt    # Dependências
└── README.md          # Documentação
```

## 🚀 Instalação

```bash
# Clonar repositório
git clone <repo_url>
cd hrf_pipeline

# Instalar dependências
pip install -r requirements.txt
```

## 💻 Uso

### Uso Básico

```bash
python main.py --dataset_path ./hrf_images --output_path ./results
```

### Uso Avançado

```bash
python main.py \
  --dataset_path /data/hrf \
  --output_path /results \
  --clahe_clip 3.0 \
  --clahe_grid 10 10 \
  --verbose
```

### Uso Programático

```python
from image_processor import HRFImageProcessor
from visualizer import HRFVisualizer
from statistics import HRFStatisticalAnalyzer

# Processar imagem única
processor = HRFImageProcessor()
green_original, green_clahe = processor.process_image("image.jpg")

# Análise estatística
analyzer = HRFStatisticalAnalyzer()
metrics = analyzer.calculate_image_quality_metrics(green_original, green_clahe)

# Visualizações
visualizer = HRFVisualizer()
fig = visualizer.plot_image_comparison(green_original, green_clahe)
```

## 📊 Saídas Geradas

### Estrutura de Resultados

```
results/
├── processed_images/         # Imagens processadas (.png)
├── visualizations/          # Gráficos científicos
│   ├── image_comparison.png
│   ├── histogram_analysis.png
│   ├── processing_statistics.png
│   ├── roi_analysis.png
│   └── publication_figure.png
└── reports/                # Relatórios detalhados
    ├── executive_summary.txt
    ├── processing_report.txt
    ├── statistical_report.txt
    ├── effectiveness_report.txt
    └── complete_analysis.json
```

### Métricas Calculadas

#### Qualidade de Imagem

- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error
- **SSIM**: Structural Similarity Index (aproximação)
- **Entropia**: Medida de informação da imagem

#### Estatísticas Descritivas

- Média, mediana, desvio padrão
- Quartis e percentis
- Assimetria e curtose
- Coeficiente de variação

#### Testes Estatísticos

- **Normalidade**: Shapiro-Wilk, D'Agostino-Pearson, Kolmogorov-Smirnov
- **Comparativos**: ANOVA, Kruskal-Wallis, Mann-Whitney U

## 🔬 Metodologia Científica

### Processamento CLAHE

- **Parâmetros otimizados**: clip_limit=2.0, tile_grid=8x8
- **Base científica**: Zuiderveld (1994), Fraz et al. (2012)
- **Validação**: Métricas de qualidade automáticas

### Análise Estatística

- **Significância**: α = 0.05 (configurável)
- **Testes robustos**: Paramétricos e não-paramétricos
- **Correção múltipla**: Aplicada quando apropriado

### Visualizações

- **Padrão científico**: Seaborn + Matplotlib
- **Pronto para publicação**: DPI=300, formatação acadêmica
- **Reprodutível**: Seeds fixas, parâmetros documentados

## 📈 Exemplos de Resultados

### Métricas Típicas para Imagens HRF

- **PSNR**: 25-35 dB (processamento efetivo)
- **Contraste**: 1.2-1.8x de melhoria
- **Entropia**: +0.5-2.0 bits de ganho

### Interpretação Automática

```
EFETIVIDADE: Alta (PSNR > 30 dB)
CONSISTÊNCIA: Boa (CV < 0.2)
RECOMENDAÇÃO: Manter parâmetros atuais
```

## 🛠️ Parâmetros Configuráveis

| Parâmetro            | Padrão | Descrição                 |
| -------------------- | ------ | ------------------------- |
| `clahe_clip_limit`   | 2.0    | Limite de clipping CLAHE  |
| `clahe_tile_grid`    | (8,8)  | Grade de tiles CLAHE      |
| `significance_level` | 0.05   | Nível α para testes       |
| `figsize`            | (12,8) | Tamanho das figuras       |
| `dpi`                | 300    | Resolução para publicação |

## 🔍 Validação e Qualidade

### Validação de Entrada

- Verificação de formato de imagem
- Validação de dimensões mínimas
- Detecção de arquivos corrompidos

### Controle de Qualidade

- Logs detalhados de processamento
- Tratamento robusto de erros
- Métricas de validação automáticas

### Testes Unitários

```bash
# Executar testes (quando disponíveis)
python -m pytest tests/
```

## 📚 Referências Científicas

1. **Zuiderveld, K.** (1994). Contrast limited adaptive histogram equalization. _Graphics gems IV_, 474-485.

2. **Fraz, M. M., et al.** (2012). An approach to localize the retinal blood vessels using bit planes and centerline detection. _Computer methods and programs in biomedicine_, 108(2), 600-616.

3. **Wang, Z., et al.** (2004). Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_, 13(4), 600-612.

## 🤝 Contribuição

### Estrutura de Código

- **Clean Code**: Funções pequenas, responsabilidade única
- **DRY Principle**: Evita repetição de código
- **Fail Fast**: Validação antecipada de parâmetros
- **Logging**: Rastreabilidade completa

### Padrões de Commit

```
feat: adicionar nova métrica de qualidade
fix: corrigir cálculo de entropia
docs: atualizar exemplos de uso
test: adicionar testes para visualizador
```

## 📄 Licença

Este projeto segue padrões acadêmicos abertos para pesquisa em imagens médicas.

## 🆘 Suporte

Para dúvidas técnicas ou sugestões:

1. Verificar logs em `hrf_processing.log`
2. Consultar documentação inline
3. Revisar relatórios gerados automaticamente

---

**Desenvolvido por Atlas de Conhecimento**
_Pipeline científica para análise rigorosa de imagens oculares HRF_
