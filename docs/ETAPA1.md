# Processamento de Imagens Retinianas - Etapa 1

## Objetivo

Implementação inicial de um pipeline científico para processamento de imagens médicas retinianas, com ênfase em melhoramento de contraste e análise quantitativa de resultados.

---

## Funcionalidades Implementadas

### 1. **Pré-processamento Automatizado**

- Extração do canal verde para realce vascular
- Aplicação de CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Suporte híbrido CPU/GPU via CuPy (fallback automático)

### 2. **Análise Quantitativa**

- Métricas de qualidade calculadas por imagem:
  - Razão de contraste (σ_processado/σ_original)
  - PSNR (Peak Signal-to-Noise Ratio)
  - Variação média de intensidade
  - Ganho de entropia (placeholder para implementação futura)

### 3. **Gestão Científica de Dados**

```python
# Estrutura de diretórios
.
├── data/images/      # Imagens originais (HRF, DRIVE, etc.)
├── processed/        # Imagens processadas (CLAHE aplicado)
├── visualizations/   # Comparações originais/processadas
└── results/          # Dados quantitativos (JSON/Métricas)

```
