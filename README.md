# HRF Illumination Correction Analysis - Referências e Citações

## Versão Aprimorada do Projeto

Este projeto implementa uma análise comparativa rigorosa de métodos de correção de iluminação em imagens de fundo de olho, seguindo padrões acadêmicos atuais (2024-2025).

## Principais Melhorias Implementadas

### 1. Correções Metodológicas Críticas

- **Correção FDR Benjamini-Hochberg** em lugar da correção Bonferroni excessivamente conservadora
- **Parâmetros do filtro de Frangi corrigidos** (c=15 em vez de 500, seguindo literatura padrão)
- **Métricas padrão IEEE adicionadas** (PSNR, SSIM)
- **Análise de poder estatístico** implementada

### 2. Conformidade Acadêmica

- Referências reais da literatura utilizadas (não inventadas)
- Documentação completa de parâmetros com citações
- Análise estatística rigorosa seguindo padrões atuais
- Visualizações aprimoradas para publicação

## Referências Bibliográficas Utilizadas

### Métricas de Qualidade de Imagem (Padrão IEEE)

1. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004)**
   - _Image quality assessment: From error visibility to structural similarity_
   - IEEE Transactions on Image Processing, 13(4), 600-612
   - **Aplicação**: Implementação das métricas PSNR e SSIM
   - **Link**: https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
   - **DOI**: 10.1109/TIP.2003.819861

### Correção Estatística para Múltiplos Testes

2. **Benjamini, Y., & Hochberg, Y. (1995)**
   - _Controlling the false discovery rate: a practical and powerful approach to multiple testing_
   - Journal of the Royal Statistical Society: Series B (Methodological), 57(1), 289-300
   - **Aplicação**: Implementação da correção FDR em substituição ao Bonferroni
   - **Link**: https://www.jstor.org/stable/2346101
   - **DOI**: 10.1111/j.2517-6161.1995.tb02031.x

### Filtro de Frangi para Segmentação Vascular

3. **Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998)**
   - _Multiscale vessel enhancement filtering_
   - MICCAI 1998, Lecture Notes in Computer Science, vol 1496
   - **Aplicação**: Parâmetros corretos para o filtro de detecção vascular (c=15)
   - **Link**: https://link.springer.com/chapter/10.1007/BFb0056195
   - **DOI**: 10.1007/BFb0056195

### Métodos CLAHE para Imagens de Retina

4. **Alwazzan, M. J., Ismael, M. A., & Ahmed, A. N. (2021)**
   - _A hybrid algorithm to enhance colour retinal fundus images using a wiener filter and clahe_
   - Journal of Digital Imaging, 34(3), 750-759
   - **Aplicação**: Parâmetros otimizados para CLAHE (clip_limit=3.0, tile_grid=(8,8))
   - **Link**: https://pubmed.ncbi.nlm.nih.gov/34291375/
   - **DOI**: 10.1007/s10278-021-00567-8

### Implementações OpenCV

5. **OpenCV Development Team**

   - _OpenCV: Histograms - 2: Histogram Equalization_
   - **Aplicação**: Implementação de PSNR e SSIM
   - **Link**: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

6. **PyImageSearch - Adrian Rosebrock**
   - _OpenCV Histogram Equalization and Adaptive Histogram Equalization (CLAHE)_
   - **Aplicação**: Implementação e validação de parâmetros CLAHE
   - **Link**: https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/

### Bibliotecas Python para Análise Estatística

7. **SciPy Development Team**

   - _SciPy.stats.false_discovery_control_
   - **Aplicação**: Implementação da correção FDR
   - **Link**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html

8. **Scikit-Image Development Team**
   - _skimage.metrics - Peak Signal-to-Noise Ratio and Structural Similarity_
   - **Aplicação**: Implementação de PSNR e SSIM
   - **Link**: https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html

### Análise de Poder Estatístico

9. **Cohen, J. (1988)**
   - _Statistical power analysis for the behavioral sciences_
   - Lawrence Erlbaum Associates
   - **Aplicação**: Cálculo de tamanho de efeito e poder estatístico
   - **ISBN**: 0-8058-0283-5

### Datasets e Benchmarks

10. **Papers with Code - HRF Dataset**
    - _High-Resolution Fundus Image Database_
    - **Aplicação**: Validação do dataset utilizado
    - **Link**: https://paperswithcode.com/dataset/hrf

### Diretrizes para Análise de Imagens Médicas

11. **Editage Insights (2024)**
    - _Statistical approaches for analyzing imaging data: An overview_
    - **Aplicação**: Diretrizes para análise estatística em imagens médicas
    - **Link**: https://www.editage.com/insights/statistical-approaches-for-analyzing-imaging-data-an-overview

## Implementações de Referência

### Repositórios GitHub Consultados

12. **aizvorski/video-quality**

    - _Video quality metrics, reference implementation in python: VIF, SSIM, PSNR_
    - **Link**: https://github.com/aizvorski/video-quality
    - **Aplicação**: Validação de implementações PSNR/SSIM

13. **dongb5/Retinex**
    - _Python implementation of multi scale retinex with color restoration_
    - **Link**: https://github.com/dongb5/Retinex
    - **Aplicação**: Validação de parâmetros MSR/MSRCR

## Artigos Relacionados ao Projeto Original

### Referências do Artigo Base

14. **Tian et al. (2021)**

    - _Blood Vessel Segmentation of Fundus Retinal Images Based on Improved Frangi and Mathematical Morphology_
    - Journal: Computational and Mathematical Methods in Medicine
    - **DOI**: 10.1155/2021/4761517

15. **Yang et al. (2020)**

    - _Frangi based multi-scale level sets for retinal vascular segmentation_
    - Journal: Computer Methods and Programs in Biomedicine
    - **DOI**: 10.1016/j.cmpb.2020.105752

16. **Mahapatra et al. (2022)**
    - _A novel framework for retinal vessel segmentation using optimal improved frangi filter_
    - Journal: Computers in Biology and Medicine
    - **DOI**: 10.1016/j.compbiomed.2022.105770

## Tutoriais e Documentações Técnicas

### PSNR e SSIM

17. **GeeksforGeeks**
    - _Python | Peak Signal-to-Noise Ratio (PSNR)_
    - **Link**: https://www.geeksforgeeks.org/python/python-peak-signal-to-noise-ratio-psnr/
    - **Data**: Janeiro 2020

### Correção FDR

18. **R-bloggers**

    - _The Benjamini-Hochberg procedure (FDR) and P-Value Adjusted Explained_
    - **Link**: https://www.r-bloggers.com/2023/07/the-benjamini-hochberg-procedure-fdr-and-p-value-adjusted-explained/
    - **Data**: Julho 2023

19. **Statistics How To** - _Benjamini-Hochberg Procedure_ - **Link**: https://www.statisticshowto.com/benjamini-hochberg-procedure/ - **Data**: Outubro 2024

        "retinal components are more intense in the fundus image's green channel"

    Fonte: Interactive Blood Vessel Segmentation from Retinal Fundus Image Based on Canny Edge Detector (2021). PMC8512020
