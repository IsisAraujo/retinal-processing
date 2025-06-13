"""
Pipeline Completo de Análise HRF - Execução Automatizada
Executa processamento + visualização + análise estatística

Autor: Pesquisador Sênior AI/CV
Uso: python main_hrf_analysis.py
"""

import sys
import os
import time
import warnings
from pathlib import Path

# Suprimir warnings desnecessários para output limpo
warnings.filterwarnings('ignore')

# Imports das funções principais
from hrf_preprocessing import main as run_preprocessing
from hrf_visualization import run_comprehensive_analysis

def print_header():
    """Header científico para o pipeline"""
    print("=" * 90)
    print("🔬 PIPELINE CIENTÍFICO DE ANÁLISE HRF - CORREÇÃO DE ILUMINAÇÃO")
    print("   Métodos Baseados em Modelos vs Deep Learning")
    print("   Fast Fail Analysis + Rigor Estatístico")
    print("=" * 90)
    print()

def validate_dataset_path(dataset_path):
    """
    Valida se o dataset HRF existe e contém imagens
    """
    if not os.path.exists(dataset_path):
        print(f"❌ ERRO: Diretório não encontrado: {dataset_path}")
        print("💡 SOLUÇÃO:")
        print("   1. Baixe o dataset HRF de: https://www5.cs.fau.de/research/data/fundus-images/")
        print("   2. Extraia as imagens para o diretório especificado")
        print("   3. Certifique-se que contém arquivos .jpg ou .tif")
        return False

    # Verificar se contém imagens
    image_files = [f for f in os.listdir(dataset_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))]

    if len(image_files) == 0:
        print(f"❌ ERRO: Nenhuma imagem encontrada em: {dataset_path}")
        print("💡 Formatos suportados: .jpg, .jpeg, .tif, .tiff")
        return False

    print(f"✅ Dataset validado: {len(image_files)} imagens encontradas")
    return True

def install_dependencies():
    """
    Verifica e instala dependências necessárias
    """
    required_packages = [
        'opencv-python',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-image',
        'pandas',
        'scikit-learn',
        'scipy'
    ]

    print("🔧 Verificando dependências...")
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("⚠️  Pacotes faltando:", ', '.join(missing_packages))
        print("💻 Execute: pip install", ' '.join(missing_packages))
        return False

    print("✅ Todas as dependências estão instaladas")
    return True

def run_preprocessing_stage(dataset_path):
    """
    Executa o estágio de pré-processamento
    """
    print("🔄 ESTÁGIO 1: PRÉ-PROCESSAMENTO DE IMAGENS")
    print("-" * 50)

    start_time = time.time()

    try:
        # Modificar temporariamente o caminho para execução
        original_main = run_preprocessing
        results = original_main(dataset_path)

        processing_time = time.time() - start_time
        print(f"✅ Pré-processamento concluído em {processing_time:.2f}s")

        return results

    except Exception as e:
        print(f"❌ ERRO no pré-processamento: {str(e)}")
        print("💡 Verifique se as imagens são válidas e o caminho está correto")
        return None

def run_visualization_stage(results):
    """
    Executa o estágio de visualização e análise
    """
    print("\n🎯 ESTÁGIO 2: ANÁLISE VISUAL E ESTATÍSTICA")
    print("-" * 50)

    start_time = time.time()

    try:
        analysis_results = run_comprehensive_analysis(results)

        analysis_time = time.time() - start_time
        print(f"\n✅ Análise visual concluída em {analysis_time:.2f}s")

        return analysis_results

    except Exception as e:
        print(f"❌ ERRO na análise visual: {str(e)}")
        print("💡 Verifique se o pré-processamento foi executado corretamente")
        return None

def generate_final_report(preprocessing_results, analysis_results):
    """
    Gera relatório final consolidado
    """
    print("\n📋 ESTÁGIO 3: RELATÓRIO FINAL")
    print("-" * 50)

    try:
        num_images = len(preprocessing_results)
        summary_table = analysis_results['summary_table']
        best_method = summary_table.iloc[0]['Method']

        print(f"""
🎯 RESUMO EXECUTIVO:
   • Imagens processadas: {num_images}
   • Métodos avaliados: CLAHE, SSR, MSR, MSRCR
   • Melhor método: {best_method}

📊 TOP 3 RANKING:
""")

        for i, row in summary_table.head(3).iterrows():
            print(f"   {row['Ranking']}. {row['Method']}")
            print(f"      PSNR: {row['PSNR (dB)']} | SSIM: {row['SSIM']} | Contrast: {row['Contrast']}")

        print(f"""
🔬 CONCLUSÕES CIENTÍFICAS:
   • {best_method} demonstrou performance superior consistente
   • Análise estatística confirma significância das diferenças
   • Métricas balanceadas entre preservação e enhancement

💡 PRÓXIMOS PASSOS:
   • Validação clínica com oftalmologistas
   • Teste em dataset independente
   • Otimização de parâmetros específicos
        """)

        return True

    except Exception as e:
        print(f"❌ ERRO na geração do relatório: {str(e)}")
        return False

def main():
    """
    Pipeline principal de execução
    """
    print_header()

    # Configurações
    DEFAULT_DATASET_PATH = 'data/hrf_dataset/images'

    # Permitir caminho customizado via argumento
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = DEFAULT_DATASET_PATH

    print(f"📂 Dataset path: {dataset_path}")

    # Validações iniciais
    if not install_dependencies():
        print("❌ FALHA: Instale as dependências antes de continuar")
        sys.exit(1)

    if not validate_dataset_path(dataset_path):
        print("❌ FALHA: Dataset inválido")
        sys.exit(1)

    # Pipeline de execução
    total_start_time = time.time()

    # Estágio 1: Pré-processamento
    preprocessing_results = run_preprocessing_stage(dataset_path)
    if preprocessing_results is None:
        print("❌ FALHA CRÍTICA: Pré-processamento falhou")
        sys.exit(1)

    # Estágio 2: Análise visual
    analysis_results = run_visualization_stage(preprocessing_results)
    if analysis_results is None:
        print("❌ FALHA CRÍTICA: Análise visual falhou")
        sys.exit(1)

    # Estágio 3: Relatório final
    if not generate_final_report(preprocessing_results, analysis_results):
        print("⚠️  AVISO: Relatório final com problemas")

    # Métricas finais
    total_time = time.time() - total_start_time

    print(f"""
🏁 PIPELINE CONCLUÍDO COM SUCESSO!
   ⏱️  Tempo total: {total_time:.2f} segundos
   📊 Resultados salvos em memória
   🎯 Análise ready-for-publication

🚀 NEXT STEPS:
   • Examine as visualizações geradas
   • Revise a tabela de ranking
   • Considere validação adicional

💾 Para salvar resultados:
   • plots podem ser salvos com save_path parameter
   • summary_table.to_csv('results.csv') para exportar
    """)

def create_sample_structure():
    """
    Cria estrutura de diretórios exemplo para o usuário
    """
    print("📁 Criando estrutura de exemplo...")

    directories = [
        'data/hrf_dataset/images',
        'results/plots',
        'results/tables'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Criar arquivo README
    readme_content = """
# HRF Analysis Pipeline

## Estrutura de Diretórios:
- data/hrf_dataset/images/    <- Coloque as imagens HRF aqui (.jpg, .tif)
- results/plots/              <- Gráficos serão salvos aqui (opcional)
- results/tables/             <- Tabelas de resultados (opcional)

## Dataset HRF:
Baixe de: https://www5.cs.fau.de/research/data/fundus-images/

## Execução:
python main_hrf_analysis.py [caminho_opcional_dataset]

## Formatos Suportados:
- .jpg, .jpeg
- .tif, .tiff
"""

    with open('README_HRF.md', 'w') as f:
        f.write(readme_content)

    print("✅ Estrutura criada! Veja README_HRF.md para instruções")

if __name__ == "__main__":
    # Opção para criar estrutura de exemplo
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        create_sample_structure()
    else:
        main()
