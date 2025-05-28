from pathlib import Path
import numpy as np
from config import HRFConfig
from processor import RetinalProcessor
from visualizer import visualize_results, plot_batch_metrics

def main():
    try:
        config = HRFConfig()
        config.validate()

        processor = RetinalProcessor(config)

        # Processamento paralelo
        print("Iniciando processamento em lote...")
        results = processor.process_batch(max_workers=4)

        # Análise de uma amostra
        sample = next(config.paths['input'].glob('*.jpg'), None)
        if sample:
            print(f"\nAnalisando amostra: {sample.name}")
            green, processed = processor.process_image(sample)
            metrics = visualize_results(
                green,
                processed,
                sample.name,
                config.paths['visualizations']
            )
            print(f"Métricas da amostra: {metrics}")

        # Relatório final
        print(f"\n{'='*50}")
        print(f"RELATÓRIO DE PROCESSAMENTO")
        print(f"{'='*50}")
        print(f"Total de imagens: {results['sucessos'] + results['falhas']}")
        print(f"Processadas com sucesso: {results['sucessos']}")
        print(f"Falhas: {results['falhas']}")

        if results['tempos']:
            print(f"\nTempo médio por imagem: {np.mean(results['tempos']):.3f}s")
            print(f"Tempo total: {sum(results['tempos']):.1f}s")

        if results['metricas']:
            # Estatísticas das métricas
            ganhos_contraste = [m['ganho_contraste'] for m in results['metricas']]
            ganhos_entropia = [m['ganho_entropia'] for m in results['metricas']]

            print(f"\nMÉTRICAS AGREGADAS:")
            print(f"Ganho médio de contraste: {np.mean(ganhos_contraste):.2f}x")
            print(f"Ganho médio de entropia: {np.mean(ganhos_entropia):.3f} bits")

            # Identificar casos problemáticos
            problemas = [m for m in results['metricas'] if m['ganho_contraste'] < 1.0]
            if problemas:
                print(f"\nAVISO: {len(problemas)} imagens com redução de contraste")

            # Gerar visualização agregada
            plot_batch_metrics(results['metricas'], config.paths['visualizations'])
            print("\nGráficos de análise agregada salvos em 'visualizations/batch_analysis.png'")

    except Exception as e:
        print(f"Erro crítico: {e}")
        raise

if __name__ == "__main__":
    main()
