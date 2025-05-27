from pathlib import Path
import numpy as np  # Adicionar importação do numpy
from config import HRFConfig
from processor import RetinalProcessor
from visualizer import visualize_results

def main():
    try:
        config = HRFConfig()
        config.validate()

        processor = RetinalProcessor(config)
        results = processor.process_batch()

        # Processamento de exemplo para visualização
        sample = next(iter(config.paths['input'].glob('*')), None)
        if sample:
            green, processed = processor.process_image(sample)
            visualize_results(
                green,
                processed,
                sample.name,
                config.paths['visualizations']  # Novo argumento
            )

        print(f"Processamento completo. Sucessos: {results['sucessos']}, Falhas: {results['falhas']}")
        print(f"Tempo médio por imagem: {np.mean(results['tempos']):.3f}s")

    except Exception as e:
        print(f"Falha crítica: {e}")
        raise

if __name__ == "__main__":
    main()
