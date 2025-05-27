import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_results(original: np.ndarray, processed: np.ndarray, filename: str, output_dir: Path):
    output_path = output_dir / f"{Path(filename).stem}_processing_result.png"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Processamento Retinal - {filename}", fontsize=14)

    for a, img, label in zip(ax, [original, processed], ['Original', 'Processado']):
        a.imshow(img, cmap='gray')
        a.set_title(label)
        a.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
