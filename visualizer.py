import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
import cv2

def visualize_results(original: np.ndarray, processed: np.ndarray,
                     filename: str, output_dir: Path):
    """Visualização científica com histogramas e métricas"""
    output_path = output_dir / f"{Path(filename).stem}_analysis.png"

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Imagens principais
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])

    ax1.imshow(original, cmap='gray')
    ax1.set_title('Canal Verde Original', fontsize=12)
    ax1.axis('off')

    ax2.imshow(processed, cmap='gray')
    ax2.set_title('Após CLAHE Adaptativo', fontsize=12)
    ax2.axis('off')

    # Diferença
    ax3 = fig.add_subplot(gs[0:2, 2])
    diff = cv2.absdiff(processed, original)
    im = ax3.imshow(diff, cmap='hot')
    ax3.set_title('Mapa de Diferenças', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # Histogramas
    ax4 = fig.add_subplot(gs[2, :])
    hist_orig, bins = np.histogram(original.flatten(), 256, [0, 256])
    hist_proc, _ = np.histogram(processed.flatten(), 256, [0, 256])

    ax4.plot(hist_orig, 'b-', alpha=0.7, label='Original')
    ax4.plot(hist_proc, 'r-', alpha=0.7, label='CLAHE')
    ax4.set_xlabel('Intensidade')
    ax4.set_ylabel('Frequência')
    ax4.set_title('Distribuição de Intensidades')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Métricas no título
    contrast_gain = np.std(processed) / np.std(original)
    mean_shift = np.mean(processed) - np.mean(original)

    fig.suptitle(f'{filename} - Ganho de Contraste: {contrast_gain:.2f}x | '
                 f'Deslocamento Médio: {mean_shift:.1f}', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {'contrast_gain': contrast_gain, 'mean_shift': mean_shift}

def plot_batch_metrics(metrics: List[Dict], output_dir: Path):
    """Gera gráficos agregados das métricas do batch"""
    if not metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análise Agregada do Processamento CLAHE', fontsize=16)

    # Extrair dados
    contraste = [m['ganho_contraste'] for m in metrics]
    entropia = [m['ganho_entropia'] for m in metrics]
    nitidez = [m['ganho_nitidez'] for m in metrics]
    psnr = [m['psnr'] for m in metrics]

    # Histograma de ganho de contraste
    ax = axes[0, 0]
    ax.hist(contraste, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', label='Sem ganho')
    ax.set_xlabel('Ganho de Contraste')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição do Ganho de Contraste')
    ax.legend()

    # Scatter plot entropia vs contraste
    ax = axes[0, 1]
    scatter = ax.scatter(contraste, entropia, c=psnr, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Ganho de Contraste')
    ax.set_ylabel('Ganho de Entropia (bits)')
    ax.set_title('Relação Contraste-Entropia')
    plt.colorbar(scatter, ax=ax, label='PSNR (dB)')

    # Box plot das métricas
    ax = axes[1, 0]
    data = [contraste, nitidez, [p/40 for p in psnr]]  # Normalizar PSNR
    ax.boxplot(data, labels=['Contraste', 'Nitidez', 'PSNR/40'])
    ax.set_ylabel('Valor Normalizado')
    ax.set_title('Distribuição das Métricas')
    ax.grid(True, alpha=0.3)

    # Estatísticas resumidas
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    ESTATÍSTICAS RESUMIDAS:

    Ganho de Contraste:
    • Média: {np.mean(contraste):.2f}
    • Mediana: {np.median(contraste):.2f}
    • Desvio: {np.std(contraste):.2f}

    Ganho de Entropia:
    • Média: {np.mean(entropia):.3f} bits
    • Casos positivos: {sum(1 for e in entropia if e > 0)}/{len(entropia)}

    PSNR:
    • Média: {np.mean(psnr):.1f} dB
    • Mínimo: {np.min(psnr):.1f} dB

    Casos problemáticos:
    • Contraste < 1.0: {sum(1 for c in contraste if c < 1.0)}
    • PSNR < 30 dB: {sum(1 for p in psnr if p < 30)}
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_dir / 'batch_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
