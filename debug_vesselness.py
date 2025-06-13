import cv2
import numpy as np
from pathlib import Path
from hrf_metrics import MetricsCalculator
from hrf_utils import normalize_image_for_display # Importa a nova função
from typing import List

def debug_vesselness_map(image_path: str, output_dir: str = "debug_output",
                         scales: List[int] = None, beta: float = None, c: float = None):
    """
    Loads an image, calculates its Frangi vesselness map, and saves it.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    print(f"Processando imagem: {image_path.name}")

    # Get the raw float vesselness map, passing Frangi parameters and debug_output_dir
    raw_vesselness_map = MetricsCalculator._get_vesselness_map(img, scales=scales, beta=beta, c=c, debug_output_dir=output_dir)

    # Save the raw float vesselness map (e.g., as a .npy file for later analysis)
    raw_output_filename = output_dir / f"vesselness_map_raw_{image_path.stem}.npy"
    np.save(str(raw_output_filename), raw_vesselness_map)
    print(f"Mapa de vesselness bruto salvo em: {raw_output_filename}")

    # Normalize for display (0-255, CV_8U)
    display_vesselness_map = normalize_image_for_display(raw_vesselness_map)

    # Save the normalized map for display
    display_output_filename = output_dir / f"vesselness_map_display_{image_path.stem}.png"
    cv2.imwrite(str(display_output_filename), display_vesselness_map)
    print(f"Mapa de vesselness normalizado para display salvo em: {display_output_filename}")

    # Optional: Apply a more aggressive contrast stretch for better visualization
    # Ajuste alpha e beta conforme necessário para realçar os vasos
    # alpha > 1.0 aumenta o contraste, beta ajusta o brilho
    aggressive_display_vesselness_map = normalize_image_for_display(raw_vesselness_map, alpha=3.0, beta=0.0)
    aggressive_output_filename = output_dir / f"vesselness_map_aggressive_display_{image_path.stem}.png"
    cv2.imwrite(str(aggressive_output_filename), aggressive_display_vesselness_map)
    print(f"Mapa de vesselness com contraste agressivo salvo em: {aggressive_output_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug Frangi vesselness map generation."
    )
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument("--output_dir", default="debug_output", help="Directory to save the vesselness map.")
    parser.add_argument("--scales", nargs="+", type=int, help="List of scales for Frangi filter (e.g., 1 2 3 4)")
    parser.add_argument("--beta", type=float, help="Beta parameter for Frangi filter")
    parser.add_argument("--c", type=float, help="C parameter for Frangi filter")

    args = parser.parse_args()

    debug_vesselness_map(args.image_path, args.output_dir, args.scales, args.beta, args.c)


