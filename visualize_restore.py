"""Zapis odrestaurowanego / powiększonego pojedynczego zdjęcia.

Użycie:
    python visualize_restore.py <obraz>
    python visualize_restore.py <obraz> --output img/restored.png
"""
import argparse

import cv2

from config import ARTIFACTS
from pipeline import restore, quality
from pipeline.detect_align import imread


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _sharpness(bgr):
    return quality.sharpness(_rgb(bgr))


def main():
    parser = argparse.ArgumentParser(
        description="Zapisuje gotowy obraz po pipeline.restore.restore()."
    )
    parser.add_argument("image", help="sciezka do obrazu wejsciowego")
    parser.add_argument(
        "--output",
        default=str(ARTIFACTS / "restored.png"),
        help="sciezka pliku wynikowego",
    )
    parser.add_argument(
        "--backend",
        default="baseline",
        choices=["baseline", "gfpgan"],
        help="backend restauracji z pipeline.restore",
    )
    args = parser.parse_args()

    restore.set_backend(args.backend)
    original = imread(args.image)
    restored = restore.restore(original)

    if not cv2.imwrite(args.output, restored):
        raise SystemExit(f"Nie udalo sie zapisac pliku: {args.output}")

    print(f"Zapisano odrestaurowany obraz: {args.output}")
    print(f"Oryginal:       {original.shape[1]}x{original.shape[0]} px, sharp={_sharpness(original):.2f}")
    print(f"Po restauracji: {restored.shape[1]}x{restored.shape[0]} px, sharp={_sharpness(restored):.2f}")


if __name__ == "__main__":
    main()
