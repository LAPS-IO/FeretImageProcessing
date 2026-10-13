#!/usr/bin/env python3
"""List every ``.jpg`` image under ``RAIZ/campanha/data/estacao/`` into a CSV.

For each ``RAIZ/<campanha>/<data>/`` one CSV is written to
``outputs/<campanha>/<data>/lista_imagens.csv`` with the image name and its full
path, covering all station folders of that date. Non-``.jpg`` files are ignored.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent
OUTPUT_CSV_NAME = "lista_imagens.csv"
IMAGE_EXTENSION = ".jpg"


def _subdirs(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir())


def list_date_images(date_dir: Path) -> List[Path]:
    """Return the ``.jpg`` files directly inside each station folder of ``date_dir``."""
    images: List[Path] = []
    for station_dir in _subdirs(date_dir):
        images.extend(
            sorted(
                p
                for p in station_dir.iterdir()
                if p.is_file() and p.suffix.lower() == IMAGE_EXTENSION
            )
        )
    return images


def write_image_list(images: List[Path], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "image_path"])
        for image in images:
            writer.writerow([image.name, str(image)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cria outputs/<campanha>/<data>/lista_imagens.csv com nome e caminho "
            "completo de todas as imagens .jpg em RAIZ/<campanha>/<data>/<estacao>/."
        )
    )
    parser.add_argument("root", type=Path, help="Pasta RAIZ com as campanhas.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs",
        help="Pasta de saída (padrão: ./outputs).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Erro: não é uma pasta: {root}", file=sys.stderr)
        sys.exit(1)

    for campaign_dir in _subdirs(root):
        for date_dir in _subdirs(campaign_dir):
            images = list_date_images(date_dir)
            if not images:
                print(f"[vazio] {date_dir}: nenhuma imagem {IMAGE_EXTENSION}")
                continue
            csv_path = args.output / campaign_dir.name / date_dir.name / OUTPUT_CSV_NAME
            write_image_list(images, csv_path)
            print(f"[ok]    {date_dir}: {len(images)} imagens → {csv_path}")


if __name__ == "__main__":
    main()
