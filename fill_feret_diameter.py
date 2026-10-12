#!/usr/bin/env python3
"""Fill missing ``diameter_px`` / ``diameter_um`` in ``atributos_por_roi.csv``.

Walks every run folder under ``outputs/`` (e.g. ``outputs/campanha/dia/``) and,
for each ``atributos_por_roi.csv`` with empty diameter cells, looks the values up
in the sibling ``feret_diameters.csv`` by source file name + ``left``/``top``.
Only empty cells are written; existing values are never overwritten.

Older ``feret_diameters.csv`` files only have a ``diameter`` column (pixels); in
that case it is used as ``diameter_px`` and ``diameter_um`` is derived with the
run's ``um_per_pixel`` (``run_metadata.txt``) or ``--um-per-pixel``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent
ATRIBUTOS_CSV_NAME = "atributos_por_roi.csv"
FERET_CSV_NAME = "feret_diameters.csv"
METADATA_NAME = "run_metadata.txt"
DEFAULT_UM_PER_PIXEL = 13.8
DIAMETER_COLUMNS = ("diameter_px", "diameter_um")

FeretKey = Tuple[str, int, int]


def _optional_float(value: object) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_missing(value: object) -> bool:
    return value is None or str(value).strip() == ""


def read_um_per_pixel(run_dir: Path) -> Optional[float]:
    """Return ``um_per_pixel`` from ``run_metadata.txt`` (``key : value`` or ``key = value``)."""
    path = run_dir / METADATA_NAME
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (":", "="):
            if sep in line:
                key, _, value = line.partition(sep)
                if key.strip() == "um_per_pixel":
                    return _optional_float(value.strip())
                break
    return None


def load_feret_diameters(
    csv_path: Path, um_per_pixel: float
) -> Dict[FeretKey, Tuple[Optional[float], Optional[float]]]:
    """Map ``(stem, left, top)`` → ``(diameter_px, diameter_um)``."""
    index: Dict[FeretKey, Tuple[Optional[float], Optional[float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        px_col = "diameter_px" if "diameter_px" in fields else (
            "diameter" if "diameter" in fields else None
        )
        um_col = "diameter_um" if "diameter_um" in fields else None
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            try:
                left = int(float(row["left"]))
                top = int(float(row["top"]))
            except (KeyError, ValueError, TypeError):
                continue
            diam_px = _optional_float(row.get(px_col)) if px_col else None
            diam_um = _optional_float(row.get(um_col)) if um_col else None
            if diam_um is None and diam_px is not None:
                diam_um = diam_px * um_per_pixel
            index[(Path(file_name).stem, left, top)] = (diam_px, diam_um)
    return index


def discover_run_dirs(outputs: Path) -> List[Path]:
    return sorted(p.parent for p in outputs.rglob(ATRIBUTOS_CSV_NAME) if p.is_file())


def fill_run(run_dir: Path, default_um_per_pixel: float, dry_run: bool) -> None:
    atributos_path = run_dir / ATRIBUTOS_CSV_NAME
    feret_path = run_dir / FERET_CSV_NAME

    with open(atributos_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in DIAMETER_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    n_missing = sum(
        1 for row in rows if any(_is_missing(row.get(c)) for c in DIAMETER_COLUMNS)
    )
    if n_missing == 0:
        print(f"[ok]      {run_dir}: diâmetros completos ({len(rows)} ROIs)")
        return
    if not feret_path.is_file():
        print(
            f"[aviso]   {run_dir}: {n_missing} ROIs sem diâmetro, "
            f"mas {FERET_CSV_NAME} não existe"
        )
        return

    um_per_pixel = read_um_per_pixel(run_dir) or default_um_per_pixel
    index = load_feret_diameters(feret_path, um_per_pixel)

    n_filled = 0
    n_unmatched = 0
    for row in rows:
        if not any(_is_missing(row.get(c)) for c in DIAMETER_COLUMNS):
            continue
        try:
            key = (
                (row.get("source_stem") or "").strip(),
                int(float(row["left"])),
                int(float(row["top"])),
            )
        except (KeyError, ValueError, TypeError):
            n_unmatched += 1
            continue
        diam_px, diam_um = index.get(key, (None, None))
        filled = False
        if _is_missing(row.get("diameter_px")) and diam_px is not None:
            row["diameter_px"] = f"{diam_px:.6f}"
            filled = True
        if _is_missing(row.get("diameter_um")) and diam_um is not None:
            row["diameter_um"] = f"{diam_um:.6f}"
            filled = True
        if filled:
            n_filled += 1
        else:
            n_unmatched += 1

    prefix = "[dry-run]" if dry_run else "[fill]   "
    print(
        f"{prefix} {run_dir}: {n_missing} ROIs sem diâmetro → "
        f"{n_filled} preenchidas, {n_unmatched} sem correspondência "
        f"(um_per_pixel={um_per_pixel})"
    )
    if dry_run or n_filled == 0:
        return

    tmp_path = atributos_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, atributos_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preenche diameter_px/diameter_um faltantes em atributos_por_roi.csv "
            "usando feret_diameters.csv (match por arquivo + left/top), "
            "para todas as runs em outputs/."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPO / "outputs",
        help="Pasta outputs/ ou pasta de uma run (padrão: ./outputs).",
    )
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=DEFAULT_UM_PER_PIXEL,
        help=(
            "µm/px usado para calcular diameter_um quando o feret_diameters.csv "
            "não tem essa coluna e run_metadata.txt não informa "
            f"(padrão: {DEFAULT_UM_PER_PIXEL})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra o que seria preenchido, sem alterar arquivos.",
    )
    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"Erro: não é uma pasta: {args.path}", file=sys.stderr)
        sys.exit(1)
    run_dirs = discover_run_dirs(args.path.resolve())
    if not run_dirs:
        print(f"Nenhum {ATRIBUTOS_CSV_NAME} encontrado em {args.path}")
        return
    for run_dir in run_dirs:
        try:
            fill_run(run_dir, args.um_per_pixel, args.dry_run)
        except Exception as e:
            print(f"[erro]    {run_dir}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
