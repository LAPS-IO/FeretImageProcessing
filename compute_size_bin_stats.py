#!/usr/bin/env python3
"""
Read ``feret_diameters.csv`` and count objects in diameter (µm) bins:

    <100 | [100, 500) | [500, 1000) | >=1000

Reports counts per image and per config folder (Config 01 / Config 03), with
folder totals normalized by the number of images in that folder.

Config membership is taken from subfolders next to the CSV (same run directory).
By default that is ``background_difference_watershed_side_by_side/<Config *>/*``;
use ``--config-from watershed`` for ``background_difference_watershed`` (``.npz``),
or ``--config-dir DIR`` for an arbitrary folder with the same layout.

Usage::

    python3 compute_size_bin_stats.py outputs/run20/feret_diameters.csv
    python3 compute_size_bin_stats.py outputs/run20/feret_diameters.csv --config-from watershed
    python3 compute_size_bin_stats.py outputs/run20/feret_diameters.csv -o outputs/run20
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

BIN_KEYS = ("lt_100", "100_500", "500_1000", "ge_1000")
BIN_LABELS = {
    "lt_100": "<100",
    "100_500": "[100, 500)",
    "500_1000": "[500, 1000)",
    "ge_1000": ">=1000",
}

# Preset subfolders of a run used to resolve config membership.
CONFIG_FROM_PRESETS = {
    "side_by_side": "background_difference_watershed_side_by_side",
    "watershed": "background_difference_watershed",
}

# Files accepted under each config subfolder (side-by-side is image; watershed is .npz).
CONFIG_FILE_SUFFIXES = {
    ".npz",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def diameter_bin(diameter_um: float) -> str:
    if diameter_um < 100:
        return "lt_100"
    if diameter_um < 500:
        return "100_500"
    if diameter_um < 1000:
        return "500_1000"
    return "ge_1000"


def empty_counts() -> dict[str, int]:
    return {k: 0 for k in BIN_KEYS}


def load_config_map(source_dir: Path) -> dict[str, str]:
    """Map Feret CSV ``file_name`` (``.npz`` basename) → config folder name.

    Scans ``source_dir/<config>/*`` for label ``.npz`` or preview images and keys
    each entry as ``stem.npz`` so it matches ``feret_diameters.csv``.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing config source dir: {source_dir}")

    mapping: dict[str, str] = {}
    for cfg_dir in sorted(
        p for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for path in sorted(cfg_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in CONFIG_FILE_SUFFIXES:
                continue
            key = path.with_suffix(".npz").name
            if key in mapping:
                raise ValueError(
                    f"Duplicate file stem across configs: {key} "
                    f"({mapping[key]} and {cfg_dir.name})"
                )
            mapping[key] = cfg_dir.name
    if not mapping:
        raise FileNotFoundError(f"No config files under {source_dir}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count Feret objects by diameter_um bins per image and per config."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path("outputs/run20/feret_diameters.csv"),
        help="Path to feret_diameters.csv (default: outputs/run20/feret_diameters.csv)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output CSVs (default: same directory as the input CSV)",
    )
    parser.add_argument(
        "--config-from",
        choices=sorted(CONFIG_FROM_PRESETS),
        default="side_by_side",
        help=(
            "Which run subfolder provides config membership "
            f"(default: side_by_side → {CONFIG_FROM_PRESETS['side_by_side']}; "
            f"watershed → {CONFIG_FROM_PRESETS['watershed']}). "
            "Ignored when --config-dir is set."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Explicit folder with <config>/* files used for config membership "
            "(overrides --config-from)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        print(f"Not a file: {csv_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = csv_path.parent
    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config_dir is not None:
        config_source = args.config_dir.resolve()
    else:
        config_source = (run_dir / CONFIG_FROM_PRESETS[args.config_from]).resolve()

    try:
        config_map = load_config_map(config_source)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # per-image counts; also track config and ensure images with zero objects appear
    per_image: dict[str, dict[str, int]] = {
        name: empty_counts() for name in config_map
    }
    image_config: dict[str, str] = dict(config_map)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "diameter_um" not in reader.fieldnames:
            print("CSV missing diameter_um column", file=sys.stderr)
            sys.exit(1)
        if "file_name" not in reader.fieldnames:
            print("CSV missing file_name column", file=sys.stderr)
            sys.exit(1)

        unknown = 0
        for row in reader:
            file_name = row["file_name"]
            if file_name not in per_image:
                # Object from an image not found under the config source folders
                unknown += 1
                per_image[file_name] = empty_counts()
                image_config[file_name] = "unknown"
            d = float(row["diameter_um"])
            per_image[file_name][diameter_bin(d)] += 1

    if unknown:
        print(
            f"Warning: {unknown} rows from file_name(s) not in {config_source}",
            file=sys.stderr,
        )

    # --- per-image CSV ---
    per_image_path = out_dir / "size_bin_counts_per_image.csv"
    fieldnames = ["config", "file_name", *BIN_KEYS, "total"]
    with per_image_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for file_name in sorted(per_image, key=lambda n: (image_config[n], n)):
            counts = per_image[file_name]
            writer.writerow(
                {
                    "config": image_config[file_name],
                    "file_name": file_name,
                    **counts,
                    "total": sum(counts.values()),
                }
            )

    # --- per-config aggregates (raw + mean per image) ---
    folder_totals: dict[str, dict[str, int]] = defaultdict(empty_counts)
    folder_n_images: dict[str, int] = defaultdict(int)
    for file_name, counts in per_image.items():
        cfg = image_config[file_name]
        folder_n_images[cfg] += 1
        for k in BIN_KEYS:
            folder_totals[cfg][k] += counts[k]

    per_folder_path = out_dir / "size_bin_counts_per_folder.csv"
    folder_fields = [
        "config",
        "n_images",
        *[f"{k}_total" for k in BIN_KEYS],
        "total_objects",
        *[f"{k}_per_image" for k in BIN_KEYS],
        "objects_per_image",
    ]
    with per_folder_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=folder_fields)
        writer.writeheader()
        for cfg in sorted(folder_totals):
            n = folder_n_images[cfg]
            totals = folder_totals[cfg]
            total_objects = sum(totals.values())
            row: dict[str, object] = {
                "config": cfg,
                "n_images": n,
                "total_objects": total_objects,
                "objects_per_image": total_objects / n if n else 0.0,
            }
            for k in BIN_KEYS:
                row[f"{k}_total"] = totals[k]
                row[f"{k}_per_image"] = totals[k] / n if n else 0.0
            writer.writerow(row)

    # --- console summary ---
    print(f"Read {csv_path}")
    print(f"Config source: {config_source}")
    print(f"Wrote {per_image_path}")
    print(f"Wrote {per_folder_path}")
    print()
    print("Diameter bins (µm):", " | ".join(BIN_LABELS[k] for k in BIN_KEYS))
    print()
    for cfg in sorted(folder_totals):
        n = folder_n_images[cfg]
        totals = folder_totals[cfg]
        print(f"{cfg}  (n_images={n})")
        for k in BIN_KEYS:
            mean = totals[k] / n if n else 0.0
            print(
                f"  {BIN_LABELS[k]:>12}:  total={totals[k]:6d}  "
                f"per_image={mean:8.3f}"
            )
        tot = sum(totals.values())
        print(
            f"  {'all':>12}:  total={tot:6d}  "
            f"per_image={tot / n if n else 0.0:8.3f}"
        )
        print()


if __name__ == "__main__":
    main()
