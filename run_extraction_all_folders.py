#!/usr/bin/env python3
"""
Run ``extract_rois.py`` for every frame subfolder of a run, using its watershed labels.

The npz parent is ``<run_dir>/background_difference_watershed`` and crops are written inside the run
at ``<run_dir>/roi_crops/<name>/`` (override with ``-o``). For each subfolder present under BOTH the
images parent and the watershed folder (matched by name), run::

    python3 extract_rois.py <images_parent>/<name> <run_dir>/background_difference_watershed/<name>
        -o <output>/<name> [extra]

Usage::

    python3 run_extraction_all_folders.py <images_parent> <run_dir> [-o OUTPUT] [--border P]

Per-subfolder output dirs keep identical frame filenames across folders from colliding. Subfolders
missing from either side are skipped with a warning.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Subfolder of a run holding the watershed label .npz files (see background_difference.py).
WATERSHED_SUBDIR = "background_difference_watershed"


def matching_subfolders(images_parent: Path, npz_parent: Path) -> list[str]:
    img_subs = {
        p.name
        for p in images_parent.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }
    npz_subs = {
        p.name
        for p in npz_parent.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }
    return sorted(img_subs & npz_subs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run extract_rois.py for each frame subfolder of a run, using its watershed labels; "
            "crops are written inside the run folder."
        )
    )
    parser.add_argument(
        "images_parent",
        type=Path,
        help="Parent folder holding the image subfolders.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=f"Run folder (e.g. outputs/run9); npz come from <run_dir>/{WATERSHED_SUBDIR}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output parent; each subfolder writes to <output>/<name>/ (default: <run_dir>/roi_crops).",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=10,
        metavar="P",
        help="Pixels added on each side of every component bbox (default: 10).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        metavar="A",
        help="Skip components with pixel area < A (default: 0).",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.98,
        metavar="F",
        help="Skip whole-image components covering >= F of width and height (default: 0.98).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Pass --verbose to extract_rois.py.",
    )
    args = parser.parse_args()

    images_parent = args.images_parent.resolve()
    run_dir = args.run_dir.resolve()
    if not images_parent.is_dir():
        print(f"Not a directory: {images_parent}", file=sys.stderr)
        sys.exit(1)
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    npz_parent = run_dir / WATERSHED_SUBDIR
    if not npz_parent.is_dir():
        print(f"Not a directory: {npz_parent}", file=sys.stderr)
        sys.exit(1)

    script = Path(__file__).resolve().parent / "extract_rois.py"
    if not script.is_file():
        print(f"Missing: {script}", file=sys.stderr)
        sys.exit(1)

    names = matching_subfolders(images_parent, npz_parent)
    if not names:
        print(
            f"No common subfolders between {images_parent} and {npz_parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_parent = (
        args.output.resolve() if args.output is not None else run_dir / "roi_crops"
    )
    extra = [
        "--border",
        str(int(args.border)),
        "--min-area",
        str(int(args.min_area)),
        "--max-coverage",
        str(float(args.max_coverage)),
    ]
    if args.verbose:
        extra.append("--verbose")

    for name in names:
        img_dir = images_parent / name
        npz_dir = npz_parent / name
        out_dir = out_parent / name
        cmd = [
            sys.executable,
            str(script),
            str(img_dir),
            str(npz_dir),
            "-o",
            str(out_dir),
            *extra,
        ]
        print("---", flush=True)
        print(" ", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if r.returncode != 0:
            print(
                f"Command failed with exit {r.returncode} for: {name}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
