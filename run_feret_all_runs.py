#!/usr/bin/env python3
"""
For each subdirectory of a given path that contains ``background_difference_watershed``, run::

    python3 feret.py <run_dir> --viz-random 100 --csv

Usage::

    python3 run_feret_all_runs.py <parent_folder>

Example: parent holds ``run0``, ``run1``, … each structured like ``feret.py`` expects.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def iter_run_dirs(parent: Path) -> list[Path]:
    ws_name = "background_difference_watershed"
    out: list[Path] = []
    for p in sorted(parent.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / ws_name).is_dir():
            out.append(p)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <parent_folder>", file=sys.stderr)
        print(
            "  Runs feret.py with --viz-random 100 --csv on each subfolder that has "
            "background_difference_watershed/.",
            file=sys.stderr,
        )
        sys.exit(1)

    parent = Path(sys.argv[1]).resolve()
    if not parent.is_dir():
        print(f"Not a directory: {parent}", file=sys.stderr)
        sys.exit(1)

    script = Path(__file__).resolve().parent / "feret.py"
    if not script.is_file():
        print(f"Missing: {script}", file=sys.stderr)
        sys.exit(1)

    runs = iter_run_dirs(parent)
    if not runs:
        print(
            f"No subfolders with background_difference_watershed under: {parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    feret_args = ["--viz-random", "100", "--csv"]

    for run_dir in runs:
        cmd = [sys.executable, str(script), str(run_dir), *feret_args]
        print("---", flush=True)
        print(" ", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if r.returncode != 0:
            print(
                f"Command failed with exit {r.returncode} for: {run_dir}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
