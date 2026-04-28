#!/usr/bin/env python3
"""
For each direct child directory of a given path, run::

    python3 background_difference.py <subfolder> --rolling --watershed
        --watershed-t 4 --watershed-side-by-side --no-diff --circular-morph 11

Usage::

    python3 run_watershed_all_subfolders.py <parent_path>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <parent_path>",
            file=sys.stderr,
        )
        print(
            "  Runs background_difference with --rolling --watershed (and fixed flags) "
            "on each subfolder of parent_path.",
            file=sys.stderr,
        )
        sys.exit(1)

    parent = Path(sys.argv[1]).resolve()
    if not parent.is_dir():
        print(f"Not a directory: {parent}", file=sys.stderr)
        sys.exit(1)

    script = Path(__file__).resolve().parent / "background_difference.py"
    if not script.is_file():
        print(f"Missing: {script}", file=sys.stderr)
        sys.exit(1)

    subs = sorted(
        p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not subs:
        print(f"No subfolders under: {parent}", file=sys.stderr)
        sys.exit(1)

    extra = [
        "--rolling",
        "--watershed",
        "--watershed-t",
        "4",
        "--watershed-side-by-side",
        "--no-diff",
    ]

    for sub in subs:
        cmd = [sys.executable, str(script), str(sub), *extra]
        print("---", flush=True)
        print(" ", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if r.returncode != 0:
            print(
                f"Command failed with exit {r.returncode} for: {sub}",
                file=sys.stderr,
            )
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
