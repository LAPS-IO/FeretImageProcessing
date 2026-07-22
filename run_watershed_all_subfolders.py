#!/usr/bin/env python3
"""
For each direct child directory of a given path, run::

    python3 background_difference.py <subfolder> --rolling --watershed
        --watershed-t 4 --watershed-side-by-side --no-diff --circular-morph 11
        [--min-depth D | --no-depth-filter]

Usage::

    python3 run_watershed_all_subfolders.py <parent_path> [--min-depth D | --no-depth-filter]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Runs background_difference with --rolling --watershed (and other fixed flags) "
            "on each subfolder of parent_path."
        )
    )
    parser.add_argument("parent_path", type=Path, help="Parent folder; each direct subfolder is processed")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--min-depth",
        type=int,
        default=None,
        metavar="D",
        help="Passed through to background_difference.py's --min-depth (its own default is 500 if omitted).",
    )
    depth_group.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="Passed through to background_difference.py's --no-depth-filter (disables depth filtering).",
    )
    args = parser.parse_args()

    parent = args.parent_path.resolve()
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
    if args.no_depth_filter:
        extra.append("--no-depth-filter")
    elif args.min_depth is not None:
        extra.extend(["--min-depth", str(args.min_depth)])

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