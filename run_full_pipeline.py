#!/usr/bin/env python3
"""Run segmentation, Feret measurement, and ROI extraction for an image tree.

The input directory must contain one or more subdirectories with images. A new
``runN`` directory is created under ``--output-base`` and used by all stages.

Example:

    python3 run_full_pipeline.py "/path/to/Dia 08" -v
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pnm",
    ".pgm",
    ".ppm",
}
RUN_RE = re.compile(r"^run\d+$")


def run_command(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_directories(output_base: Path) -> set[Path]:
    if not output_base.is_dir():
        return set()
    return {
        path.resolve()
        for path in output_base.iterdir()
        if path.is_dir() and RUN_RE.fullmatch(path.name)
    }


def validate_image_tree(images_root: Path) -> None:
    subdirectories = [
        path
        for path in images_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if not subdirectories:
        raise ValueError(
            f"{images_root} must contain at least one image subdirectory"
        )

    has_image = any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for directory in subdirectories
        for path in directory.rglob("*")
    )
    if not has_image:
        raise ValueError(f"No supported images found below {images_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment an image tree, compute Feret diameters, and extract one ROI "
            "crop per segmented component."
        )
    )
    parser.add_argument(
        "images_root",
        type=Path,
        help="Folder containing the image subfolders to process.",
    )
    parser.add_argument(
        "-o",
        "--output-base",
        type=Path,
        default=REPO / "outputs",
        help="Base directory where a new runN is created (default: ./outputs).",
    )
    parser.add_argument(
        "--rolling-width",
        type=int,
        default=5,
        metavar="W",
        help="Neighbor frames on each side for rolling background (default: 5).",
    )
    parser.add_argument(
        "--watershed-threshold",
        type=int,
        default=4,
        metavar="T",
        help="Watershed foreground threshold (default: 4).",
    )
    parser.add_argument(
        "--watershed-min-area",
        type=int,
        default=250,
        metavar="A",
        help="Discard segmented instances smaller than A pixels (default: 250).",
    )
    parser.add_argument(
        "--circular-morph",
        type=int,
        default=11,
        metavar="R",
        help="Circular morphology radius used by segmentation (default: 11).",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=None,
        metavar="D",
        help="Only process images with a filename depth strictly greater than D.",
    )
    parser.add_argument(
        "--no-segmentation-preview",
        action="store_true",
        help="Do not create watershed side-by-side preview images.",
    )
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=13.8,
        metavar="U",
        help="Micrometers represented by one pixel (default: 13.8).",
    )
    parser.add_argument(
        "--edge-strip",
        type=int,
        default=2,
        metavar="P",
        help="Clear a P-pixel image-edge band before Feret measurement (default: 2).",
    )
    parser.add_argument(
        "--roi-border",
        type=int,
        default=10,
        metavar="P",
        help="Pixels added around each ROI bounding box (default: 10).",
    )
    parser.add_argument(
        "--roi-min-area",
        type=int,
        default=0,
        metavar="A",
        help="Skip ROI components smaller than A pixels (default: 0).",
    )
    parser.add_argument(
        "--roi-max-coverage",
        type=float,
        default=0.98,
        metavar="F",
        help="Skip ROIs covering at least F of both image dimensions (default: 0.98).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose segmentation and ROI extraction output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = args.images_root.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()

    try:
        if not images_root.is_dir():
            raise ValueError(f"Not a directory: {images_root}")
        validate_image_tree(images_root)
        if args.rolling_width < 1:
            raise ValueError("--rolling-width must be at least 1")
        if not 0 <= args.watershed_threshold <= 255:
            raise ValueError("--watershed-threshold must be between 0 and 255")
        if args.watershed_min_area < 0:
            raise ValueError("--watershed-min-area must be non-negative")
        if args.circular_morph < 1:
            raise ValueError("--circular-morph must be at least 1")
        if args.um_per_pixel <= 0:
            raise ValueError("--um-per-pixel must be positive")
        if args.edge_strip < 0 or args.roi_border < 0 or args.roi_min_area < 0:
            raise ValueError("edge strip, ROI border, and ROI minimum area cannot be negative")
        if not 0 < args.roi_max_coverage <= 1:
            raise ValueError("--roi-max-coverage must be in (0, 1]")
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    scripts = {
        "segmentation": REPO / "background_difference.py",
        "feret": REPO / "feret.py",
        "roi": REPO / "run_extraction_all_folders.py",
    }
    missing = [str(path) for path in scripts.values() if not path.is_file()]
    if missing:
        print(f"Missing required script(s): {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    runs_before = run_directories(output_base)
    segmentation_command = [
        sys.executable,
        str(scripts["segmentation"]),
        str(images_root),
        "--rolling",
        "--rolling-width",
        str(args.rolling_width),
        "--watershed",
        "--watershed-t",
        str(args.watershed_threshold),
        "--watershed-min-area",
        str(args.watershed_min_area),
        "--circular-morph",
        str(args.circular_morph),
        "--no-diff",
        "--output",
        str(output_base),
    ]
    if args.min_depth is None:
        segmentation_command.append("--no-depth-filter")
    else:
        segmentation_command.extend(["--min-depth", str(args.min_depth)])
    if not args.no_segmentation_preview:
        segmentation_command.append("--watershed-side-by-side")
    if args.verbose:
        segmentation_command.append("--verbose")

    print("Stage 1/3: segmentation", flush=True)
    run_command(segmentation_command)

    new_runs = run_directories(output_base) - runs_before
    if len(new_runs) != 1:
        names = ", ".join(sorted(path.name for path in new_runs)) or "none"
        print(
            f"Expected exactly one new run directory, found: {names}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    run_dir = new_runs.pop()

    print("\nStage 2/3: Feret diameter computation", flush=True)
    run_command(
        [
            sys.executable,
            str(scripts["feret"]),
            str(run_dir),
            "--edge-strip",
            str(args.edge_strip),
            "--um-per-pixel",
            str(args.um_per_pixel),
        ]
    )

    roi_command = [
        sys.executable,
        str(scripts["roi"]),
        str(images_root),
        str(run_dir),
        "--border",
        str(args.roi_border),
        "--min-area",
        str(args.roi_min_area),
        "--max-coverage",
        str(args.roi_max_coverage),
    ]
    if args.verbose:
        roi_command.append("--verbose")

    print("\nStage 3/3: ROI extraction", flush=True)
    run_command(roi_command)

    print(f"\nPipeline complete: {run_dir}", flush=True)
    print(f"Feret CSV: {run_dir / 'feret_diameters.csv'}", flush=True)
    print(f"ROI crops: {run_dir / 'roi_crops'}", flush=True)


if __name__ == "__main__":
    main()
