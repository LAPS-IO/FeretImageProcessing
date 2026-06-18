#!/usr/bin/env python3
"""
Extract per-component ROI crops from images using their ``.npz`` segmentations.

Inputs: a folder of images and a folder of ``.npz`` label maps (key ``labels``, uint16, with the
``background_difference`` convention: 0 = void/background, 1 = sure background, 65535 = inter-instance
edge, and 2..65534 = instance ids). For each ``.npz`` the matching image is found at the **same
relative path** under the images folder (any common image extension).

For every 8-connected **instance** component, a crop is taken around its bounding box with a
``--border`` (default 10) pixel margin on each side, clamped to the image. Components whose bounding
box is contained inside another component's bounding box are skipped (only the outer one is kept).

Components that span (almost) the whole image are ignored (see ``--max-coverage``). All crops are
saved directly in the output folder (no per-image subfolders); the source relative path is flattened
into each filename to keep names unique::

    <output>/<rel_flattened>_x<left>_y<top>.png

By default the output folder is the run directory: if the npz folder is inside a run's
``background_difference_watershed``, crops go to ``<run_dir>/roi_crops`` (override with ``-o``).

Install ``tqdm`` (``pip install tqdm``) for a progress bar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

# ``background_difference`` npz ``labels`` (uint16): 0/1 = background, 65535 = edge, 2..65534 = instances.
LABEL_INSTANCE_MIN = 2
LABEL_INSTANCE_MAX = 65534

# Run subfolder that holds watershed label .npz files (see background_difference.py).
WATERSHED_SUBDIR = "background_difference_watershed"

# Candidate image extensions, tried in order for the file matching each ``.npz`` (same relpath).
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# Inclusive integer bbox: (top, left, bottom, right).
BBox = Tuple[int, int, int, int]


def find_image_for_npz(
    npz_path: Path, npz_root: Path, images_root: Path
) -> Optional[Path]:
    """Image at the same relative path as ``npz_path`` (under ``images_root``), any known extension."""
    rel = npz_path.relative_to(npz_root)
    for ext in IMAGE_EXTENSIONS:
        cand = (images_root / rel).with_suffix(ext)
        if cand.is_file():
            return cand
    # Also accept an exact-stem match if the npz stem itself carries the original suffix.
    stem_dir = (images_root / rel).parent
    if stem_dir.is_dir():
        for ext in IMAGE_EXTENSIONS:
            cand = stem_dir / (rel.stem + ext)
            if cand.is_file():
                return cand
    return None


def component_bboxes(lab: np.ndarray, min_area: int) -> List[Tuple[int, BBox, int]]:
    """
    For every 8-connected instance component (label 2..65534), return
    ``(label_id, (top, left, bottom, right), area_px)``. Components with area < ``min_area`` skipped.
    """
    out: List[Tuple[int, BBox, int]] = []
    for uid in np.unique(lab):
        ui = int(uid)
        if not (LABEL_INSTANCE_MIN <= ui <= LABEL_INSTANCE_MAX):
            continue
        mask = (lab == ui).astype(np.uint8)
        n_cc, _cc, stats, _cent = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )
        for k in range(1, n_cc):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[k, cv2.CC_STAT_LEFT])
            y = int(stats[k, cv2.CC_STAT_TOP])
            w = int(stats[k, cv2.CC_STAT_WIDTH])
            h = int(stats[k, cv2.CC_STAT_HEIGHT])
            out.append((ui, (y, x, y + h - 1, x + w - 1), area))
    return out


def _bbox_area(b: BBox) -> int:
    top, left, bottom, right = b
    return (bottom - top + 1) * (right - left + 1)


def _is_whole_image(b: BBox, h: int, w: int, max_coverage: float) -> bool:
    """True if bbox ``b`` covers >= ``max_coverage`` of both image width and height."""
    top, left, bottom, right = b
    wf = (right - left + 1) / float(w)
    hf = (bottom - top + 1) / float(h)
    return wf >= max_coverage and hf >= max_coverage


def _is_inside(a: BBox, b: BBox) -> bool:
    """True if bbox ``a`` is contained within bbox ``b`` (inclusive)."""
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return bt <= at and bl <= al and ab <= bb and ar <= br


def keep_outer_components(boxes: List[BBox]) -> List[int]:
    """
    Indices of boxes that are **not** contained inside any other box. For equal boxes (mutual
    containment), only the earliest index is kept so duplicates are not all dropped.
    """
    areas = [_bbox_area(b) for b in boxes]
    keep: List[int] = []
    for i, bi in enumerate(boxes):
        contained = False
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            if not _is_inside(bi, bj):
                continue
            # Strictly larger container drops i; for equal boxes keep the lower index.
            if areas[j] > areas[i] or (areas[j] == areas[i] and j < i):
                contained = True
                break
        if not contained:
            keep.append(i)
    return keep


def crop_with_border(img: np.ndarray, bbox: BBox, border: int) -> np.ndarray:
    """Crop ``img`` to ``bbox`` grown by ``border`` px per side, clamped to the image."""
    h, w = img.shape[:2]
    top, left, bottom, right = bbox
    y0 = max(0, top - border)
    x0 = max(0, left - border)
    y1 = min(h - 1, bottom + border)
    x1 = min(w - 1, right + border)
    return img[y0 : y1 + 1, x0 : x1 + 1]


def iter_npz(npz_root: Path) -> List[Path]:
    return [p for p in sorted(npz_root.rglob("*.npz")) if p.is_file()]


def default_output_dir(npz_root: Path) -> Path:
    """
    Default crops folder. If ``npz_root`` is within a run's ``background_difference_watershed``,
    save inside that run as ``<run_dir>/roi_crops``; otherwise ``./roi_crops``.
    """
    for anc in (npz_root, *npz_root.parents):
        if anc.name == WATERSHED_SUBDIR:
            return anc.parent / "roi_crops"
    return Path("roi_crops")


def process_one(
    npz_path: Path,
    npz_root: Path,
    images_root: Path,
    out_root: Path,
    border: int,
    min_area: int,
    max_coverage: float,
    verbose: bool,
) -> int:
    """Write crops for one ``.npz``; return the number of crops written."""
    img_path = find_image_for_npz(npz_path, npz_root, images_root)
    if img_path is None:
        print(f"{npz_path}: no matching image under {images_root}", file=sys.stderr)
        return 0

    data = np.load(npz_path)
    if "labels" not in data.files:
        print(f"{npz_path}: npz has no 'labels' (keys={data.files})", file=sys.stderr)
        return 0
    lab = np.asarray(data["labels"])
    if lab.ndim != 2:
        print(f"{npz_path}: labels must be 2D, got {lab.shape}", file=sys.stderr)
        return 0

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"{img_path}: could not read image", file=sys.stderr)
        return 0

    ih, iw = img.shape[:2]
    lh, lw = lab.shape[:2]
    if (ih, iw) != (lh, lw):
        # Align labels to image pixels (nearest preserves ids); keeps crops at full image resolution.
        lab = cv2.resize(lab, (iw, ih), interpolation=cv2.INTER_NEAREST)
        if verbose:
            _log(verbose, f"  resized labels {lw}x{lh} -> {iw}x{ih} for {npz_path.name}")

    comps_all = component_bboxes(lab, min_area)
    # Drop whole-image components first: otherwise they'd contain everything and the nesting
    # filter would discard all the real components.
    comps = [c for c in comps_all if not _is_whole_image(c[1], ih, iw, max_coverage)]
    n_whole = len(comps_all) - len(comps)
    boxes = [b for _lid, b, _a in comps]
    kept = keep_outer_components(boxes)
    if verbose:
        _log(
            verbose,
            f"  {npz_path.name}: {len(comps_all)} components, {len(kept)} kept "
            f"({n_whole} whole-image, {len(comps) - len(kept)} nested)",
        )

    rel = npz_path.relative_to(npz_root)
    prefix = rel.with_suffix("").as_posix().replace("/", "__")
    written = 0
    for idx in kept:
        _lid, bbox, _area = comps[idx]
        crop = crop_with_border(img, bbox, border)
        if crop.size == 0:
            continue
        top, left, _bottom, _right = bbox
        name = f"{prefix}_x{left}_y{top}.png"
        dest = out_root / name
        if not cv2.imwrite(str(dest), crop):
            raise RuntimeError(f"Failed to write: {dest}")
        written += 1
    return written


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop one ROI per instance component (label 2..65534) from each image using its .npz "
            "labels, with a pixel border, skipping components whose bbox is inside another's."
        )
    )
    parser.add_argument(
        "images_root",
        type=Path,
        help="Folder of images (matched to .npz by identical relative path).",
    )
    parser.add_argument(
        "npz_root",
        type=Path,
        help="Folder of .npz segmentations (key 'labels', uint16).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output folder for crops. Default: <run_dir>/roi_crops when npz_root is under a run's "
            f"{WATERSHED_SUBDIR}, else ./roi_crops."
        ),
    )
    parser.add_argument(
        "--border",
        type=int,
        default=10,
        metavar="P",
        help="Pixels added on each side of every component bbox, clamped to the image (default: 10).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        metavar="A",
        help="Skip components with pixel area < A before any nesting filter (default: 0).",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.98,
        metavar="F",
        help=(
            "Skip 'whole image' components whose bbox covers >= F of both image width and height "
            "(default: 0.98). Applied before the nesting filter."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-file progress to stderr.",
    )
    args = parser.parse_args()

    images_root = args.images_root.resolve()
    npz_root = args.npz_root.resolve()
    if not images_root.is_dir():
        print(f"Not a directory: {images_root}", file=sys.stderr)
        sys.exit(1)
    if not npz_root.is_dir():
        print(f"Not a directory: {npz_root}", file=sys.stderr)
        sys.exit(1)
    if int(args.border) < 0:
        print("--border must be >= 0", file=sys.stderr)
        sys.exit(1)
    if int(args.min_area) < 0:
        print("--min-area must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < float(args.max_coverage) <= 1.0):
        print("--max-coverage must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    out_root = (
        args.output if args.output is not None else default_output_dir(npz_root)
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    npz_paths = iter_npz(npz_root)
    if not npz_paths:
        print(f"No .npz files under {npz_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for p in tqdm(npz_paths, desc="ROI crops", unit="file"):
        try:
            total += process_one(
                p,
                npz_root,
                images_root,
                out_root,
                int(args.border),
                int(args.min_area),
                float(args.max_coverage),
                bool(args.verbose),
            )
        except Exception as e:  # keep going across files; report and continue
            print(f"{p}: {e}", file=sys.stderr)

    print(f"Wrote {total} crop(s) under {out_root} ({len(npz_paths)} npz scanned)")


if __name__ == "__main__":
    main()
