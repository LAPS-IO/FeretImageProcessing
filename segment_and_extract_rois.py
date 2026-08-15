#!/usr/bin/env python3
"""Segment each image and extract its ROI crops in a single in-memory pass.

This combines :mod:`background_difference` (rolling/sampled background diff +
watershed segmentation) and :mod:`extract_rois` (per-instance ROI crops) so that
every image is read **once** and its watershed labels are used directly, without
writing or re-reading any ``.npz`` file.

For each image below ``root`` (one subfolder per date/config, images inside):

1. build the background (rolling mean of neighbors, or a sampled per-date one),
2. compute ``max(0, background - image)``,
3. run the watershed pipeline to get instance labels, then
4. crop one ROI per outer instance component into the output folder.

The instance ``--watershed-min-area`` filter already drops tiny objects during
segmentation, so the separate ROI ``--min-area`` option is intentionally removed
(it would be redundant).

With ``--feret-csv PATH`` the Feret diameter of every instance is also computed
from the very same in-memory labels and written to a CSV (columns identical to
``feret.py``), so no ``.npz`` is written or re-read at any point.

Usage::

    python3 segment_and_extract_rois.py <root> --rolling --watershed-t 4 \
        --circular-morph 11 --no-depth-filter -o roi_crops \
        --feret-csv feret_diameters.csv -v
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

import background_difference as bd
from extract_rois import (
    component_bboxes,
    crop_with_border,
    keep_outer_components,
    _is_whole_image,
)
from feret import (
    DEFAULT_UM_PER_PIXEL,
    bbox_and_max_hull_diameter,
    iter_instance_components,
    parse_filename,
    prepare_labels_for_feret,
)

FERET_CSV_HEADER = [
    "datetime",
    "file_name",
    "depth",
    "top",
    "left",
    "bottom",
    "right",
    "diameter_px",
    "diameter_um",
]

CROP_EXIST_RE = re.compile(r"_x\d+_y\d+\.(?:png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


def resolve_run_dir(out_root: Path, run_dir: Optional[Path]) -> Path:
    """Directory that receives ``run_metadata.txt`` (and usually holds roi_crops/)."""
    if run_dir is not None:
        return run_dir.resolve()
    if out_root.name == "roi_crops":
        return out_root.parent.resolve()
    return out_root.resolve()


def image_already_processed(
    img_path: Path,
    root: Path,
    out_root: Path,
    npz_root: Optional[Path],
    sbs_root: Optional[Path],
) -> bool:
    """True only if every output this run would write for ``img_path`` already exists.

    Partial runs (e.g. side-by-side present but ``roi_crops`` missing) are not skipped.
    """
    rel = img_path.relative_to(root)
    stem_path = rel.with_suffix("")

    def has_crops() -> bool:
        crop_dir = out_root / stem_path.parent
        if not crop_dir.is_dir():
            return False
        prefix = f"{stem_path.name}_x"
        for path in crop_dir.iterdir():
            if (
                path.is_file()
                and path.name.startswith(prefix)
                and CROP_EXIST_RE.search(path.name) is not None
            ):
                return True
        return False

    # Crops are always written by this pipeline.
    if not has_crops():
        return False
    if npz_root is not None and not (npz_root / rel).with_suffix(".npz").is_file():
        return False
    if sbs_root is not None and not (sbs_root / rel).is_file():
        return False
    return True


def write_run_metadata(
    run_dir: Path, lines: Sequence[Tuple[str, str]]
) -> Path:
    """Write ``run_metadata.txt`` in the same key/value format as background_difference."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_metadata.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# segment_and_extract_rois run metadata\n"
            f"# created: {datetime.datetime.now().isoformat()}\n\n"
        )
        width = max((len(k) for k, _ in lines), default=0)
        for key, value in lines:
            f.write(f"{key + ' ' * (width - len(key))} : {value}\n")
    return path


def write_feret_rows(
    writer: "csv.writer",
    labels: np.ndarray,
    img_path: Path,
    edge_strip_px: int,
    um_per_pixel: float,
) -> int:
    """Write one Feret row per instance component; return rows written.

    Mirrors ``feret.process_npz`` but works on in-memory labels. The datetime and
    depth columns are parsed from the image name rewritten with an ``.npz`` suffix
    so the CSV matches ``feret.py`` output exactly.
    """
    leaf = img_path.with_suffix(".npz").name
    dt_s, depth = parse_filename(leaf)
    lab = prepare_labels_for_feret(labels, edge_strip_px)
    written = 0
    for comp in iter_instance_components(lab):
        diam, top, left, bottom, right, _p1, _p2 = bbox_and_max_hull_diameter(comp)
        if top < 0 or diam <= 0:
            continue
        writer.writerow(
            [
                dt_s if dt_s is not None else "",
                leaf,
                depth if depth is not None else "",
                top,
                left,
                bottom,
                right,
                f"{diam:.6f}",
                f"{float(diam) * um_per_pixel:.6f}",
            ]
        )
        written += 1
    return written


def extract_crops_from_labels(
    img_full: np.ndarray,
    labels: np.ndarray,
    rel: Path,
    out_root: Path,
    border: int,
    max_coverage: float,
    verbose: bool,
) -> int:
    """Write one ROI crop per outer instance component; return crops written.

    ``labels`` uses the ``background_difference`` convention (uint16, 2..65534 =
    instances). Components are not area-filtered here because segmentation already
    applied ``--watershed-min-area``.
    """
    ih, iw = img_full.shape[:2]
    lh, lw = labels.shape[:2]
    if (ih, iw) != (lh, lw):
        labels = cv2.resize(labels, (iw, ih), interpolation=cv2.INTER_NEAREST)
        if verbose:
            print(
                f"    resized labels {lw}x{lh} -> {iw}x{ih}",
                file=sys.stderr,
                flush=True,
            )

    comps_all = component_bboxes(labels, 0)
    comps = [c for c in comps_all if not _is_whole_image(c[1], ih, iw, max_coverage)]
    n_whole = len(comps_all) - len(comps)
    boxes = [b for _lid, b, _a in comps]
    kept = keep_outer_components(boxes)
    if verbose:
        print(
            f"    {rel.as_posix()}: {len(comps_all)} components, {len(kept)} kept "
            f"({n_whole} whole-image, {len(comps) - len(kept)} nested)",
            file=sys.stderr,
            flush=True,
        )

    stem_path = rel.with_suffix("")
    crop_dir = out_root / stem_path.parent
    crop_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx in kept:
        _lid, bbox, _area = comps[idx]
        crop = crop_with_border(img_full, bbox, border)
        if crop.size == 0:
            continue
        top, left, _bottom, _right = bbox
        dest = crop_dir / f"{stem_path.name}_x{left}_y{top}.png"
        if not cv2.imwrite(str(dest), crop):
            raise RuntimeError(f"Failed to write: {dest}")
        written += 1
    return written


def _segment_and_extract_one(
    root: Path,
    img_full: np.ndarray,
    img_work: np.ndarray,
    img_path: Path,
    diff: np.ndarray,
    out_root: Path,
    *,
    watershed_t: int,
    circular_morph_radius: Optional[int],
    circular_morph_dilate_iter: int,
    circular_morph_erode_iter: int,
    watershed_min_area: int,
    border: int,
    max_coverage: float,
    orig_hw_pool_restore: Optional[Tuple[int, int]],
    feret_writer: "Optional[csv.writer]",
    edge_strip: int,
    um_per_pixel: float,
    npz_root: Optional[Path],
    sbs_root: Optional[Path],
    verbose: bool,
) -> Tuple[int, int]:
    """Segment one image and extract ROIs (+ optional Feret). Returns (crops, feret_rows)."""
    overlay, labels = bd.watershed_from_diff_threshold(
        img_work,
        diff,
        t=watershed_t,
        circular_morph_radius=circular_morph_radius,
        circular_morph_dilate_iter=circular_morph_dilate_iter,
        circular_morph_erode_iter=circular_morph_erode_iter,
        watershed_min_area=watershed_min_area,
        verbose=verbose,
    )
    labels = bd._maybe_replicate_upscale2x(labels, orig_hw_pool_restore)
    rel = img_path.relative_to(root)
    if npz_root is not None:
        npz_path = (npz_root / rel).with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, labels=labels)
    if sbs_root is not None:
        overlay_out = bd._maybe_replicate_upscale2x(overlay, orig_hw_pool_restore)
        sbs = bd._downscale_side_by_side_half(
            bd._hstack_orig_and_watershed(img_full, overlay_out)
        )
        sbs_path = sbs_root / rel
        sbs_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(sbs_path), sbs):
            raise RuntimeError(f"Failed to write: {sbs_path}")
    crops = extract_crops_from_labels(
        img_full, labels, rel, out_root, border, max_coverage, verbose
    )
    feret_rows = 0
    if feret_writer is not None:
        feret_rows = write_feret_rows(
            feret_writer, labels, img_path, edge_strip, um_per_pixel
        )
    return crops, feret_rows


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root is not a directory: {root}")

    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    min_depth = None if args.no_depth_filter else int(args.min_depth)
    circular_morph_radius = args.circular_morph_r
    use_rolling = bool(args.rolling)
    rolling_half = int(args.rolling_width)
    min_pool2 = bool(args.min_pool2)
    npz_root = args.npz_dir.resolve() if args.npz_dir is not None else None
    sbs_root = args.sbs_dir.resolve() if args.sbs_dir is not None else None
    feret_csv = args.feret_csv.resolve() if args.feret_csv is not None else None
    run_dir = resolve_run_dir(
        out_root, args.run_dir.resolve() if args.run_dir is not None else None
    )
    verbose = bool(args.verbose)
    if npz_root is not None:
        npz_root.mkdir(parents=True, exist_ok=True)
    if sbs_root is not None:
        sbs_root.mkdir(parents=True, exist_ok=True)

    metadata_path = write_run_metadata(
        run_dir,
        [
            ("command", shlex.join(sys.argv)),
            ("run_directory", str(run_dir)),
            ("root", str(root)),
            ("crops_output", str(out_root)),
            ("feret_csv", str(feret_csv) if feret_csv is not None else "None"),
            ("npz_dir", str(npz_root) if npz_root is not None else "None"),
            ("sbs_dir", str(sbs_root) if sbs_root is not None else "None"),
            ("method", str(args.method)),
            ("n_images", str(int(args.n_images))),
            ("rolling", str(use_rolling)),
            ("rolling_width", str(rolling_half)),
            ("min_pool2", str(min_pool2)),
            ("min_depth", "off" if min_depth is None else str(min_depth)),
            ("watershed_t", str(int(args.watershed_t))),
            ("watershed_min_area", str(int(args.watershed_min_area))),
            (
                "circular_morph_r",
                str(circular_morph_radius)
                if circular_morph_radius is not None
                else "None",
            ),
            (
                "circular_morph_dilate_iter",
                str(int(args.circular_morph_dilate_iter)),
            ),
            (
                "circular_morph_erode_iter",
                str(int(args.circular_morph_erode_iter)),
            ),
            ("border", str(int(args.border))),
            ("max_coverage", str(float(args.max_coverage))),
            ("edge_strip", str(int(args.edge_strip))),
            ("um_per_pixel", str(float(args.um_per_pixel))),
            ("skip_existing", str(bool(args.skip_existing))),
            ("verbose", str(verbose)),
        ],
    )
    if verbose:
        print(f"Wrote {metadata_path}", file=sys.stderr, flush=True)

    if verbose:
        cfg = (
            f"rolling half-width={rolling_half}"
            if use_rolling
            else f"method={args.method} n_images={args.n_images}"
        )
        print(
            f"segment_and_extract: root={root} out={out_root} {cfg} "
            f"watershed_t={args.watershed_t} watershed_min_area={args.watershed_min_area} "
            f"min_depth={'off' if min_depth is None else min_depth}",
            file=sys.stderr,
            flush=True,
        )

    stats = {"processed": 0, "crops": 0, "feret": 0, "skipped": 0}
    failed: list[Path] = []
    skip_existing = bool(args.skip_existing)

    def should_skip(img_path: Path) -> bool:
        if not skip_existing:
            return False
        if image_already_processed(img_path, root, out_root, npz_root, sbs_root):
            stats["skipped"] += 1
            if verbose:
                print(
                    f"  skip existing: {img_path.relative_to(root)}",
                    file=sys.stderr,
                    flush=True,
                )
            return True
        return False

    def traverse(feret_writer: "Optional[csv.writer]") -> None:
        # Children of root are frame folders (e.g. Config 01 / Basler_*_frames).
        frame_dirs = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        for frame_dir in frame_dirs:
            if verbose:
                print(f"\n[{frame_dir.relative_to(root)}]", file=sys.stderr, flush=True)

            if use_rolling:
                paths = bd._collect_images_under(frame_dir)
                if not paths:
                    if verbose:
                        print("  (no images, skip)", file=sys.stderr, flush=True)
                    continue
                by_parent = bd._group_images_by_parent(paths)
                for _parent, in_folder in by_parent.items():
                    ordered = sorted(in_folder, key=lambda p: p.name)
                    for i, img_path in enumerate(ordered):
                        if min_depth is not None:
                            depth = bd._parse_depth_from_name(img_path.name)
                            if depth is not None and depth <= min_depth:
                                continue
                        if should_skip(img_path):
                            continue
                        neighbor_paths = bd._rolling_neighbor_paths(
                            ordered, i, rolling_half
                        )
                        if not neighbor_paths:
                            print(
                                f"skipping (no neighbor frames): {img_path}",
                                file=sys.stderr,
                            )
                            continue
                        _accumulate(
                            _process_path(
                                root,
                                img_path,
                                out_root,
                                background_source=("rolling", neighbor_paths),
                                args=args,
                                min_pool2=min_pool2,
                                circular_morph_radius=circular_morph_radius,
                                feret_writer=feret_writer,
                                npz_root=npz_root,
                                sbs_root=sbs_root,
                                verbose=verbose,
                            ),
                            img_path,
                        )
            else:
                background = bd._build_background_in_ram(
                    frame_dir, args.method, int(args.n_images)
                )
                if background is None:
                    if verbose:
                        print("  (no images, skip)", file=sys.stderr, flush=True)
                    continue
                for img_path in bd._collect_images_under(frame_dir):
                    if min_depth is not None:
                        depth = bd._parse_depth_from_name(img_path.name)
                        if depth is not None and depth <= min_depth:
                            continue
                    if should_skip(img_path):
                        continue
                    _accumulate(
                        _process_path(
                            root,
                            img_path,
                            out_root,
                            background_source=("fixed", background),
                            args=args,
                            min_pool2=min_pool2,
                            circular_morph_radius=circular_morph_radius,
                            feret_writer=feret_writer,
                            npz_root=npz_root,
                            sbs_root=sbs_root,
                            verbose=verbose,
                        ),
                        img_path,
                    )

    def _accumulate(result: Tuple[bool, int, int], img_path: Path) -> None:
        ok, crops, feret_rows = result
        if ok:
            stats["processed"] += 1
            stats["crops"] += crops
            stats["feret"] += feret_rows
        else:
            failed.append(img_path)

    if feret_csv is not None:
        feret_csv.parent.mkdir(parents=True, exist_ok=True)
        append = skip_existing and feret_csv.is_file() and feret_csv.stat().st_size > 0
        with open(feret_csv, "a" if append else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(FERET_CSV_HEADER)
            traverse(writer)
    else:
        traverse(None)

    summary = (
        f"\nsegment_and_extract: processed {stats['processed']} image(s), "
        f"skipped {stats['skipped']} existing, "
        f"wrote {stats['crops']} crop(s) under {out_root}; "
        f"metadata -> {run_dir / 'run_metadata.txt'}"
    )
    if npz_root is not None:
        summary += f"; labels .npz under {npz_root}"
    if sbs_root is not None:
        summary += f"; side-by-side under {sbs_root}"
    if feret_csv is not None:
        summary += f"; {stats['feret']} Feret row(s) -> {feret_csv}"
    print(summary, flush=True)
    if failed:
        print(
            f"segment_and_extract: {len(failed)} image(s) failed and were skipped:",
            file=sys.stderr,
            flush=True,
        )
        for path in failed:
            print(f"  {path}", file=sys.stderr, flush=True)


def _process_path(
    root: Path,
    img_path: Path,
    out_root: Path,
    *,
    background_source: Tuple[str, object],
    args: argparse.Namespace,
    min_pool2: bool,
    circular_morph_radius: Optional[int],
    feret_writer: "Optional[csv.writer]",
    npz_root: Optional[Path],
    sbs_root: Optional[Path],
    verbose: bool,
) -> Tuple[bool, int, int]:
    """Read one image, segment it, extract crops (+ Feret). Returns (ok, crops, feret_rows)."""
    try:
        kind, payload = background_source
        if kind == "rolling":
            background = bd._build_background_rolling(payload)  # type: ignore[arg-type]
        else:
            background = payload  # type: ignore[assignment]

        img = bd._read_bgr(img_path)
        if img.shape != background.shape:
            raise ValueError(
                f"Image shape {img.shape} does not match background {background.shape}"
            )

        img_full = img
        orig_hw: Optional[Tuple[int, int]] = None
        if min_pool2:
            img_work, background_work, orig_hw = bd._min_pool2_pair(img, background)
        else:
            img_work, background_work = img, background
        diff = bd._diff_nonnegative(img_work, background_work)

        if verbose:
            print(
                f"  processing {img_path.relative_to(root)} …",
                file=sys.stderr,
                flush=True,
            )
        t0 = time.perf_counter()
        crops, feret_rows = _segment_and_extract_one(
            root,
            img_full,
            img_work,
            img_path,
            diff,
            out_root,
            watershed_t=int(args.watershed_t),
            circular_morph_radius=circular_morph_radius,
            circular_morph_dilate_iter=int(args.circular_morph_dilate_iter),
            circular_morph_erode_iter=int(args.circular_morph_erode_iter),
            watershed_min_area=int(args.watershed_min_area),
            border=int(args.border),
            max_coverage=float(args.max_coverage),
            orig_hw_pool_restore=orig_hw,
            feret_writer=feret_writer,
            edge_strip=int(args.edge_strip),
            um_per_pixel=float(args.um_per_pixel),
            npz_root=npz_root,
            sbs_root=sbs_root,
            verbose=verbose,
        )
        if verbose:
            extra = f", {feret_rows} feret" if feret_writer is not None else ""
            print(
                f"  done {img_path.relative_to(root)} "
                f"({crops} crop(s){extra}, {time.perf_counter() - t0:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
        return True, crops, feret_rows
    except Exception as e:
        print(f"ERROR processing image {img_path}: {e}", file=sys.stderr, flush=True)
        return False, 0, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment each image (background diff + watershed) and extract one ROI crop per "
            "instance component in a single in-memory pass (no .npz written or re-read). "
            "The redundant ROI --min-area is omitted; use --watershed-min-area instead."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder: one subfolder per date/config, each holding images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("roi_crops"),
        help="Output folder for ROI crops (default: ./roi_crops).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for run_metadata.txt (default: parent of --output when that folder "
            "is named roi_crops, otherwise --output itself)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip images that already have all outputs this run would write (ROI crops, "
            "and .npz / side-by-side when those dirs are set). Partial outputs are "
            "reprocessed. When set with an existing feret CSV, new rows are appended."
        ),
    )

    seg = parser.add_argument_group("segmentation")
    seg.add_argument(
        "--method",
        choices=["mean", "median"],
        default="mean",
        help="(non-rolling) How to combine background frames (default: mean).",
    )
    seg.add_argument(
        "-n",
        "--n-images",
        type=int,
        default=25,
        metavar="N",
        help="(non-rolling) Random images for the per-date background (default: 25).",
    )
    seg.add_argument(
        "--rolling",
        action="store_true",
        help="Per-file background = mean of the W previous + W next frames in the same folder.",
    )
    seg.add_argument(
        "--rolling-width",
        type=int,
        default=5,
        metavar="W",
        help="With --rolling, neighbors on each side (default: 5).",
    )
    seg.add_argument(
        "--min-depth",
        type=int,
        default=500,
        metavar="D",
        help="Only process images whose filename depth tag is > D (default: 500).",
    )
    seg.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="Disable depth filtering entirely (overrides --min-depth).",
    )
    seg.add_argument(
        "--min-pool2",
        action="store_true",
        help="Run the pipeline at half resolution via 2x2 min-pool, then upscale labels back.",
    )
    seg.add_argument(
        "--watershed-t",
        type=int,
        default=8,
        metavar="T",
        help="Watershed threshold: diff gray > T = foreground, ==0 = background (default: 8).",
    )
    seg.add_argument(
        "--watershed-min-area",
        type=int,
        default=250,
        metavar="A",
        help="Remove instance labels with pixel count < A after merge; 0 disables (default: 250).",
    )
    seg.add_argument(
        "--circular-morph",
        type=int,
        default=None,
        nargs="?",
        const=2,
        dest="circular_morph_r",
        metavar="R",
        help="Dilate then erode the label-derived binary with a circular SE of radius R (default off; const 2).",
    )
    seg.add_argument(
        "--circular-morph-dilate-iter",
        type=int,
        default=1,
        metavar="N",
        help="(with --circular-morph) Dilate iterations (default: 1).",
    )
    seg.add_argument(
        "--circular-morph-erode-iter",
        type=int,
        default=1,
        metavar="N",
        help="(with --circular-morph) Erode iterations after dilate (default: 1).",
    )
    seg.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also save each label map as a compressed .npz (key 'labels') under DIR, mirroring "
            "the input relative paths (same layout as background_difference_watershed). Skipped "
            "when omitted."
        ),
    )
    seg.add_argument(
        "--sbs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also save a half-size 'original | watershed overlay' side-by-side image under DIR, "
            "mirroring the input relative paths (same as background_difference_watershed_side_by_side). "
            "Skipped when omitted."
        ),
    )

    roi = parser.add_argument_group("ROI extraction")
    roi.add_argument(
        "--border",
        type=int,
        default=10,
        metavar="P",
        help="Pixels added on each side of every component bbox, clamped (default: 10).",
    )
    roi.add_argument(
        "--max-coverage",
        type=float,
        default=0.98,
        metavar="F",
        help="Skip whole-image components covering >= F of both dimensions (default: 0.98).",
    )

    feret = parser.add_argument_group("Feret (optional)")
    feret.add_argument(
        "--feret-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also compute Feret diameters from the same in-memory labels and write them to "
            "this CSV (columns match feret.py). Feret is skipped when omitted."
        ),
    )
    feret.add_argument(
        "--um-per-pixel",
        type=float,
        default=DEFAULT_UM_PER_PIXEL,
        metavar="U",
        help=f"Micrometers per pixel for diameter_um (default: {DEFAULT_UM_PER_PIXEL}).",
    )
    feret.add_argument(
        "--edge-strip",
        type=int,
        default=2,
        metavar="P",
        help="Clear a P-pixel image-edge band before Feret measurement (default: 2).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-folder / per-image progress to stderr.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.rolling and int(args.rolling_width) < 1:
        print("--rolling-width must be >= 1 with --rolling", file=sys.stderr)
        sys.exit(1)
    if not args.rolling and int(args.n_images) < 1:
        print("-n/--n-images must be >= 1", file=sys.stderr)
        sys.exit(1)
    if not 0 <= int(args.watershed_t) <= 255:
        print("--watershed-t must be 0..255", file=sys.stderr)
        sys.exit(1)
    if int(args.watershed_min_area) < 0:
        print("--watershed-min-area must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.circular_morph_r is not None and int(args.circular_morph_r) < 1:
        print("--circular-morph radius must be >= 1", file=sys.stderr)
        sys.exit(1)
    if int(args.border) < 0:
        print("--border must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < float(args.max_coverage) <= 1.0):
        print("--max-coverage must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if int(args.edge_strip) < 0:
        print("--edge-strip must be >= 0", file=sys.stderr)
        sys.exit(1)
    if float(args.um_per_pixel) <= 0:
        print("--um-per-pixel must be > 0", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
