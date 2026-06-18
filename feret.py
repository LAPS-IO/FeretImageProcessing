#!/usr/bin/env python3
"""
Given a path to a ``runN`` folder (e.g. ``outputs/run0``), scan every ``*.npz`` under
``background_difference_watershed/``, load ``labels`` (uint16), and write one CSV row per
connected **instance** region (labels 2--65534 only; void/background 0/1 and edges 65535 are
ignored) with Feret-style maximum diameter (max pairwise distance on the convex hull of the outer
contour) in pixels and micrometers (``diameter_um`` = ``diameter_px`` × ``--um-per-pixel``, default
13.8 µm/px), plus axis-aligned bounding box. By default a two-pixel **edge band** of labels
is cleared (``--edge-strip``) to drop spurious border instances that can look like a white frame
and produce very long Feret chords.

Expected filename stem (relative path leaf without dirs):

    YYYY-MM-DD_hh_mm_ss.<anything>_<sign><7-digit-depth>.npz

Example: ``2025-04-23_14_30_00.png_-0000123.npz`` → datetime ``2025-04-23 14:30:00``, depth ``-123``.

Install ``tqdm`` (``pip install tqdm``) for a progress bar.

Use ``--viz-random N`` to save ``N`` randomly chosen label maps as PNGs with Feret diameter lines
(green) and numeric labels in micrometers (cyan; pixel length × ``--um-per-pixel``, default 13.8).
Optionally pass ``--viz-original-root DIR`` so each stacked image
uses the matching ``.jpg`` under ``DIR`` (same subfolders and filenames as ``*.npz``, ``.jpg`` in
place of ``.npz``). Viz runs **first**. With ``--viz-random``, add ``--csv`` to write the diameter CSV
for **all** npz files afterward; without ``--viz-random``, the CSV is written by default (viz disabled).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

# ``background_difference`` npz ``labels`` (uint16): 0 = void/background, 65535 = edge,
# 2..65534 = instance ids. Feret is computed **only** for instance components, never for 0/1/65535.
LABEL_INSTANCE_MIN = 2
LABEL_INSTANCE_MAX = 65534

# Physical length per image pixel (micrometers); multiply Feret diameter in px for viz / ``diameter_um`` CSV.
DEFAULT_UM_PER_PIXEL = 13.8

# Matches leaf: date_time.rest_depth.npz with signed 7-digit depth
NAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.([^_]+)_([+-])(\d{7})\.npz$"
)


def prepare_labels_for_feret(lab: np.ndarray, edge_strip_px: int) -> np.ndarray:
    """
    Copy ``labels`` with a band of ``edge_strip_px`` pixels along each image edge forced to 0.
    Removes spurious instance labels on the frame (common watershed artifact) that create thin
    white borders and huge hull chords / long diagonal Feret lines.
    Use ``edge_strip_px=0`` to disable.
    """
    out = np.asarray(lab).copy()
    if edge_strip_px <= 0:
        return out
    h, w = out.shape[:2]
    s = min(int(edge_strip_px), h // 2, w // 2)
    if s <= 0:
        return out
    out[:s, :] = 0
    out[-s:, :] = 0
    out[:, :s] = 0
    out[:, -s:] = 0
    return out


def parse_filename(name: str) -> tuple[str | None, int | None]:
    """Return (datetime_iso_space, depth) or (None, None) if pattern does not match."""
    m = NAME_RE.match(name)
    if not m:
        return None, None
    date_s, hh, mm, ss, _mid, sign, digs = m.groups()
    dt_s = f"{date_s} {hh}:{mm}:{ss}"
    depth = int(sign + digs)
    return dt_s, depth


def bbox_and_max_hull_diameter(
    mask_u8: np.ndarray,
) -> tuple[float, int, int, int, int, tuple[int, int], tuple[int, int]]:
    """
    ``mask_u8`` binary 0/255. Returns (diameter_px, top, left, bottom, right, p1, p2).
    ``p1``/``p2`` are ``(x, y)`` endpoints of the longest hull chord; ``(-1, -1)`` if undefined.
    ``top``/``left``/``bottom``/``right`` are inclusive integer indices (row/col).
    """
    ys = np.where(mask_u8 >= 128)[0]
    xs = np.where(mask_u8 >= 128)[1]
    if ys.size == 0:
        return 0.0, -1, -1, -1, -1, (-1, -1), (-1, -1)
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, top, left, bottom, right, (-1, -1), (-1, -1)

    pts = np.vstack(contours)[:, 0, :]  # Nx2 as (x, y)
    if len(pts) < 2:
        return 0.0, top, left, bottom, right, (-1, -1), (-1, -1)

    hull = cv2.convexHull(pts.astype(np.float32))
    h = hull[:, 0, :]  # Kx2 (x, y)
    k = len(h)
    if k < 2:
        return 0.0, top, left, bottom, right, (-1, -1), (-1, -1)

    dmax = 0.0
    bi = bj = 0
    for i in range(k):
        for j in range(i + 1, k):
            dx = float(h[i, 0] - h[j, 0])
            dy = float(h[i, 1] - h[j, 1])
            d = (dx * dx + dy * dy) ** 0.5
            if d > dmax:
                dmax = d
                bi, bj = i, j
    p1 = (int(round(h[bi, 0])), int(round(h[bi, 1])))
    p2 = (int(round(h[bj, 0])), int(round(h[bj, 1])))
    return dmax, top, left, bottom, right, p1, p2


def _line_mask_inside_component(
    comp_u8: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    thickness: int,
) -> np.ndarray:
    """
    Pixels belonging to the thick segment ``p1``–``p2`` that also lie on the instance
    ``comp_u8`` (255 = fg). Avoids drawing Feret chords across void/background where the hull
    chord cuts outside a concave mask.
    """
    h, w = comp_u8.shape[:2]
    line = np.zeros((h, w), dtype=np.uint8)
    cv2.line(line, p1, p2, 255, thickness, lineType=cv2.LINE_8)
    fg = (comp_u8 >= 128).astype(np.uint8) * 255
    return cv2.bitwise_and(line, fg)


def _is_instance_label(uid: int) -> bool:
    return LABEL_INSTANCE_MIN <= uid <= LABEL_INSTANCE_MAX


def iter_instance_components(lab: np.ndarray):
    """
    Yield binary ``uint8`` masks (0/255) for each 8-connected **instance** component only.
    Skips void/background (``0``, ``1``) and inter-instance edges (``65535``); does not measure
    Feret on those label values.
    """
    for uid in np.unique(lab):
        ui = int(uid)
        if not _is_instance_label(ui):
            continue
        mask = (lab == ui).astype(np.uint8) * 255
        n_cc, cc = cv2.connectedComponents(mask, connectivity=8)
        for kk in range(1, n_cc):
            yield ((cc == kk).astype(np.uint8)) * 255


def iter_watershed_npz(run_dir: Path) -> list[Path]:
    ws = run_dir / "background_difference_watershed"
    if not ws.is_dir():
        raise FileNotFoundError(f"Not a directory: {ws}")
    out: list[Path] = []
    for p in sorted(ws.rglob("*.npz")):
        if p.is_file():
            out.append(p)
    return out


def process_npz(
    path: Path, writer: csv.writer, edge_strip_px: int, um_per_pixel: float
) -> None:
    leaf = path.name
    dt_s, depth = parse_filename(leaf)

    data = np.load(path)
    if "labels" not in data.files:
        raise KeyError(f"{path}: npz has no array 'labels', keys={data.files}")
    lab = prepare_labels_for_feret(np.asarray(data["labels"]), edge_strip_px)
    if lab.ndim != 2:
        raise ValueError(f"{path}: labels must be 2D, shape={lab.shape}")

    for comp in iter_instance_components(lab):
        diam, top, left, bottom, right, _p1, _p2 = bbox_and_max_hull_diameter(comp)
        if top < 0 or diam <= 0:
            continue
        diam_um = float(diam) * um_per_pixel
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
                f"{diam_um:.6f}",
            ]
        )


def draw_feret_overlays_bgr(
    vis: np.ndarray,
    lab: np.ndarray,
    *,
    um_per_pixel: float = DEFAULT_UM_PER_PIXEL,
) -> None:
    """
    Draw Feret chords (green) and diameter labels in micrometers (cyan) onto ``vis``
    (BGR, same H×W as ``lab``). Labels are centered horizontally on each component’s
    bounding box; with 50% probability the text is placed just **above** the top edge,
    otherwise just **below** the bottom edge. Text uses ``diameter_px * um_per_pixel``.
    ``lab`` must already include any edge-strip preprocessing (call ``prepare_labels_for_feret`` first).
    """
    th = max(1, min(lab.shape) // 400 + 1)
    ih, iw = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Readable on large frames (was 0.35 px — effectively invisible on ~2k–4k images).
    font_scale = 0.5*max(0.7, min(2.5, min(ih, iw) / 500.0))
    font_th = max(1, min(4, int(round(font_scale * 1.8))))
    for comp in iter_instance_components(lab):
        diam, top, left, bottom, right, p1, p2 = bbox_and_max_hull_diameter(comp)
        if diam <= 0 or p1[0] < 0 or p2[0] < 0:
            continue
        lm = _line_mask_inside_component(comp, p1, p2, th)
        vis[lm > 0] = (0, 255, 0)
        cx_bb = (int(left) + int(right)) // 2
        diam_um = float(diam) * um_per_pixel
        label = f"{diam_um:.1f} um"
        (tw, th_txt), bl = cv2.getTextSize(label, font, font_scale, font_th)
        tx = int(np.clip(cx_bb - tw // 2, 0, max(0, iw - tw - 1)))
        gap = max(6, int(round(font_scale * 8)))
        # ``putText`` ``y`` = baseline; glyph extent ~ [y - th_txt + bl, y + bl].
        if random.random() < 0.5:
            ty = int(np.clip(int(top) - gap - bl, th_txt, ih - 1))
        else:
            ty = int(
                np.clip(int(bottom) + gap + th_txt + bl, th_txt + bl, ih - 1)
            )
        # Outline so cyan reads on both light and dark backgrounds.
        cv2.putText(
            vis,
            label,
            (tx, ty),
            font,
            font_scale,
            (0, 0, 0),
            font_th + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (tx, ty),
            font,
            font_scale,
            (0, 255, 255),
            font_th,
            cv2.LINE_AA,
        )


def render_binary_with_diameters(
    lab: np.ndarray,
    edge_strip_px: int,
    *,
    um_per_pixel: float = DEFAULT_UM_PER_PIXEL,
) -> np.ndarray:
    """Binary instance mask (white on black BGR); Feret lines/text only on instance pixels."""
    lab = prepare_labels_for_feret(lab, edge_strip_px)
    inst = (
        (lab >= LABEL_INSTANCE_MIN)
        & (lab <= LABEL_INSTANCE_MAX)
    ).astype(np.uint8) * 255
    vis = cv2.cvtColor(inst, cv2.COLOR_GRAY2BGR)
    draw_feret_overlays_bgr(vis, lab, um_per_pixel=um_per_pixel)
    return vis


def render_on_original_bgr(
    base_bgr: np.ndarray,
    lab: np.ndarray,
    edge_strip_px: int,
    *,
    um_per_pixel: float = DEFAULT_UM_PER_PIXEL,
) -> np.ndarray:
    """Resize ``base_bgr`` to label size if needed, then draw Feret overlays."""
    lab = prepare_labels_for_feret(lab, edge_strip_px)
    h, w = lab.shape[:2]
    vis = np.asarray(base_bgr).copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim != 3 or vis.shape[2] < 3:
        raise ValueError(f"Expected H×W or H×W×3 image, got shape {vis.shape}")
    else:
        vis = vis[:, :, :3].copy()
    if vis.shape[0] != h or vis.shape[1] != w:
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)
    draw_feret_overlays_bgr(vis, lab, um_per_pixel=um_per_pixel)
    return vis


def npz_path_to_original_jpg(npz_path: Path, ws_root: Path, original_root: Path) -> Path:
    """Same relative path as ``npz`` under ``ws_root``, but under ``original_root`` with ``.jpg``."""
    rel = npz_path.relative_to(ws_root)
    return (original_root / rel).with_suffix(".jpg")


def viz_npz_path_to_png_path(npz_path: Path, ws_root: Path, viz_dir: Path) -> Path:
    """Flat unique name under viz_dir from path relative to watershed root."""
    try:
        rel = npz_path.relative_to(ws_root)
    except ValueError:
        rel = Path(npz_path.name)
    stem = rel.as_posix().replace("/", "__")
    return viz_dir / f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute maximum hull-based Feret diameter per instance region in "
            "run_dir/background_difference_watershed/**/*.npz"
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run folder (e.g. outputs/run0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <run_dir>/feret_diameters.csv)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help=(
            "With --viz-random N>0: after saving viz PNGs, write diameter CSV for all npz "
            "files. Ignored when --viz-random is 0 (CSV is always written in that case)."
        ),
    )
    parser.add_argument(
        "--viz-random",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Save N randomly chosen npz files as binary PNGs with diameter lines drawn "
            "(runs first). Combine with --csv to compute diameters for every npz afterward "
            "(default: 0, disabled)"
        ),
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Directory for viz PNGs (default: <run_dir>/feret_viz_random)",
    )
    parser.add_argument(
        "--viz-original-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "With --viz-random: draw Feret lines on matching JPEGs under DIR instead of binary "
            "masks. Layout mirrors background_difference_watershed: same subfolders and basename "
            "as each .npz but with .jpg (e.g. DIR/foo/bar_x.npz → DIR/foo/bar_x.jpg). If a file "
            "is missing or unreadable, falls back to the binary viz for that sample."
        ),
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=None,
        metavar="S",
        help="RNG seed for --viz-random (default: nondeterministic)",
    )
    parser.add_argument(
        "--edge-strip",
        type=int,
        default=2,
        metavar="P",
        help=(
            "Clear labels to 0 in a P-pixel band on each image edge before Feret/viz (default: 2). "
            "Reduces spurious border segments that cause full-image diagonals. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--um-per-pixel",
        type=float,
        default=DEFAULT_UM_PER_PIXEL,
        metavar="U",
        help=(
            "Micrometers per image pixel: viz text and CSV ``diameter_um`` = Feret diameter (px) × U "
            f"(default: {DEFAULT_UM_PER_PIXEL})."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    if int(args.edge_strip) < 0:
        print("--edge-strip must be >= 0", file=sys.stderr)
        sys.exit(1)
    um_pp = float(args.um_per_pixel)
    if um_pp <= 0:
        print("--um-per-pixel must be > 0", file=sys.stderr)
        sys.exit(1)

    viz_orig = args.viz_original_root
    if viz_orig is not None:
        viz_orig = viz_orig.resolve()
        if not viz_orig.is_dir():
            print(f"Not a directory: {viz_orig}", file=sys.stderr)
            sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = run_dir / "feret_diameters.csv"
    else:
        out_path = out_path.resolve()

    paths = iter_watershed_npz(run_dir)
    if not paths:
        print(f"No .npz files under {run_dir / 'background_difference_watershed'}", file=sys.stderr)
        sys.exit(1)

    ws_root = run_dir / "background_difference_watershed"

    n_viz = int(args.viz_random)
    # Without --viz-random, keep legacy behavior: always write CSV. With --viz-random N>0, CSV
    # only when --csv is passed (viz runs first).
    want_csv = (n_viz == 0) or bool(args.csv)

    if viz_orig is not None and n_viz <= 0:
        print("--viz-original-root requires --viz-random N with N > 0", file=sys.stderr)
        sys.exit(1)

    if n_viz > 0:
        if args.viz_seed is not None:
            random.seed(args.viz_seed)
        take = min(n_viz, len(paths))
        picked = random.sample(paths, take)
        viz_dir = args.viz_dir
        if viz_dir is None:
            viz_dir = run_dir / "feret_viz_random"
        else:
            viz_dir = viz_dir.resolve()
        viz_dir.mkdir(parents=True, exist_ok=True)

        esp = int(args.edge_strip)
        for p in tqdm(picked, desc="Viz PNG", unit="img"):
            data = np.load(p)
            if "labels" not in data.files:
                print(f"{p}: skip viz, no labels", file=sys.stderr)
                continue
            lab = np.asarray(data["labels"])
            if lab.ndim != 2:
                continue
            if viz_orig is not None:
                jpg_path = npz_path_to_original_jpg(p, ws_root, viz_orig)
                base = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
                if base is None:
                    print(
                        f"{p}: missing or unreadable JPEG {jpg_path}, using binary viz",
                        file=sys.stderr,
                    )
                    img = render_binary_with_diameters(lab, esp, um_per_pixel=um_pp)
                else:
                    try:
                        img = render_on_original_bgr(
                            base, lab, esp, um_per_pixel=um_pp
                        )
                    except ValueError as e:
                        print(f"{p}: {e}; using binary viz", file=sys.stderr)
                        img = render_binary_with_diameters(lab, esp, um_per_pixel=um_pp)
            else:
                img = render_binary_with_diameters(lab, esp, um_per_pixel=um_pp)
            outp = viz_npz_path_to_png_path(p, ws_root, viz_dir)
            outp.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(outp), img):
                print(f"Failed to write {outp}", file=sys.stderr)
                sys.exit(1)

        print(f"Wrote {take} viz PNGs under {viz_dir}")

    if want_csv:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
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
            )
            esp = int(args.edge_strip)
            for p in tqdm(paths, desc="Watershed npz", unit="file"):
                try:
                    process_npz(p, w, esp, um_pp)
                except Exception as e:
                    print(f"{p}: {e}", file=sys.stderr)
                    sys.exit(1)

        print(f"Wrote {out_path} ({len(paths)} files scanned)")
    elif n_viz > 0:
        print(
            "Skipping CSV (--csv not set; use --csv after --viz-random to export all diameters)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
