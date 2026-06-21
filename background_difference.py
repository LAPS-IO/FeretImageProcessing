#!/usr/bin/env python3
"""
For each date folder, either sample N random images (mean or median) to build one background,
or with --rolling use a per-image mean of the W files before/after in the same directory
(filename order). Then save max(0, image - background) for each file.
Layout: <output>/background_difference/<relpath_from_root>/ and the same relpath under
<output>/background_difference_background/ for the background image used for that frame.
Optional: with --bboxes, also <output>/background_difference_otsu_bboxes/ (after full watershed,
one axis-aligned box per instance label ``>= 2`` on the original: 100..999 px yellow, 1000+ green,
each bbox expanded by ``BBOX_OUTWARD_MARGIN_PX`` per side, clipped; instances with area < 100 skipped).
Use --no-diff to skip writing the diff image folder; requires --bboxes.
With --bboxes --bboxes-side-by-side, also save a half-size diff | original+bboxes image to
<output>/runN/background_difference_otsu_bboxes_side_by_side/.
Use -v / --verbose for stderr progress, including per-step watershed timings and slow-loop progress.
With --min-pool2, pad to even H×W (edge replicate), 2×2 min-pool **image** and **background**, run the
pipeline at half resolution, then replicate each pixel to 2×2 (no interpolation) and crop to the
original size for all saved outputs; watershed side-by-side uses the full-res frame on the left.
With --watershed-seeds, save initial marker maps (green=fg seed, red=bg seed, else original) under
<output>/runN/background_difference_watershed_seeds/.
"""

from __future__ import annotations

import argparse
import datetime
import random
import re
import shlex
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

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
Method = Literal["mean", "median"]

# Depth tag in image filenames: ``..._<sign><7-digit-depth>.<ext>`` (e.g. ``..._+0000512.jpg``).
DEPTH_RE = re.compile(r"_([+-]\d{7})\.[^.]+$")


def _parse_depth_from_name(name: str) -> Optional[int]:
    """Signed depth parsed from an image filename, or ``None`` if the tag is absent."""
    m = DEPTH_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _vlog(verbose: bool, msg: str, indent: str = "      ") -> None:
    if verbose:
        print(f"{indent}{msg}", file=sys.stderr, flush=True)


def _group_images_by_parent(paths: List[Path]) -> dict[Path, List[Path]]:
    g: dict[Path, List[Path]] = {}
    for p in paths:
        g.setdefault(p.parent, []).append(p)
    return g


def _rolling_neighbor_paths(
    ordered: Sequence[Path], index: int, half: int
) -> List[Path]:
    return list(ordered[index - half : index]) + list(ordered[index + 1 : index + 1 + half])


def _build_background_rolling(
    neighbor_paths: Sequence[Path],
) -> np.ndarray:
    if not neighbor_paths:
        raise ValueError("rolling background needs at least one neighbor image")
    stack = _load_stack(neighbor_paths)
    return _aggregate(stack, "mean")


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _collect_images_under(dir_path: Path) -> List[Path]:
    paths: List[Path] = []
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def _load_stack(
    paths: Sequence[Path],
) -> np.ndarray:
    arrs: List[np.ndarray] = []
    ref_shape: Tuple[int, ...] | None = None
    for path in paths:
        img = _read_bgr(path)
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(
                f"Size mismatch: {path} has shape {img.shape}, expected {ref_shape}. "
                "All sampled images must have the same dimensions."
            )
        arrs.append(img.astype(np.float32))
    return np.stack(arrs, axis=0)


def _aggregate(stack: np.ndarray, method: Method) -> np.ndarray:
    if method == "mean":
        out = np.mean(stack, axis=0)
    else:
        out = np.median(stack, axis=0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _build_background_in_ram(
    date_dir: Path, method: Method, n: int
) -> Optional[np.ndarray]:
    all_images = _collect_images_under(date_dir)
    if not all_images:
        return None
    k = min(n, len(all_images))
    sample = (
        all_images
        if k == len(all_images)
        else random.sample(all_images, k)
    )
    stack = _load_stack(sample)
    return _aggregate(stack, method)


def _diff_nonnegative(
    image: np.ndarray, background: np.ndarray
) -> np.ndarray:
    d = background.astype(np.float32) - image.astype(np.float32)
    d = np.maximum(0.0, d)
    return np.clip(np.round(d), 0, 255).astype(np.uint8)


def _pad_to_even_hw(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right with replicate so H and W are even. Returns (padded, (orig_h, orig_w))."""
    h, w = bgr.shape[:2]
    orig = (h, w)
    ph = (2 - h % 2) % 2
    pw = (2 - w % 2) % 2
    if ph == 0 and pw == 0:
        return bgr, orig
    return cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE), orig


def _min_pool2x2(bgr: np.ndarray) -> np.ndarray:
    """Non-overlapping 2×2 blocks → one pixel (min over the 4). H and W must be even."""
    h, w = bgr.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"_min_pool2x2 needs even H,W, got {h}×{w}")
    if bgr.ndim == 2:
        return bgr.reshape(h // 2, 2, w // 2, 2).min(axis=(1, 3))
    c = bgr.shape[2]
    return bgr.reshape(h // 2, 2, w // 2, 2, c).min(axis=(1, 3))


def _min_pool2_pair(
    img: np.ndarray, background: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Pad to even size, 2×2 min-pool **img** and **background** (same geometry), return
    (img_pooled, bg_pooled, orig_hw) with orig_hw = (H, W) before pad (for later crop).
    """
    if img.shape != background.shape:
        raise ValueError("img and background must match shape for min-pool")
    ip, orig_hw = _pad_to_even_hw(img)
    bp, orig_bg = _pad_to_even_hw(background)
    if orig_hw != orig_bg:
        raise ValueError("internal: orig_hw mismatch after pad")
    if ip.shape != bp.shape:
        raise ValueError("internal: padded shapes differ")
    return _min_pool2x2(ip), _min_pool2x2(bp), orig_hw


def _replicate_upscale2x_crop(
    arr: np.ndarray, orig_hw: Tuple[int, int]
) -> np.ndarray:
    """Each pixel → 2×2 block (no interpolation); crop to orig_hw (top-left region)."""
    oh, ow = orig_hw
    if arr.ndim == 2:
        up = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        return np.ascontiguousarray(up[:oh, :ow])
    up = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    return np.ascontiguousarray(up[:oh, :ow, :])


def _maybe_replicate_upscale2x(
    arr: np.ndarray, orig_hw: Optional[Tuple[int, int]]
) -> np.ndarray:
    if orig_hw is None:
        return arr
    return _replicate_upscale2x_crop(arr, orig_hw)


def _bgr_by_component_area(area: int) -> Tuple[int, int, int]:
    """BBox color (BGR) for components >= 100 px: 100..999 yellow, 1000+ green."""
    if area < 1000:
        return (0, 255, 255)
    return (0, 255, 0)


# Outward expansion (per side) for ``--bboxes`` rectangles (watershed instance bboxes).
BBOX_OUTWARD_MARGIN_PX: int = 10


def _to_bgr3(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x


def _hstack_diff_and_bbox_viz(
    diff_bgr: np.ndarray, bbox_viz: np.ndarray
) -> np.ndarray:
    left = _to_bgr3(diff_bgr)
    if left.shape[0] != bbox_viz.shape[0] or left.shape[1] != bbox_viz.shape[1]:
        raise ValueError("diff and bbox viz must match in HxW for side-by-side")
    return np.hstack((left, bbox_viz))


def _downscale_side_by_side_half(bgr: np.ndarray) -> np.ndarray:
    """``h×w`` BGR/gray (``uint8``) image scaled to ``(h/2) × (w/2)`` (integer division)."""
    h, w = bgr.shape[:2]
    b = _to_bgr3(bgr) if bgr.ndim == 2 else bgr
    return cv2.resize(
        b, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA
    )


def _circular_morph_dilate_erode_u8(
    gray_u8: np.ndarray,
    radius: int,
    dilate_iter: int,
    erode_iter: int,
) -> np.ndarray:
    """
    Dilate (``dilate_iter``) then erode (``erode_iter``) on 2D :class:`uint8` with a circular
    (ellipse) flat SE of radius ``radius`` (kernel ``2*radius+1``).
    """
    if radius < 1 or (dilate_iter <= 0 and erode_iter <= 0):
        return gray_u8
    u: np.ndarray = np.asarray(gray_u8, dtype=np.uint8)
    if u.ndim != 2:
        raise ValueError("expected 2D uint8 for circular morph")
    k = 2 * radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for _ in range(dilate_iter):
        u = cv2.dilate(u, se, borderType=cv2.BORDER_REPLICATE)
    for _ in range(erode_iter):
        u = cv2.erode(u, se, borderType=cv2.BORDER_REPLICATE)
    return u


def _binary_from_merged_watershed_labels(m_viz: np.ndarray) -> np.ndarray:
    """
    ``uint8`` mask 0/255: foreground instance pixels or inter-instance line (``>=2`` or ``-1``),
    matching the union used in :func:`_merge_watershed_labels_by_cc8`.
    """
    m = m_viz
    u8 = ((m == -1) | (m >= 2)).astype(np.uint8) * 255
    return u8


def _refine_merged_watershed_for_morph(
    m_viz: np.ndarray, binary_morphed: np.ndarray
) -> np.ndarray:
    """
    Pixels outside ``binary_morphed`` become sure background (``1``);
    then refresh fg–fg :func:`-1` edges.
    """
    m2 = m_viz.copy()
    m2[binary_morphed < 128] = 1
    m2[m2 == -1] = 0
    m2[_fg_fg_4n_edge_mask(m2)] = -1
    return m2


def _remove_small_watershed_instances(
    m_viz: np.ndarray, min_area: int, *, verbose: bool = False
) -> np.ndarray:
    """
    Drop instance labels (ids ``>= 2``) with pixel count below ``min_area`` (reassign to ``1``);
    then refresh :func:`-1` inter-instance edges. If ``min_area`` is ``0``, return ``m_viz`` unchanged.
    """
    if min_area <= 0:
        return m_viz
    m = m_viz.copy()
    labels = [int(x) for x in np.unique(m) if int(x) >= 2]
    n_inst = len(labels)
    if verbose and n_inst:
        _vlog(verbose, f"remove small instances: checking {n_inst} ids (min_area={min_area})")
    step = max(1, n_inst // 25) if n_inst > 25 else 1
    for j, lab in enumerate(labels):
        if verbose and (j == 0 or (j + 1) % step == 0 or j == n_inst - 1):
            _vlog(verbose, f"remove small: instance {j + 1}/{n_inst} (label {lab})")
        mask = m == lab
        if int(np.count_nonzero(mask)) < min_area:
            m[mask] = 1
    m[m == -1] = 0
    m[_fg_fg_4n_edge_mask(m)] = -1
    return m


def _next_run_subdir(out_base: Path) -> Path:
    """
    Create ``<out_base>/run0``, ``run1``, ... (max existing ``run<digits>`` + 1) and return it.
    """
    out_base = out_base.resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    run_re = re.compile(r"^run(\d+)$")
    last = -1
    for child in out_base.iterdir():
        if not child.is_dir():
            continue
        mo = run_re.match(child.name)
        if mo is not None:
            last = max(last, int(mo.group(1)))
    nxt = out_base / f"run{last + 1}"
    nxt.mkdir(parents=True, exist_ok=True)
    return nxt


def _write_run_metadata(
    run_dir: Path,
    lines: Sequence[Tuple[str, str]],
) -> None:
    path = run_dir / "run_metadata.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# background_difference run metadata\n# created: {datetime.datetime.now().isoformat()}\n\n"
        )
        w = max(len(k) for k, _ in lines) if lines else 0
        for k, v in lines:
            f.write(f"{k + ' ' * (w - len(k))} : {v}\n")


def _fg_fg_4n_edge_mask(merged: np.ndarray) -> np.ndarray:
    """4-neighbor edges where two different labels are both foregound (>= 2)."""
    a = merged
    e = np.zeros(a.shape, dtype=bool)
    e[1:, :] |= (a[1:, :] >= 2) & (a[:-1, :] >= 2) & (a[1:, :] != a[:-1, :])
    e[:, 1:] |= (a[:, 1:] >= 2) & (a[:, :-1] >= 2) & (a[:, 1:] != a[:, :-1])
    return e


def _merge_watershed_labels_by_cc8(
    m_out: np.ndarray, *, verbose: bool = False
) -> np.ndarray:
    """
    Merge watershed output by turning it into a single binary image, then
    8-CC: foreground OR old watershed line (``-1``) is 1, else 0. That way
    a 1~pixel boundary still connects the two sides for CC. New labels are
    ``1..`` from :func:`cv2.connectedComponents`, re-mapped to ``1`` = sure
    bg, ``2..`` = each merged blob. Finally ``-1`` is placed on 4~neighbor
    transitions between two *different* fg ids (``>=2``) for display (not on
    fg~bg, so small regions can still be tinted).
    """
    # Binary: "anything that is not strict background" for CC (lines included).
    binary_u8 = ((m_out == -1) | (m_out > 1)).astype(np.uint8) * 255
    n, cc = cv2.connectedComponents(
        binary_u8, connectivity=8, ltype=cv2.CV_32S
    )
    n_fg = n - 1
    if verbose:
        h, w = m_out.shape[:2]
        _vlog(
            verbose,
            f"merge CC8: {n_fg} components on {w}×{h} (remap loop can be slow if N is large)",
        )
    merged = np.ones(m_out.shape, dtype=np.int32)
    merged[m_out == 0] = 0
    step = max(500, n_fg // 20) if n_fg > 500 else max(1, n_fg // 5) if n_fg > 5 else 1
    for k in range(1, n):
        if verbose and n_fg > 1 and (k == 1 or k % step == 0 or k == n - 1):
            _vlog(verbose, f"merge CC8: remap component {k}/{n_fg}")
        merged[cc == k] = k + 1
    merged[_fg_fg_4n_edge_mask(merged)] = -1
    return merged


# ``-1`` (fg–fg boundary) is stored as this value in :func:`_merged_labels_to_uint16`.
WATERSHED_LABEL_BOUNDARY_U16: np.uint16 = np.uint16(65535)


def _overlay_watershed_merged(
    bgr: np.ndarray,
    m_viz: np.ndarray,
    alpha: float = 0.5,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Colorized overlay (BGR) for side-by-side and visualization."""
    base = bgr.astype(np.float32)
    overlay = base.copy()
    mmax = int(np.max(m_viz[m_viz > 0])) if np.any(m_viz > 0) else 1
    n_col = max(0, mmax - 1)
    if verbose and n_col > 0:
        _vlog(verbose, f"overlay: tinting up to {n_col} instance label(s) (max id {mmax})")
    step = max(200, n_col // 25) if n_col > 200 else max(1, n_col // 5) if n_col > 5 else 1
    for lab in range(2, mmax + 1):
        if verbose and n_col > 1 and (lab == 2 or (lab - 1) % step == 0 or lab == mmax):
            _vlog(verbose, f"overlay: label {lab}/{mmax}")
        sel = m_viz == lab
        if not np.any(sel):
            continue
        hue = (lab * 47) % 180
        c = (
            cv2.cvtColor(np.uint8([[[hue, 180, 220]]]), cv2.COLOR_HSV2BGR)
            [0, 0, :]
            .astype(np.float32)
        )
        overlay[sel] = alpha * base[sel] + (1 - alpha) * c
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay[m_viz == -1] = (255, 255, 255)
    return overlay


def _merged_labels_to_uint16(m_viz: np.ndarray) -> np.ndarray:
    """
    Merged int32 map (``-1`` = edge, ``0`` = unknown, ``1`` = sure bg, ``2+`` = instances)
    to a single **uint16** array; ``-1`` becomes :data:`WATERSHED_LABEL_BOUNDARY_U16` (65535).
    """
    a = m_viz.astype(np.int64, copy=True)
    a[m_viz == -1] = int(WATERSHED_LABEL_BOUNDARY_U16)
    mx = int(np.max(a)) if a.size else 0
    if (a < 0).any():
        raise ValueError("Watershed labels contain invalid negatives after -1 map")
    if mx > 65534:
        raise ValueError(
            f"Watershed label id {mx} does not fit in uint16 with 65535 reserved for edges"
        )
    return a.astype(np.uint16)


def _watershed_instance_bboxes_viz(
    labels_u16: np.ndarray,
    orig_bgr: np.ndarray,
    *,
    outward_margin_px: int = BBOX_OUTWARD_MARGIN_PX,
    min_area_px: int = 100,
) -> np.ndarray:
    """
    Draw axis-aligned boxes on ``orig_bgr`` for each watershed instance (label ``>= 2``,
    excluding :data:`WATERSHED_LABEL_BOUNDARY_U16`), after the full watershed pipeline.
    Colors match :func:`_bgr_by_component_area` (segmentation convention). Boxes are expanded
    by ``outward_margin_px`` per side and clipped to the image.
    """
    if labels_u16.shape[:2] != orig_bgr.shape[:2]:
        raise ValueError(
            f"labels and original must match in size, got {labels_u16.shape} vs {orig_bgr.shape}"
        )
    vis = orig_bgr.copy()
    H, W = vis.shape[:2]
    m = int(max(0, outward_margin_px))
    bnd = int(WATERSHED_LABEL_BOUNDARY_U16)
    ids = [
        int(u)
        for u in np.unique(labels_u16)
        if int(u) >= 2 and int(u) != bnd
    ]
    for lid in ids:
        mask = labels_u16 == np.uint16(lid)
        area = int(np.count_nonzero(mask))
        if area < min_area_px:
            continue
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        rx0 = max(0, x0 - m)
        ry0 = max(0, y0 - m)
        rx1 = min(W - 1, x1 + m)
        ry1 = min(H - 1, y1 + m)
        color = _bgr_by_component_area(area)
        cv2.rectangle(
            vis, (rx0, ry0), (rx1, ry1), color, thickness=2, lineType=cv2.LINE_8
        )
    return vis


# BGR for :func:`watershed_seeds_visualization_bgr` (red = sure background seed, green = fg seed).
_WATERSHED_SEED_BG_BGR: Tuple[int, int, int] = (0, 0, 255)
_WATERSHED_SEED_FG_BGR: Tuple[int, int, int] = (0, 255, 0)


def _watershed_markers_int32_from_gray(
    gray: np.ndarray, t: int
) -> Tuple[np.ndarray, bool]:
    """
    Build the initial ``markers`` image passed to :func:`cv2.watershed`:
    ``0`` = unknown, ``1`` = sure background seed, ``>= 2`` = foreground seeds
    (connected components of eroded fg, or raw fg if eroded mask is empty).

    Returns
    -------
    markers : ndarray
        int32, same shape as ``gray``.
    has_fg : bool
        True iff ``gray > t`` anywhere (same condition as running full watershed).
    """
    t = int(np.clip(t, 0, 255))
    h, w = gray.shape
    fg = (gray > t).astype(np.uint8) * 255
    has_fg = bool(np.any(fg))
    markers = np.zeros((h, w), dtype=np.int32)
    bg_sure = gray == 0
    if np.any(bg_sure):
        markers[bg_sure] = 1
    else:
        markers[0, :] = 1
        markers[-1, :] = 1
        markers[:, 0] = 1
        markers[:, -1] = 1
    in_between = (gray > 0) & (gray <= t)
    markers[in_between] = 0
    if not has_fg:
        return markers, False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_seed = cv2.erode(fg, kernel, iterations=1)
    n, m = cv2.connectedComponents(fg_seed, connectivity=8, ltype=cv2.CV_32S)
    for lab in range(1, n):
        markers[m == lab] = lab + 1
    if not np.any(fg_seed) and np.any(fg):
        n2, m2 = cv2.connectedComponents(fg, connectivity=8, ltype=cv2.CV_32S)
        for lab in range(1, n2):
            markers[(m2 == lab)] = lab + 1
    return markers, True


def watershed_seeds_visualization_bgr(
    orig_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    t: int,
) -> np.ndarray:
    """
    RGB/BGR view of **initial** watershed seeds on top of the original frame:
    green = foreground seed pixels, red = background seed pixels, unchanged BGR elsewhere (unknown).
    """
    bgr = _to_bgr3(orig_bgr) if orig_bgr.ndim == 2 else orig_bgr
    if diff_bgr.shape[:2] != bgr.shape[:2]:
        raise ValueError("diff and orig must share H, W for watershed seeds")
    if diff_bgr.ndim == 2:
        gray = diff_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    markers, _ = _watershed_markers_int32_from_gray(gray, t)
    vis = bgr.copy()
    vis[markers == 1] = _WATERSHED_SEED_BG_BGR
    vis[markers >= 2] = _WATERSHED_SEED_FG_BGR
    return vis


def watershed_from_diff_threshold(
    orig_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    t: int = 8,
    circular_morph_radius: Optional[int] = None,
    circular_morph_dilate_iter: int = 1,
    circular_morph_erode_iter: int = 1,
    watershed_min_area: int = 100,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pixels on diff (grayscale) with value > t → foreground, value == 0 → background;
    values in (0, t] are unknown. Build markers, run :func:`cv2.watershed` on a BGR copy
    of the original, then binarize ``(m>1)|m==-1``, 8-CC, map labels, then
    4~neighbor fg~fg **-1**.

    If ``circular_morph_radius`` is set, dilate then erode a **binary mask** from the merged
    label map (instance ∪ line, same as :func:`_merge_watershed_labels_by_cc8`); that mask
    is used to remove exterior pixels and to refresh **-1** edges (the BGR diff is not morphed).

    If ``watershed_min_area`` is positive, instance labels (``>=2``) with area below that
    many pixels are removed (set to sure background) and **-1** edges are refreshed.

    Returns
    -------
    overlay_bgr : np.ndarray
        BGR :class:`uint8` (original with tinted regions; boundaries white) for sbs.
    labels_u16 : np.ndarray
        2D :class:`uint16` label map. ``65535`` = inter-instance edge (``-1`` in int32);
        ``0`` = unknown; ``1`` = background; ``2+`` = instance id.
    """
    bgr = _to_bgr3(orig_bgr) if orig_bgr.ndim == 2 else orig_bgr
    if diff_bgr.shape[:2] != bgr.shape[:2]:
        raise ValueError("diff and orig must share H, W for watershed")
    if diff_bgr.ndim == 2:
        gray = diff_bgr.astype(np.uint8, copy=False)
    else:
        gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if verbose:
        _vlog(
            verbose,
            f"watershed: {w}×{h} px, T={int(np.clip(t, 0, 255))}, min_area={watershed_min_area}",
        )
    t0 = time.perf_counter()
    markers, has_fg = _watershed_markers_int32_from_gray(gray, t)
    if not has_fg:
        if verbose:
            _vlog(verbose, "watershed: no foreground (diff > T); return trivial labels")
        labels_fill = np.ones((h, w), dtype=np.uint16)
        return bgr.copy(), labels_fill
    if verbose:
        mx = int(markers.max())
        _vlog(
            verbose,
            f"watershed: markers ready ({time.perf_counter() - t0:.2f}s, fg seed ids up to {mx})",
        )

    if verbose:
        _vlog(verbose, "watershed: cv2.watershed (often the slowest step) …")
    t_ws = time.perf_counter()
    ws = bgr.copy()
    m_out = cv2.watershed(ws, markers)
    if verbose:
        _vlog(verbose, f"watershed: cv2.watershed done ({time.perf_counter() - t_ws:.2f}s)")

    t_merge = time.perf_counter()
    m_viz = _merge_watershed_labels_by_cc8(m_out, verbose=verbose)
    if verbose:
        _vlog(verbose, f"watershed: merge CC8 total ({time.perf_counter() - t_merge:.2f}s)")

    r = circular_morph_radius
    du, eu = int(circular_morph_dilate_iter), int(circular_morph_erode_iter)
    if r is not None and r >= 1 and (du > 0 or eu > 0):
        if verbose:
            _vlog(
                verbose,
                f"watershed: circular morph r={r} dilate={du} erode={eu} …",
            )
        t_m = time.perf_counter()
        b0 = _binary_from_merged_watershed_labels(m_viz)
        b1 = _circular_morph_dilate_erode_u8(b0, r, du, eu)
        if not np.array_equal(b0, b1):
            m_viz = _refine_merged_watershed_for_morph(m_viz, b1)
        if verbose:
            _vlog(verbose, f"watershed: circular morph done ({time.perf_counter() - t_m:.2f}s)")
    t_small = time.perf_counter()
    m_viz = _remove_small_watershed_instances(
        m_viz, int(watershed_min_area), verbose=verbose
    )
    if verbose:
        _vlog(
            verbose,
            f"watershed: remove small instances done ({time.perf_counter() - t_small:.2f}s)",
        )
    t_ov = time.perf_counter()
    overlay = _overlay_watershed_merged(bgr, m_viz, verbose=verbose)
    if verbose:
        _vlog(verbose, f"watershed: overlay done ({time.perf_counter() - t_ov:.2f}s)")
    t_u16 = time.perf_counter()
    labels_u16 = _merged_labels_to_uint16(m_viz)
    if verbose:
        _vlog(
            verbose,
            f"watershed: uint16 labels ({time.perf_counter() - t_u16:.2f}s)",
        )
    return overlay, labels_u16


def _hstack_orig_and_watershed(
    orig_bgr: np.ndarray, wvis_bgr: np.ndarray
) -> np.ndarray:
    a = _to_bgr3(orig_bgr) if orig_bgr.ndim == 2 else orig_bgr
    b = wvis_bgr
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
        raise ValueError("original and watershed viz must match in HxW for side-by-side")
    return np.hstack((a, b))


def _process_one_image(
    root: Path,
    img: np.ndarray,
    img_path: Path,
    background: np.ndarray,
    diff: np.ndarray,
    out_base: Path | None,
    bg_out_base: Path | None,
    bbox_base: Path | None,
    bbox_sbs_base: Path | None,
    save_otsu_bboxes: bool,
    save_otsu_bboxes_side_by_side: bool,
    save_watershed: bool,
    save_watershed_side_by_side: bool,
    watershed_t: int,
    circular_morph_radius: Optional[int],
    circular_morph_dilate_iter: int,
    circular_morph_erode_iter: int,
    watershed_min_area: int,
    ws_base: Path | None,
    ws_sbs_base: Path | None,
    save_watershed_seeds: bool,
    ws_seeds_base: Path | None,
    verbose: bool = False,
    orig_hw_pool_restore: Optional[Tuple[int, int]] = None,
    img_fullres: Optional[np.ndarray] = None,
) -> None:
    rel = img_path.relative_to(root)
    if out_base is not None:
        if verbose:
            _vlog(verbose, f"write diff + background ({rel}) …")
        t0 = time.perf_counter()
        dest = out_base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        diff_out = _maybe_replicate_upscale2x(diff, orig_hw_pool_restore)
        if not cv2.imwrite(str(dest), diff_out):
            raise RuntimeError(f"Failed to write: {dest}")
        if bg_out_base is not None:
            bg_dest = bg_out_base / rel
            bg_dest.parent.mkdir(parents=True, exist_ok=True)
            bg_out = _maybe_replicate_upscale2x(background, orig_hw_pool_restore)
            if not cv2.imwrite(str(bg_dest), bg_out):
                raise RuntimeError(f"Failed to write: {bg_dest}")
        if verbose:
            _vlog(verbose, f"write diff + background done ({time.perf_counter() - t0:.2f}s)")
    need_watershed = (
        save_watershed
        or save_watershed_side_by_side
        or save_otsu_bboxes
        or save_otsu_bboxes_side_by_side
    )
    wvis: Optional[np.ndarray] = None
    wlab: Optional[np.ndarray] = None
    if need_watershed:
        if verbose:
            _vlog(verbose, "watershed pipeline …")
        t_ws_all = time.perf_counter()
        wvis, wlab = watershed_from_diff_threshold(
            img,
            diff,
            t=watershed_t,
            circular_morph_radius=circular_morph_radius,
            circular_morph_dilate_iter=circular_morph_dilate_iter,
            circular_morph_erode_iter=circular_morph_erode_iter,
            watershed_min_area=watershed_min_area,
            verbose=verbose,
        )
        if verbose:
            _vlog(
                verbose,
                f"watershed pipeline done ({time.perf_counter() - t_ws_all:.2f}s total)",
            )
    if save_otsu_bboxes or save_otsu_bboxes_side_by_side:
        if verbose:
            _vlog(verbose, "instance bboxes (watershed labels, +margin) …")
        t0 = time.perf_counter()
        if wlab is None:
            raise RuntimeError("internal: watershed labels missing for --bboxes")
        bviz = _watershed_instance_bboxes_viz(wlab, img)
        bviz_out = _maybe_replicate_upscale2x(bviz, orig_hw_pool_restore)
        if bbox_base is not None:
            bdest = bbox_base / rel
            bdest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(bdest), bviz_out):
                raise RuntimeError(f"Failed to write: {bdest}")
        if bbox_sbs_base is not None:
            diff_sbs = _maybe_replicate_upscale2x(diff, orig_hw_pool_restore)
            sbs = _downscale_side_by_side_half(
                _hstack_diff_and_bbox_viz(diff_sbs, bviz_out)
            )
            sdest = bbox_sbs_base / rel
            sdest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(sdest), sbs):
                raise RuntimeError(f"Failed to write: {sdest}")
        if verbose:
            _vlog(
                verbose,
                f"instance bboxes done ({time.perf_counter() - t0:.2f}s)",
            )
    if save_watershed and ws_base is not None:
        if wlab is None or wvis is None:
            raise RuntimeError("internal: watershed outputs missing for --watershed")
        if verbose:
            _vlog(verbose, f"write watershed .npz ({rel}) …")
        t_npz = time.perf_counter()
        wpath = (ws_base / rel).with_suffix(".npz")
        wpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            wlab_out = _maybe_replicate_upscale2x(wlab, orig_hw_pool_restore)
            np.savez_compressed(wpath, labels=wlab_out)
        except OSError as e:
            raise RuntimeError(f"Failed to write: {wpath}") from e
        if verbose:
            _vlog(verbose, f"write .npz done ({time.perf_counter() - t_npz:.2f}s)")
    if save_watershed_side_by_side and ws_sbs_base is not None:
        if wvis is None:
            raise RuntimeError("internal: watershed viz missing for side-by-side")
        if verbose:
            _vlog(verbose, "side-by-side downscale + write …")
        t_sbs = time.perf_counter()
        img_left = img_fullres if img_fullres is not None else img
        wvis_out = _maybe_replicate_upscale2x(wvis, orig_hw_pool_restore)
        sbs = _downscale_side_by_side_half(
            _hstack_orig_and_watershed(img_left, wvis_out)
        )
        sp = ws_sbs_base / rel
        sp.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(sp), sbs):
            raise RuntimeError(f"Failed to write: {sp}")
        if verbose:
            _vlog(verbose, f"side-by-side done ({time.perf_counter() - t_sbs:.2f}s)")
    if save_watershed_seeds and ws_seeds_base is not None:
        if verbose:
            _vlog(verbose, "watershed seeds (red=bg, green=fg, else original) …")
        t_sd = time.perf_counter()
        seeds_bgr = watershed_seeds_visualization_bgr(img, diff, watershed_t)
        seeds_out = _maybe_replicate_upscale2x(seeds_bgr, orig_hw_pool_restore)
        sdest = ws_seeds_base / rel
        sdest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(sdest), seeds_out):
            raise RuntimeError(f"Failed to write: {sdest}")
        if verbose:
            _vlog(verbose, f"watershed seeds done ({time.perf_counter() - t_sd:.2f}s)")


def run(
    root: Path,
    out_root: Path,
    method: Method = "mean",
    n: int = 25,
    use_rolling: bool = False,
    rolling_half: int = 5,
    save_diff: bool = True,
    save_otsu_bboxes: bool = False,
    save_otsu_bboxes_side_by_side: bool = False,
    save_watershed: bool = False,
    save_watershed_side_by_side: bool = False,
    save_watershed_seeds: bool = False,
    watershed_t: int = 8,
    circular_morph_radius: Optional[int] = None,
    circular_morph_dilate_iter: int = 1,
    circular_morph_erode_iter: int = 1,
    watershed_min_area: int = 100,
    verbose: bool = False,
    min_pool2: bool = False,
    min_depth: Optional[int] = 500,
) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root is not a directory: {root}")
    processed = 0
    if verbose:
        if use_rolling:
            cfg = f"rolling half-width={rolling_half}"
        else:
            cfg = f"method={method} n_images={n}"
        print(
            f"background_difference: root={root} out={out_root.resolve()} {cfg} "
            f"save_diff={save_diff} min_pool2={min_pool2} "
            f"min_depth={'off' if min_depth is None else min_depth}",
            file=sys.stderr,
            flush=True,
        )
    out_base: Path | None = None
    bg_out_base: Path | None = None
    if save_diff:
        out_base = (out_root / "background_difference").resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        bg_out_base = (out_root / "background_difference_background").resolve()
        bg_out_base.mkdir(parents=True, exist_ok=True)
    bbox_base: Path | None = None
    if save_otsu_bboxes:
        bbox_base = (out_root / "background_difference_otsu_bboxes").resolve()
        bbox_base.mkdir(parents=True, exist_ok=True)
    bbox_sbs_base: Path | None = None
    if save_otsu_bboxes_side_by_side:
        bbox_sbs_base = (
            out_root / "background_difference_otsu_bboxes_side_by_side"
        ).resolve()
        bbox_sbs_base.mkdir(parents=True, exist_ok=True)
    ws_base: Path | None = None
    if save_watershed:
        ws_base = (out_root / "background_difference_watershed").resolve()
        ws_base.mkdir(parents=True, exist_ok=True)
    ws_sbs_base: Path | None = None
    if save_watershed_side_by_side:
        ws_sbs_base = (
            out_root / "background_difference_watershed_side_by_side"
        ).resolve()
        ws_sbs_base.mkdir(parents=True, exist_ok=True)
    ws_seeds_base: Path | None = None
    if save_watershed_seeds:
        ws_seeds_base = (
            out_root / "background_difference_watershed_seeds"
        ).resolve()
        ws_seeds_base.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for date_dir in date_dirs:
        if verbose:
            print(f"\n[{date_dir.relative_to(root)}]", file=sys.stderr, flush=True)
        if use_rolling:
            paths = _collect_images_under(date_dir)
            if not paths:
                if verbose:
                    print("  (no images, skip)", file=sys.stderr, flush=True)
                continue
            by_parent = _group_images_by_parent(paths)
            for _parent, in_folder in by_parent.items():
                ordered = sorted(in_folder, key=lambda p: p.name)
                if verbose:
                    print(
                        f"  folder {_parent.relative_to(root)}: {len(ordered)} images",
                        file=sys.stderr,
                        flush=True,
                    )
                for i, img_path in enumerate(ordered):
                    if min_depth is not None:
                        depth = _parse_depth_from_name(img_path.name)
                        if depth is not None and depth <= min_depth:
                            if verbose:
                                _vlog(
                                    verbose,
                                    f"  skip (depth {depth} <= {min_depth}): {img_path.name}",
                                )
                            continue
                    n_paths = _rolling_neighbor_paths(ordered, i, rolling_half)
                    if not n_paths:
                        print(
                            f"skipping (no neighbor frames): {img_path}",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        background = _build_background_rolling(n_paths)
                    except (ValueError, RuntimeError) as e:
                        print(
                            f"skipping {img_path}: {e}",
                            file=sys.stderr,
                        )
                        continue
                    img = _read_bgr(img_path)
                    if img.shape != background.shape:
                        raise ValueError(
                            f"Image shape {img.shape} does not match background "
                            f"{background.shape} for {img_path}"
                        )
                    img_full = img
                    orig_hw: Optional[Tuple[int, int]] = None
                    if min_pool2:
                        img, background_work, orig_hw = _min_pool2_pair(
                            img, background
                        )
                    else:
                        background_work = background
                    diff = _diff_nonnegative(img, background_work)
                    if verbose:
                        print(
                            f"    processing {img_path.relative_to(root)} …",
                            file=sys.stderr,
                            flush=True,
                        )
                    _process_one_image(
                        root,
                        img,
                        img_path,
                        background_work,
                        diff,
                        out_base,
                        bg_out_base,
                        bbox_base,
                        bbox_sbs_base,
                        save_otsu_bboxes,
                        save_otsu_bboxes_side_by_side,
                        save_watershed,
                        save_watershed_side_by_side,
                        watershed_t,
                        circular_morph_radius,
                        circular_morph_dilate_iter,
                        circular_morph_erode_iter,
                        watershed_min_area,
                        ws_base,
                        ws_sbs_base,
                        save_watershed_seeds,
                        ws_seeds_base,
                        verbose=verbose,
                        orig_hw_pool_restore=orig_hw,
                        img_fullres=img_full if min_pool2 else None,
                    )
                    processed += 1
                    if verbose:
                        print(
                            f"    done {img_path.relative_to(root)}",
                            file=sys.stderr,
                            flush=True,
                        )
        else:
            background = _build_background_in_ram(date_dir, method, n)
            if background is None:
                if verbose:
                    print("  (no images, skip)", file=sys.stderr, flush=True)
                continue
            if verbose:
                h, w = background.shape[:2]
                print(
                    f"  background {w}x{h} ({method}, sampled from date tree)",
                    file=sys.stderr,
                    flush=True,
                )
            for img_path in _collect_images_under(date_dir):
                if min_depth is not None:
                    depth = _parse_depth_from_name(img_path.name)
                    if depth is not None and depth <= min_depth:
                        if verbose:
                            _vlog(
                                verbose,
                                f"  skip (depth {depth} <= {min_depth}): {img_path.name}",
                            )
                        continue
                img = _read_bgr(img_path)
                if img.shape != background.shape:
                    raise ValueError(
                        f"Image shape {img.shape} does not match background "
                        f"{background.shape} for {img_path}"
                    )
                img_full = img
                orig_hw: Optional[Tuple[int, int]] = None
                if min_pool2:
                    img, background_work, orig_hw = _min_pool2_pair(img, background)
                else:
                    background_work = background
                diff = _diff_nonnegative(img, background_work)
                if verbose:
                    print(
                        f"  processing {img_path.relative_to(root)} …",
                        file=sys.stderr,
                        flush=True,
                    )
                _process_one_image(
                    root,
                    img,
                    img_path,
                    background_work,
                    diff,
                    out_base,
                    bg_out_base,
                    bbox_base,
                    bbox_sbs_base,
                    save_otsu_bboxes,
                    save_otsu_bboxes_side_by_side,
                    save_watershed,
                    save_watershed_side_by_side,
                    watershed_t,
                    circular_morph_radius,
                    circular_morph_dilate_iter,
                    circular_morph_erode_iter,
                    watershed_min_area,
                    ws_base,
                    ws_sbs_base,
                    save_watershed_seeds,
                    ws_seeds_base,
                    verbose=verbose,
                    orig_hw_pool_restore=orig_hw,
                    img_fullres=img_full if min_pool2 else None,
                )
                processed += 1
                if verbose:
                    print(
                        f"  done {img_path.relative_to(root)}",
                        file=sys.stderr,
                        flush=True,
                    )
    if verbose:
        print(
            f"\nbackground_difference: done, processed {processed} image(s)",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per date: background in RAM from N random images; save max(0, difference) for each "
            "image. Each run creates a new <output>/run0, run1, … folder and writes "
            "<output>/runN/background_difference/ (unless --no-diff) and matching backgrounds under "
            "<output>/runN/background_difference_background/. With --bboxes, watershed instance boxes under "
            "<output>/runN/background_difference_otsu_bboxes/ (after full watershed, margin per side); "
            "optional half-size …_bboxes_side_by_side/; "
            "uint16 label .npz with --watershed under …/background_difference_watershed/; "
            "and half-size …/background_difference_watershed_side_by_side/ with --watershed-side-by-side; "
            "initial marker seeds (green fg, red bg) with --watershed-seeds under …/background_difference_watershed_seeds/. "
            "A run_metadata.txt in each runN records parameters."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder: one subfolder per date, each with subfolders of images",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "median"],
        default="mean",
        help="How to combine background frames (default: mean)",
    )
    parser.add_argument(
        "-n",
        "--n-images",
        type=int,
        default=25,
        metavar="N",
        help="(non-rolling) Number of random images for the per-date background (default: 25)",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help=(
            "Per file: background = mean of the <width> previous + <width> next images in the same "
            "folder (by filename, lexicographic). Ignores --n and --method. "
            "Omit the current image from the mean."
        ),
    )
    parser.add_argument(
        "--rolling-width",
        type=int,
        default=5,
        metavar="W",
        help="With --rolling, neighbors on each side (default: 5, i.e. up to 10 others)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Base output directory: each run creates a new <output>/run0, run1, … and saves under it (default: ./outputs)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Print progress to stderr: run configuration, each file, and per-step timings inside "
            "watershed (cv2.watershed, merge CC8, remove small instances, overlay, .npz, side-by-side)."
        ),
    )
    parser.add_argument(
        "--min-pool2",
        action="store_true",
        help=(
            "After optional even-HW pad (edge replicate), replace each 2×2 block in the frame and "
            "in the background with the minimum BGR (or gray) over those 4 pixels; run diff / "
            "watershed / etc. on the half-size tensors; for every saved image and label map, upscale "
            "by duplicating each pixel into a 2×2 block (no interpolation) and crop to the original "
            "resolution. Watershed side-by-side uses the full-resolution input on the left."
        ),
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=500,
        metavar="D",
        help=(
            "Only process images whose depth tag is strictly greater than D (default 500). "
            "Depth is parsed from the filename suffix '_<sign><7 digits>.<ext>' (e.g. '..._+0000512.jpg'). "
            "Images without a depth tag are always processed. Pass -1 (or any value below the minimum "
            "possible depth) to disable depth filtering."
        ),
    )
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="Disable depth filtering entirely (overrides --min-depth).",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Do not write <output>/runN/background_difference/ (diff). Use another output flag (bboxes, watershed, …).",
    )
    parser.add_argument(
        "--bboxes",
        action="store_true",
        help=(
            "Run the full watershed pipeline, then draw one axis-aligned box per instance (label >= 2) "
            "on the original (skip area < 100; 100–999 yellow, 1000+ green); "
            f"each box expanded by {BBOX_OUTWARD_MARGIN_PX} px per side (clipped). "
            "Saves to <output>/runN/background_difference_otsu_bboxes/ (same relpath as inputs)."
        ),
    )
    parser.add_argument(
        "--bboxes-side-by-side",
        action="store_true",
        help=(
            "Save diff and original+bboxes side-by-side (half width and height vs. full hstack) to "
            "<output>/runN/background_difference_otsu_bboxes_side_by_side/ (can be used with or "
            "without --bboxes; does not require it)"
        ),
    )
    parser.add_argument(
        "--bbox-threshold",
        type=int,
        default=None,
        nargs="?",
        const=16,
        metavar="T",
        dest="bbox_fixed_threshold",
        help=(
            "Ignored (kept for compatibility). Bboxes use watershed instance labels; see --watershed-t "
            "and --watershed-min-area."
        ),
    )
    parser.add_argument(
        "--watershed",
        action="store_true",
        help=(
            "Segment the original (BGR) with cv2.watershed: diff > T = foreground, diff==0 = "
            "background, 0<diff<=T = unknown. After watershed, 8-CC on fg∪boundaries merges "
            "connected seeds. Saves H×W uint16 label map (65535=edge) as compressed .npz (array "
            "key: labels) to <output>/runN/background_difference_watershed/ (use --watershed-min-area to drop small instances). "
        ),
    )
    parser.add_argument(
        "--watershed-t",
        type=int,
        default=8,
        metavar="T",
        help="Threshold T for watershed outputs (default: 8). diff gray > T = fg, ==0 = bg.",
    )
    parser.add_argument(
        "--watershed-min-area",
        type=int,
        default=250,
        metavar="A",
        dest="watershed_min_area",
        help=(
            "After merge (and optional circular-morph on the label mask), remove instance label ids "
            "(value >=2) with pixel count < A; 0 disables (default: 250)."
        ),
    )
    parser.add_argument(
        "--watershed-side-by-side",
        action="store_true",
        help=(
            "Save original (left) | watershed overlay (right) at half width and height to "
            "<output>/runN/background_difference_watershed_side_by_side/ (no need for --watershed; "
            "reuses the same T and min-area as --watershed-t / --watershed-min-area)"
        ),
    )
    parser.add_argument(
        "--watershed-seeds",
        action="store_true",
        dest="watershed_seeds",
        help=(
            "Save BGR images of cv2.watershed **initial** markers to "
            "<output>/runN/background_difference_watershed_seeds/: green = foreground seed pixels, "
            "red = background seed pixels, unchanged color = unknown (uses --watershed-t; does not require --watershed)."
        ),
    )
    parser.add_argument(
        "--circular-morph",
        type=int,
        default=None,
        nargs="?",
        const=2,
        dest="circular_morph_r",
        metavar="R",
        help=(
            "Watershed only: build a 0/255 binary from merged instance labels (and inter-instance "
            "line pixels), then dilate then erode that mask with a flat circular (ellipse) SE of "
            "radius R (kernel 2*R+1); results are applied to the label map, not the BGR diff. With "
            "no value, R=2. See --circular-morph-dilate-iter / --circular-morph-erode-iter."
        ),
    )
    parser.add_argument(
        "--circular-morph-dilate-iter",
        type=int,
        default=1,
        metavar="N",
        help="(with --circular-morph) Dilate iterations on the label-derived binary (default: 1).",
    )
    parser.add_argument(
        "--circular-morph-erode-iter",
        type=int,
        default=1,
        metavar="N",
        help="(with --circular-morph) Erode iterations after dilate (default: 1).",
    )
    args = parser.parse_args()
    if not args.rolling and args.n_images < 1:
        print("N must be at least 1", file=sys.stderr)
        sys.exit(1)
    rw = int(args.rolling_width)
    if rw < 0:
        print("--rolling-width must be >= 0", file=sys.stderr)
        sys.exit(1)
    if rw == 0 and args.rolling:
        print("Use --rolling-width > 0 with --rolling", file=sys.stderr)
        sys.exit(1)
    bft = args.bbox_fixed_threshold
    if bft is not None and (bft < 0 or bft > 255):
        print("--bbox-threshold must be 0..255", file=sys.stderr)
        sys.exit(1)
    wt = int(args.watershed_t)
    if wt < 0 or wt > 255:
        print("--watershed-t must be 0..255", file=sys.stderr)
        sys.exit(1)
    cmr = args.circular_morph_r
    if cmr is not None and cmr < 1:
        print("--circular-morph radius must be >= 1", file=sys.stderr)
        sys.exit(1)
    cdi, cei = int(args.circular_morph_dilate_iter), int(
        args.circular_morph_erode_iter
    )
    if cdi < 0 or cei < 0:
        print("circular-morph *-iter must be >= 0", file=sys.stderr)
        sys.exit(1)
    wma = int(args.watershed_min_area)
    if wma < 0:
        print("--watershed-min-area must be >= 0", file=sys.stderr)
        sys.exit(1)
    min_depth = None if args.no_depth_filter else int(args.min_depth)
    save_diff = not args.no_diff
    save_otsu_bboxes = bool(args.bboxes)
    save_otsu_bboxes_sbs = bool(args.bboxes_side_by_side)
    save_watershed = bool(args.watershed)
    save_watershed_sbs = bool(args.watershed_side_by_side)
    save_watershed_seeds = bool(args.watershed_seeds)
    if not save_diff and not save_otsu_bboxes and not save_otsu_bboxes_sbs and not save_watershed and not save_watershed_sbs and not save_watershed_seeds:
        print(
            "--no-diff with nothing to save: add --bboxes, --bboxes-side-by-side, --watershed, "
            "--watershed-side-by-side, --watershed-seeds, or remove --no-diff",
            file=sys.stderr,
        )
        sys.exit(1)
    run_dir = _next_run_subdir(args.output.resolve())
    _write_run_metadata(
        run_dir,
        [
            ("command", shlex.join(sys.argv)),
            (
                "output_base",
                str(args.output.resolve()),
            ),
            ("run_directory", str(run_dir.resolve())),
            ("root", str(args.root.resolve())),
            ("method", args.method),
            ("n_images", str(int(args.n_images))),
            ("rolling", str(bool(args.rolling))),
            ("rolling_width", str(rw)),
            ("no_diff", str(args.no_diff)),
            ("bboxes", str(args.bboxes)),
            ("bboxes_side_by_side", str(args.bboxes_side_by_side)),
            (
                "bbox_fixed_threshold",
                str(bft) if bft is not None else "None (Otsu if bboxes)",
            ),
            ("bbox_outward_margin_px", str(BBOX_OUTWARD_MARGIN_PX)),
            ("watershed", str(args.watershed)),
            ("watershed_side_by_side", str(args.watershed_side_by_side)),
            ("watershed_seeds", str(bool(args.watershed_seeds))),
            ("watershed_t", str(wt)),
            ("watershed_min_area", str(wma)),
            (
                "circular_morph_r",
                str(cmr) if cmr is not None else "None",
            ),
            ("circular_morph_dilate_iter", str(cdi)),
            ("circular_morph_erode_iter", str(cei)),
            ("verbose", str(bool(args.verbose))),
            ("min_pool2", str(bool(args.min_pool2))),
            ("min_depth", "off" if min_depth is None else str(min_depth)),
        ],
    )
    run(
        args.root,
        run_dir,
        method=args.method,
        n=args.n_images,
        use_rolling=bool(args.rolling),
        rolling_half=rw,
        save_diff=save_diff,
        save_otsu_bboxes=save_otsu_bboxes,
        save_otsu_bboxes_side_by_side=save_otsu_bboxes_sbs,
        save_watershed=save_watershed,
        save_watershed_side_by_side=save_watershed_sbs,
        save_watershed_seeds=save_watershed_seeds,
        watershed_t=wt,
        circular_morph_radius=cmr,
        circular_morph_dilate_iter=cdi,
        circular_morph_erode_iter=cei,
        watershed_min_area=wma,
        verbose=bool(args.verbose),
        min_pool2=bool(args.min_pool2),
        min_depth=min_depth,
    )


if __name__ == "__main__":
    main()
