#!/usr/bin/env python3
"""Build ``atributos_por_roi.csv`` for one or more run folders.

For each PNG under ``roi_crops/``, joins the Feret diameter from
``feret_diameters.csv`` (match on source filename + left/top) and computes
several contrast measures from the watershed ``.npz`` labels and the source
(or crop) image intensities.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable


REPO = Path(__file__).resolve().parent
FERET_CSV_NAME = "feret_diameters.csv"
ROI_SUBDIR = "roi_crops"
WATERSHED_SUBDIR = "background_difference_watershed"
OUTPUT_CSV_NAME = "atributos_por_roi.csv"
OUTPUT_LOG_NAME = "atributos_por_roi.log"
DEFAULT_BORDER = 10
LABEL_INSTANCE_MIN = 2
LABEL_INSTANCE_MAX = 65534
# Local-background ring: elliptical SE of size K×K (K = 2*radius+1).
BG_DILATE_RADIUS = 4  # → kernel 9×9


class ProcessingCancelled(Exception):
    """Raised when the user requests to stop mid-run."""


CROP_NAME_RE = re.compile(
    r"^(?P<stem>.+)_x(?P<left>\d+)_y(?P<top>\d+)$", re.IGNORECASE
)
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

CONTRAST_COLUMNS = (
    "contrast_rms",
    "contrast_michelson",
    "contrast_range",
    "contrast_cv",
    "contrast_weber",
    "contrast_mean_diff",
)


@dataclass(frozen=True)
class FeretKey:
    file_stem: str
    left: int
    top: int


@dataclass
class FeretRow:
    file_name: str
    left: int
    top: int
    diameter_px: Optional[float]
    diameter_um: Optional[float]
    bottom: Optional[int] = None
    right: Optional[int] = None


def read_run_metadata(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "run_metadata.txt"
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def discover_run_dirs(path: Path) -> List[Path]:
    """Return run directories under ``path`` (or ``[path]`` if it is already a run)."""
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    if (path / ROI_SUBDIR).is_dir():
        return [path]
    runs: List[Path] = []
    for child in sorted(path.rglob(ROI_SUBDIR)):
        if child.is_dir():
            runs.append(child.parent.resolve())
    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique: List[Path] = []
    for run in runs:
        if run not in seen:
            seen.add(run)
            unique.append(run)
    return unique


def parse_crop_name(path: Path) -> Optional[Tuple[str, int, int]]:
    """Return ``(image_stem, left, top)`` from an ROI filename, or ``None``."""
    m = CROP_NAME_RE.match(path.stem)
    if not m:
        return None
    return m.group("stem"), int(m.group("left")), int(m.group("top"))


def load_feret_index(csv_path: Path) -> Dict[FeretKey, FeretRow]:
    index: Dict[FeretKey, FeretRow] = {}
    if not csv_path.is_file():
        return index
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            try:
                left = int(float(row["left"]))
                top = int(float(row["top"]))
            except (KeyError, ValueError, TypeError):
                continue
            bottom = _optional_int(row.get("bottom"))
            right = _optional_int(row.get("right"))
            diam_px = _optional_float(row.get("diameter_px"))
            diam_um = _optional_float(row.get("diameter_um"))
            stem = Path(file_name).stem
            key = FeretKey(stem, left, top)
            index[key] = FeretRow(
                file_name, left, top, diam_px, diam_um, bottom, right
            )
    return index


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_labels(npz_path: Path) -> Optional[np.ndarray]:
    try:
        data = np.load(npz_path)
        if "labels" not in data.files:
            return None
        lab = np.asarray(data["labels"])
        if lab.ndim != 2:
            return None
        return lab
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, ValueError):
        return None


def find_npz_for_crop(
    crop_path: Path, crops_root: Path, npz_root: Path, image_stem: str
) -> Optional[Path]:
    """Map an ROI crop path to its ``.npz`` under the watershed folder."""
    rel = crop_path.relative_to(crops_root)
    parent = rel.parent
    cand = (npz_root / parent / image_stem).with_suffix(".npz")
    if cand.is_file():
        return cand
    # Fallback: search by stem under parent (rare naming mismatches).
    search_dir = npz_root / parent
    if search_dir.is_dir():
        hits = list(search_dir.glob(f"{image_stem}.npz"))
        if len(hits) == 1:
            return hits[0]
    return None


def find_source_image(
    images_root: Optional[Path],
    crops_root: Path,
    crop_path: Path,
    image_stem: str,
) -> Optional[Path]:
    if images_root is None or not images_root.is_dir():
        return None
    rel_parent = crop_path.relative_to(crops_root).parent
    base = images_root / rel_parent / image_stem
    for ext in IMAGE_EXTENSIONS:
        cand = base.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def label_id_at_bbox(
    labels: np.ndarray, left: int, top: int
) -> Optional[int]:
    """Pick the instance id whose bbox top-left matches ``(top, left)``."""
    hit = lookup_component(bbox_top_left_index(labels), labels, left, top)
    return None if hit is None else hit[0]


def bbox_top_left_index(
    labels: np.ndarray,
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """Map ``(left, top)`` → ``(label_id, bottom, right)`` (largest area wins)."""
    index: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    areas: Dict[Tuple[int, int], int] = {}
    for uid in np.unique(labels):
        ui = int(uid)
        if not (LABEL_INSTANCE_MIN <= ui <= LABEL_INSTANCE_MAX):
            continue
        ys, xs = np.where(labels == ui)
        if ys.size == 0:
            continue
        left = int(xs.min())
        top = int(ys.min())
        bottom = int(ys.max())
        right = int(xs.max())
        key = (left, top)
        area = int(ys.size)
        if area > areas.get(key, -1):
            areas[key] = area
            index[key] = (ui, bottom, right)
    return index


def lookup_component(
    index: Dict[Tuple[int, int], Tuple[int, int, int]],
    labels: np.ndarray,
    left: int,
    top: int,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(label_id, bottom, right)`` for the component at ``(left, top)``."""
    hit = index.get((left, top))
    if hit is not None:
        return hit
    for dl, dt in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        hit = index.get((left + dl, top + dt))
        if hit is not None:
            return hit
    h, w = labels.shape
    if 0 <= top < h and 0 <= left < w:
        v = int(labels[top, left])
        if LABEL_INSTANCE_MIN <= v <= LABEL_INSTANCE_MAX:
            ys, xs = np.where(labels == v)
            if ys.size:
                return v, int(ys.max()), int(xs.max())
    return None


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img[..., 0]
    return g.astype(np.float64)


def crop_window(
    h: int, w: int, top: int, left: int, bottom: int, right: int, border: int
) -> Tuple[int, int, int, int]:
    y0 = max(0, top - border)
    x0 = max(0, left - border)
    y1 = min(h, bottom + border + 1)
    x1 = min(w, right + border + 1)
    return y0, x0, y1, x1


def compute_contrasts(
    gray: np.ndarray, fg_mask: np.ndarray, *, bg_dilate_radius: int = BG_DILATE_RADIUS
) -> Dict[str, Optional[float]]:
    """Several contrast definitions on foreground vs local background."""
    out: Dict[str, Optional[float]] = {k: None for k in CONTRAST_COLUMNS}
    fg = fg_mask.astype(bool)
    if not np.any(fg):
        return out

    vals = gray[fg]
    vmin = float(vals.min())
    vmax = float(vals.max())
    vmean = float(vals.mean())
    vstd = float(vals.std())

    out["contrast_rms"] = vstd
    out["contrast_range"] = vmax - vmin
    denom_m = vmax + vmin
    out["contrast_michelson"] = (
        (vmax - vmin) / denom_m if denom_m > 1e-12 else 0.0
    )
    out["contrast_cv"] = (vstd / vmean) if abs(vmean) > 1e-12 else None

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * bg_dilate_radius + 1, 2 * bg_dilate_radius + 1),
    )
    dilated = cv2.dilate(fg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    bg = dilated & ~fg
    if np.any(bg):
        bg_mean = float(gray[bg].mean())
        out["contrast_mean_diff"] = abs(vmean - bg_mean)
        if abs(bg_mean) > 1e-12:
            out["contrast_weber"] = (vmean - bg_mean) / bg_mean
        else:
            out["contrast_weber"] = None
    return out


def contrasts_within_bbox(
    labels: np.ndarray,
    label_id: int,
    gray_full: Optional[np.ndarray],
    crop_bgr: Optional[np.ndarray],
    *,
    top: int,
    left: int,
    bottom: int,
    right: int,
    roi_border: int,
    bg_dilate_radius: int = BG_DILATE_RADIUS,
) -> Dict[str, Optional[float]]:
    """Compute contrast metrics using only the bbox region (+ margin for dilate).

    ``top/left/bottom/right`` come preferably from ``feret_diameters.csv``; the
    mask is still taken from the ``.npz`` labels inside that window.
    """
    empty = {k: None for k in CONTRAST_COLUMNS}
    h, w = labels.shape
    if bottom < top or right < left:
        return empty

    # Expand enough for the local-background ring around the object.
    margin = max(int(roi_border), int(bg_dilate_radius))
    y0, x0, y1, x1 = crop_window(h, w, top, left, bottom, right, margin)
    if y1 <= y0 or x1 <= x0:
        return empty

    labels_roi = labels[y0:y1, x0:x1]
    fg = labels_roi == label_id
    if not np.any(fg):
        return empty

    if gray_full is not None and gray_full.shape[:2] == (h, w):
        return compute_contrasts(
            gray_full[y0:y1, x0:x1],
            fg,
            bg_dilate_radius=bg_dilate_radius,
        )

    # Fallback: intensities from the saved ROI PNG (window with ``roi_border``).
    if crop_bgr is None:
        return empty
    y0b, x0b, y1b, x1b = crop_window(h, w, top, left, bottom, right, roi_border)
    fg_crop = labels[y0b:y1b, x0b:x1b] == label_id
    ch, cw = crop_bgr.shape[:2]
    if fg_crop.shape != (ch, cw):
        if fg_crop.size == 0:
            return empty
        fg_crop = cv2.resize(
            fg_crop.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return compute_contrasts(
        _to_gray_f32(crop_bgr),
        fg_crop,
        bg_dilate_radius=bg_dilate_radius,
    )


def contrasts_for_component(
    labels: np.ndarray,
    label_id: int,
    gray_full: Optional[np.ndarray],
    crop_bgr: Optional[np.ndarray],
    left: int,
    top: int,
    border: int,
    *,
    bottom: Optional[int] = None,
    right: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Back-compat wrapper: resolve bbox then call :func:`contrasts_within_bbox`."""
    if bottom is None or right is None:
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            return {k: None for k in CONTRAST_COLUMNS}
        bottom = int(ys.max()) if bottom is None else bottom
        right = int(xs.max()) if right is None else right
    return contrasts_within_bbox(
        labels,
        label_id,
        gray_full,
        crop_bgr,
        top=top,
        left=left,
        bottom=int(bottom),
        right=int(right),
        roi_border=border,
    )


def iter_roi_crops(crops_root: Path) -> Iterable[Path]:
    for p in sorted(crops_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if CROP_NAME_RE.match(p.stem):
                yield p


def count_csv_data_rows(path: Path) -> int:
    """Number of data rows in a CSV (header excluded)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def count_rois(run_dir: Path) -> int:
    crops_root = run_dir / ROI_SUBDIR
    if not crops_root.is_dir():
        return 0
    return sum(1 for _ in iter_roi_crops(crops_root))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_postprocess_log(run_dir: Path, event: str, **fields: object) -> None:
    """Append one line to ``atributos_por_roi.log`` inside ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"timestamp={_utc_now_iso()}", f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    line = " ".join(parts) + "\n"
    with open(run_dir / OUTPUT_LOG_NAME, "a", encoding="utf-8") as f:
        f.write(line)


def parse_postprocess_log(run_dir: Path) -> List[Dict[str, str]]:
    """Parse key=value lines from ``atributos_por_roi.log``."""
    path = run_dir / OUTPUT_LOG_NAME
    if not path.is_file():
        return []
    entries: List[Dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        item: Dict[str, str] = {}
        for part in line.split():
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            item[k] = v
        if item.get("event"):
            entries.append(item)
    return entries


def last_complete_postprocess(run_dir: Path) -> Optional[Dict[str, str]]:
    """Return the last ``END`` event with ``status=complete``, if any."""
    last: Optional[Dict[str, str]] = None
    for entry in parse_postprocess_log(run_dir):
        if entry.get("event") == "END" and entry.get("status") == "complete":
            last = entry
    return last


def atributos_csv_is_complete(run_dir: Path) -> bool:
    """True if ``atributos_por_roi.csv`` exists and its row count equals the ROI count."""
    csv_path = run_dir / OUTPUT_CSV_NAME
    if not csv_path.is_file():
        return False
    try:
        return count_csv_data_rows(csv_path) == count_rois(run_dir)
    except OSError:
        return False


def postprocess_already_done(run_dir: Path) -> bool:
    """True if the log records a complete run matching the current ROI count.

    Requires:
    - ``atributos_por_roi.log`` with an ``END status=complete`` whose ``n_rois``
      (or ``n_rows``) equals the current number of ROI crops;
    - ``atributos_por_roi.csv`` present with the same number of data rows.
    """
    done = last_complete_postprocess(run_dir)
    if done is None:
        return False
    n_rois = count_rois(run_dir)
    logged = done.get("n_rois") or done.get("n_rows")
    try:
        logged_n = int(logged) if logged is not None else -1
    except ValueError:
        logged_n = -1
    if logged_n != n_rois:
        return False
    return atributos_csv_is_complete(run_dir)


def process_run(
    run_dir: Path,
    *,
    images_root_override: Optional[Path] = None,
    border_override: Optional[int] = None,
    skip_existing: bool = False,
    verbose: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[Path, int, int, bool]:
    """Write ``atributos_por_roi.csv`` inside ``run_dir``.

    Returns ``(csv_path, n_rows, unmatched_feret, skipped)``.
    Raises :class:`ProcessingCancelled` if ``should_stop`` becomes true.
    """
    def _progress(msg: str) -> None:
        if progress is not None:
            progress(msg)
        elif verbose:
            print(msg, file=sys.stderr, flush=True)

    def _check_stop() -> None:
        if should_stop is not None and should_stop():
            raise ProcessingCancelled("Processamento interrompido pelo usuário.")

    run_dir = run_dir.resolve()
    crops_root = run_dir / ROI_SUBDIR
    if not crops_root.is_dir():
        raise FileNotFoundError(f"Missing {ROI_SUBDIR}/ under {run_dir}")

    out_csv = run_dir / OUTPUT_CSV_NAME
    if skip_existing and postprocess_already_done(run_dir):
        n_rows = count_csv_data_rows(out_csv)
        last = last_complete_postprocess(run_dir) or {}
        _progress(
            f"  skip existing: log {OUTPUT_LOG_NAME} marca completo "
            f"(fim={last.get('timestamp', '?')}, {n_rows} ROI(s))"
        )
        return out_csv, n_rows, 0, True

    n_rois_expected = count_rois(run_dir)
    append_postprocess_log(
        run_dir,
        "START",
        n_rois=n_rois_expected,
        csv=OUTPUT_CSV_NAME,
    )
    _progress(
        f"  {run_dir.name}: início registrado em {OUTPUT_LOG_NAME} "
        f"({n_rois_expected} ROI(s))"
    )

    try:
        result = _process_run_body(
            run_dir,
            crops_root=crops_root,
            out_csv=out_csv,
            images_root_override=images_root_override,
            border_override=border_override,
            verbose=verbose,
            progress=progress,
            should_stop=should_stop,
            _progress=_progress,
            _check_stop=_check_stop,
        )
    except ProcessingCancelled:
        append_postprocess_log(
            run_dir,
            "END",
            status="cancelled",
            n_rois=n_rois_expected,
            csv=OUTPUT_CSV_NAME,
        )
        raise
    except Exception as e:
        append_postprocess_log(
            run_dir,
            "END",
            status="error",
            n_rois=n_rois_expected,
            error=type(e).__name__,
            csv=OUTPUT_CSV_NAME,
        )
        raise

    _csv_path, n_rows, unmatched, skipped = result
    append_postprocess_log(
        run_dir,
        "END",
        status="complete",
        n_rois=n_rois_expected,
        n_rows=n_rows,
        unmatched_feret=unmatched,
        csv=OUTPUT_CSV_NAME,
    )
    _progress(
        f"  {run_dir.name}: fim registrado em {OUTPUT_LOG_NAME} "
        f"({n_rows} linha(s) em {OUTPUT_CSV_NAME})"
    )
    return result


def _process_run_body(
    run_dir: Path,
    *,
    crops_root: Path,
    out_csv: Path,
    images_root_override: Optional[Path],
    border_override: Optional[int],
    verbose: bool,
    progress: Optional[Callable[[str], None]],
    should_stop: Optional[Callable[[], bool]],
    _progress: Callable[[str], None],
    _check_stop: Callable[[], None],
) -> Tuple[Path, int, int, bool]:
    _check_stop()

    meta = read_run_metadata(run_dir)
    npz_root = Path(meta["npz_dir"]) if meta.get("npz_dir") not in (None, "", "None") else (
        run_dir / WATERSHED_SUBDIR
    )
    if not npz_root.is_dir():
        npz_root = run_dir / WATERSHED_SUBDIR

    images_root: Optional[Path] = None
    if images_root_override is not None:
        images_root = images_root_override
    elif meta.get("root") not in (None, "", "None"):
        images_root = Path(meta["root"])

    if border_override is not None:
        border = border_override
    else:
        try:
            border = int(meta.get("border", DEFAULT_BORDER))
        except ValueError:
            border = DEFAULT_BORDER

    feret_index = load_feret_index(run_dir / FERET_CSV_NAME)

    # Cache labels / gray / bbox index per source image
    labels_cache: Dict[Path, Optional[np.ndarray]] = {}
    gray_cache: Dict[Path, Optional[np.ndarray]] = {}
    bbox_index_cache: Dict[Path, Dict[Tuple[int, int], Tuple[int, int, int]]] = {}

    rows_out: List[Dict[str, object]] = []
    unmatched = 0
    crops = list(iter_roi_crops(crops_root))
    n_crops = len(crops)
    _progress(f"  {run_dir.name}: {n_crops} ROI(s) a processar…")
    report_every = max(1, n_crops // 20)  # ~5% steps
    for i, crop_path in enumerate(
        tqdm(crops, desc=run_dir.name, unit="roi", disable=not verbose), start=1
    ):
        if i == 1 or i % 25 == 0:
            _check_stop()
        parsed = parse_crop_name(crop_path)
        if parsed is None:
            continue
        image_stem, left, top = parsed
        rel_name = crop_path.relative_to(crops_root).as_posix()

        feret = feret_index.get(FeretKey(image_stem, left, top))
        if feret is None:
            unmatched += 1
            diam_px = ""
            diam_um = ""
            bbox_bottom: Optional[int] = None
            bbox_right: Optional[int] = None
        else:
            diam_px = "" if feret.diameter_px is None else f"{feret.diameter_px:.6f}"
            diam_um = "" if feret.diameter_um is None else f"{feret.diameter_um:.6f}"
            bbox_bottom = feret.bottom
            bbox_right = feret.right

        contrasts = {k: None for k in CONTRAST_COLUMNS}
        npz_path = (
            find_npz_for_crop(crop_path, crops_root, npz_root, image_stem)
            if npz_root.is_dir()
            else None
        )
        if npz_path is not None:
            if npz_path not in labels_cache:
                labels_cache[npz_path] = load_labels(npz_path)
                lab0 = labels_cache[npz_path]
                if lab0 is not None:
                    bbox_index_cache[npz_path] = bbox_top_left_index(lab0)
            labels = labels_cache[npz_path]
            if labels is not None:
                comp = lookup_component(
                    bbox_index_cache.get(npz_path, {}), labels, left, top
                )
                if comp is not None:
                    label_id, idx_bottom, idx_right = comp
                    # Prefer Feret CSV bbox; fall back to mask bbox from the index.
                    bottom = bbox_bottom if bbox_bottom is not None else idx_bottom
                    right = bbox_right if bbox_right is not None else idx_right
                    gray_full: Optional[np.ndarray] = None
                    src = find_source_image(
                        images_root, crops_root, crop_path, image_stem
                    )
                    if src is not None:
                        if src not in gray_cache:
                            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                            if img is None:
                                gray_cache[src] = None
                            else:
                                g = _to_gray_f32(img)
                                if g.shape[:2] != labels.shape:
                                    g = cv2.resize(
                                        g,
                                        (labels.shape[1], labels.shape[0]),
                                        interpolation=cv2.INTER_LINEAR,
                                    )
                                gray_cache[src] = g
                        gray_full = gray_cache[src]
                    crop_bgr = None
                    if gray_full is None:
                        crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
                    contrasts = contrasts_within_bbox(
                        labels,
                        label_id,
                        gray_full,
                        crop_bgr,
                        top=top,
                        left=left,
                        bottom=bottom,
                        right=right,
                        roi_border=border,
                    )

        row: Dict[str, object] = {
            "roi_image": rel_name,
            "source_stem": image_stem,
            "left": left,
            "top": top,
            "diameter_px": diam_px,
            "diameter_um": diam_um,
        }
        for col in CONTRAST_COLUMNS:
            val = contrasts.get(col)
            row[col] = "" if val is None else f"{val:.6f}"
        rows_out.append(row)

        if i == 1 or i == n_crops or i % report_every == 0:
            _progress(f"  {run_dir.name}: {i}/{n_crops} ROI(s)…")

    _check_stop()
    fieldnames = [
        "roi_image",
        "source_stem",
        "left",
        "top",
        "diameter_px",
        "diameter_um",
        *CONTRAST_COLUMNS,
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return out_csv, len(rows_out), unmatched, False


def process_paths(
    paths: Sequence[Path],
    *,
    images_root_override: Optional[Path] = None,
    border_override: Optional[int] = None,
    skip_existing: bool = False,
    verbose: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Tuple[Path, Path, int, int, bool]]:
    """Process each selected path (outputs root or single run)."""
    def _progress(msg: str) -> None:
        if progress is not None:
            progress(msg)
        elif verbose:
            print(msg, file=sys.stderr, flush=True)

    results: List[Tuple[Path, Path, int, int, bool]] = []
    for path in paths:
        runs = discover_run_dirs(path)
        if not runs:
            raise FileNotFoundError(
                f"Nenhuma pasta com {ROI_SUBDIR}/ encontrada em: {path}"
            )
        for run_dir in runs:
            if should_stop is not None and should_stop():
                raise ProcessingCancelled("Processamento interrompido pelo usuário.")
            _progress(f"Processando {run_dir} …")
            csv_path, n_rows, unmatched, skipped = process_run(
                run_dir,
                images_root_override=images_root_override,
                border_override=border_override,
                skip_existing=skip_existing,
                verbose=verbose,
                progress=progress,
                should_stop=should_stop,
            )
            if skipped:
                _progress(f"  → ignorado (já completo): {csv_path}")
            else:
                _progress(
                    f"  → {csv_path} ({n_rows} ROIs, {unmatched} sem Feret)"
                )
            results.append((run_dir, csv_path, n_rows, unmatched, skipped))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera atributos_por_roi.csv (nome do ROI, diâmetro de Feret, contrastes) "
            "para pastas de saída ou uma run específica."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPO / "outputs",
        help="Pasta outputs/ ou pasta de uma run (padrão: ./outputs).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Sobrescreve o 'root' do run_metadata.txt para localizar imagens originais.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=None,
        help="Borda usada nos crops (padrão: valor do metadata ou 10).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Não reprocessa uma run se atributos_por_roi.log registrar "
            "END status=complete com n_rois igual ao nº atual de ROIs "
            "e o CSV correspondente estiver completo."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    try:
        process_paths(
            [args.path],
            images_root_override=args.images_root,
            border_override=args.border,
            skip_existing=bool(args.skip_existing),
            verbose=bool(args.verbose),
        )
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
