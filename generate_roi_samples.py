#!/usr/bin/env python3
"""Generate random ROI sample sheets for each run folder.

For every directory that contains ``roi_crops/`` (and preferably
``atributos_por_roi.csv``), writes ``ROIs_samples/page_XX.png`` with a grid of
random ROI thumbnails, attribute text, and a micrometre scale bar.

Defaults: 180 ROIs, 18 per page (3 rows × 6 cols).

Examples::

    python3 generate_roi_samples.py
    python3 generate_roi_samples.py outputs/teste/campanha --seed 0
    python3 generate_roi_samples.py outputs -n 180 --per-page 18 --cols 6
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# OpenCV's bundled Qt plugins break interactive Matplotlib backends (QtAgg).
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

# cv2 may set QT_QPA_PLATFORM_PLUGIN_PATH on import — drop it again.
_cv2_qt = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
if "cv2" in _cv2_qt.replace("\\", "/"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

try:
    import matplotlib

    matplotlib.use("Agg")  # headless; must run before pyplot import
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.patches import Rectangle
except ImportError:  # pragma: no cover
    print(
        "matplotlib é necessário para este script (veja requirements.txt).",
        file=sys.stderr,
    )
    sys.exit(1)

REPO = Path(__file__).resolve().parent
ROI_SUBDIR = "roi_crops"
ATTR_CSV = "atributos_por_roi.csv"
SAMPLES_DIR = "ROIs_samples"
DEFAULT_UM_PER_PIXEL = 13.8

ATTR_FERET = (
    ("diameter_um", "Ø µm"),
    ("diameter_px", "Ø px"),
)

ATTR_CONTRAST = (
    ("contrast_rms", "RMS"),
    ("contrast_michelson", "Mich"),
    ("contrast_range", "Range"),
    ("contrast_cv", "CV"),
    ("contrast_weber", "Weber"),
    ("contrast_mean_diff", "Δμ"),
)

ATTR_KEYS = tuple(k for k, _ in ATTR_FERET) + tuple(k for k, _ in ATTR_CONTRAST)

ATTR_FONTSIZE = 6.25  # 5.0 × 1.25

# Percentile color scale: low → mid → high
COLOR_LOW = np.array([0.85, 0.12, 0.12])  # red
COLOR_MID = np.array([1.00, 0.55, 0.05])  # orange (~P50)
COLOR_HIGH = np.array([0.13, 0.65, 0.25])  # green


def discover_run_dirs(path: Path) -> List[Path]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    if (path / ROI_SUBDIR).is_dir():
        return [path]
    runs: List[Path] = []
    seen: set[Path] = set()
    for child in sorted(path.rglob(ROI_SUBDIR)):
        if child.is_dir():
            run = child.parent.resolve()
            if run not in seen:
                seen.add(run)
                runs.append(run)
    return runs


def read_um_per_pixel(run_dir: Path) -> float:
    meta = run_dir / "run_metadata.txt"
    if not meta.is_file():
        return DEFAULT_UM_PER_PIXEL
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        if "um_per_pixel" in line and (":" in line or "=" in line):
            sep = ":" if ":" in line else "="
            _, _, val = line.partition(sep)
            try:
                return float(val.strip())
            except ValueError:
                pass
    return DEFAULT_UM_PER_PIXEL


def load_attributes(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Map ``roi_image`` relative path → attribute row."""
    out: Dict[str, Dict[str, str]] = {}
    if not csv_path.is_file():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("roi_image") or "").strip()
            if key:
                out[key] = row
    return out


def list_roi_files(crops_root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files: List[Path] = []
    for p in sorted(crops_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def choose_scale_bar_um(width_px: int, um_per_pixel: float) -> float:
    """Pick a nice round scale length (~25–40% of the thumbnail width)."""
    if width_px < 2 or um_per_pixel <= 0:
        return 100.0
    target_um = width_px * um_per_pixel * 0.33
    if target_um <= 0:
        return 100.0
    exp = math.floor(math.log10(target_um))
    base = 10**exp
    for mult in (1, 2, 5, 10, 20, 50):
        cand = mult * base
        if cand >= target_um * 0.5:
            return float(cand)
    return float(10 * base)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_to_square(rgb: np.ndarray, fill: int = 230) -> Tuple[np.ndarray, int, int]:
    """Pad RGB image to a square canvas; return ``(canvas, y0, x0)`` of the content."""
    h, w = rgb.shape[:2]
    side = max(h, w, 1)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = rgb
    return canvas, y0, x0


def _fmt_attr_value(key: str, raw: str) -> str:
    raw = raw.strip()
    if raw == "":
        return "—"
    try:
        v = float(raw)
        if key in ("diameter_px", "contrast_range", "diameter_um"):
            return f"{v:.1f}"
        return f"{v:.3g}"
    except ValueError:
        return raw


def _parse_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build_attr_scales(
    attr_rows: Sequence[Dict[str, str]],
) -> Dict[str, np.ndarray]:
    """Sorted value arrays per attribute (for percentile lookup)."""
    buckets: Dict[str, List[float]] = {k: [] for k in ATTR_KEYS}
    for row in attr_rows:
        for key in ATTR_KEYS:
            v = _parse_float(row.get(key))
            if v is not None:
                buckets[key].append(v)
    return {k: np.sort(np.asarray(vs, dtype=np.float64)) for k, vs in buckets.items()}


def percentile_rank(value: float, sorted_vals: np.ndarray) -> float:
    """Empirical CDF in [0, 1] (fraction of values ≤ ``value``)."""
    if sorted_vals.size == 0:
        return 0.5
    return float(np.searchsorted(sorted_vals, value, side="right") / sorted_vals.size)


def color_from_percentile(p: float) -> Tuple[float, float, float]:
    """Red (low) → orange (P50) → green (high)."""
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.5:
        t = p / 0.5
        rgb = COLOR_LOW * (1.0 - t) + COLOR_MID * t
    else:
        t = (p - 0.5) / 0.5
        rgb = COLOR_MID * (1.0 - t) + COLOR_HIGH * t
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def attr_color(
    key: str, raw: object, scales: Dict[str, np.ndarray]
) -> Tuple[float, float, float]:
    v = _parse_float(raw)
    if v is None:
        return (0.35, 0.35, 0.35)
    return color_from_percentile(percentile_rank(v, scales.get(key, np.array([]))))


def draw_color_legend(fig, *, y: float = 0.915) -> None:
    """Small red→orange→green bar under the page header."""
    ax = fig.add_axes([0.32, y - 0.012, 0.36, 0.012])
    colors = np.array([color_from_percentile(p) for p in np.linspace(0, 1, 256)])
    strip = colors.reshape(1, -1, 3)
    ax.imshow(strip, aspect="auto", extent=[0, 1, 0, 1], origin="lower")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["baixo", "P50", "alto"], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    fig.text(
        0.50,
        y + 0.008,
        "cor dos atributos (percentil na campanha): vermelho → laranja → verde",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333",
    )

def draw_panel(
    ax,
    img_bgr: np.ndarray,
    title: str,
    row: Optional[Dict[str, str]],
    scales: Dict[str, np.ndarray],
    um_per_pixel: float,
) -> None:
    rgb = bgr_to_rgb(img_bgr)
    h0, w0 = rgb.shape[:2]
    canvas, y0, x0 = pad_to_square(rgb)
    side = canvas.shape[0]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_anchor("N")

    img_bottom = 0.36
    ax.imshow(
        canvas,
        origin="upper",
        extent=(0.04, 0.96, img_bottom, 0.98),
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )

    bar_um = choose_scale_bar_um(w0, um_per_pixel)
    bar_px = bar_um / um_per_pixel
    content_w = 0.92
    px_to_ax = content_w / side
    bar_ax = bar_px * px_to_ax
    margin_ax = max(0.02, 0.04 * content_w)
    roi_bottom_ax = img_bottom + (side - (y0 + h0)) / side * (0.98 - img_bottom)
    x_bar0 = 0.04 + (x0 / side) * content_w + margin_ax
    x_bar1 = min(0.96 - margin_ax, x_bar0 + bar_ax)
    y_bar = roi_bottom_ax + 0.02 * (0.98 - img_bottom)
    ax.plot(
        [x_bar0, x_bar1],
        [y_bar, y_bar],
        color="white",
        lw=1.8,
        solid_capstyle="butt",
        zorder=5,
        clip_on=True,
    )
    ax.plot(
        [x_bar0, x_bar1],
        [y_bar, y_bar],
        color="black",
        lw=0.7,
        solid_capstyle="butt",
        zorder=6,
        clip_on=True,
    )
    ax.text(
        (x_bar0 + x_bar1) / 2,
        y_bar + 0.015,
        f"{bar_um:g} µm",
        color="white",
        fontsize=5.5,
        ha="center",
        va="bottom",
        zorder=7,
        path_effects=[pe.withStroke(linewidth=1.8, foreground="black")],
        clip_on=True,
        transform=ax.transData,
    )

    ax.set_title(title, fontsize=5.2, pad=4, loc="left", family="monospace")

    # Attribute headers
    y0_txt = img_bottom - 0.02
    ax.text(
        0.04,
        y0_txt,
        "Feret",
        transform=ax.transAxes,
        fontsize=ATTR_FONTSIZE,
        va="top",
        ha="left",
        family="monospace",
        color="#222222",
        fontweight="bold",
        clip_on=True,
    )
    ax.text(
        0.50,
        y0_txt,
        "Contraste",
        transform=ax.transAxes,
        fontsize=ATTR_FONTSIZE,
        va="top",
        ha="left",
        family="monospace",
        color="#222222",
        fontweight="bold",
        clip_on=True,
    )

    n_lines = max(len(ATTR_FERET), len(ATTR_CONTRAST))
    dy = 0.045
    for i in range(n_lines):
        y = y0_txt - (i + 1) * dy
        if i < len(ATTR_FERET):
            key, label = ATTR_FERET[i]
            raw = (row or {}).get(key, "")
            txt = f"{label}: {_fmt_attr_value(key, raw or '')}"
            col = attr_color(key, raw, scales) if row else (0.35, 0.35, 0.35)
            ax.text(
                0.04,
                y,
                txt,
                transform=ax.transAxes,
                fontsize=ATTR_FONTSIZE,
                va="top",
                ha="left",
                family="monospace",
                color=col,
                clip_on=True,
            )
        if i < len(ATTR_CONTRAST):
            key, label = ATTR_CONTRAST[i]
            raw = (row or {}).get(key, "")
            txt = f"{label}: {_fmt_attr_value(key, raw or '')}"
            col = attr_color(key, raw, scales) if row else (0.35, 0.35, 0.35)
            ax.text(
                0.50,
                y,
                txt,
                transform=ax.transAxes,
                fontsize=ATTR_FONTSIZE,
                va="top",
                ha="left",
                family="monospace",
                color=col,
                clip_on=True,
            )

    ax.add_patch(
        Rectangle(
            (0.04, img_bottom),
            0.92,
            0.98 - img_bottom,
            fill=False,
            edgecolor="#444444",
            linewidth=0.6,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
    )


def render_page(
    items: Sequence[Tuple[Path, Optional[Dict[str, str]], str]],
    out_path: Path,
    *,
    um_per_pixel: float,
    page_index: int,
    n_pages: int,
    cols: int,
    rows: int,
    run_name: str,
    scales: Dict[str, np.ndarray],
) -> None:
    fig_w = max(14.0, 2.5 * cols)
    fig_h = max(10.5, 3.9 * rows)
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_w, fig_h), constrained_layout=False
    )
    axes_flat = np.atleast_1d(axes).ravel()

    fig.text(
        0.5,
        0.985,
        f"{run_name}  —  ROI samples  (página {page_index}/{n_pages})",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.958,
        f"escala espacial: {um_per_pixel:g} µm/px  |  grid {rows}×{cols}",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
    )
    draw_color_legend(fig, y=0.928)

    for i, ax in enumerate(axes_flat):
        if i >= len(items):
            ax.set_axis_off()
            continue
        path, row, rel = items[i]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"falha ao ler\n{path.name}",
                ha="center",
                va="center",
                fontsize=7,
            )
            continue
        short = Path(rel).name
        if len(short) > 48:
            short = short[:22] + "…" + short[-22:]
        draw_panel(ax, img, short, row, scales, um_per_pixel)

    plt.subplots_adjust(
        left=0.03,
        right=0.98,
        top=0.88,
        bottom=0.06,
        wspace=0.25,
        hspace=0.18,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_run(
    run_dir: Path,
    *,
    n_samples: int,
    per_page: int,
    cols: int,
    rows: Optional[int] = None,
    seed: Optional[int],
    verbose: bool,
) -> Path:
    crops = run_dir / ROI_SUBDIR
    if not crops.is_dir():
        raise FileNotFoundError(f"Missing {ROI_SUBDIR}/ in {run_dir}")

    files = list_roi_files(crops)
    if not files:
        raise FileNotFoundError(f"No ROI images under {crops}")

    attrs = load_attributes(run_dir / ATTR_CSV)
    scales = build_attr_scales(list(attrs.values()))
    um_pp = read_um_per_pixel(run_dir)

    rng = random.Random(seed)
    take = min(n_samples, len(files))
    chosen = rng.sample(files, take)

    items: List[Tuple[Path, Optional[Dict[str, str]], str]] = []
    for p in chosen:
        rel = p.relative_to(crops).as_posix()
        items.append((p, attrs.get(rel), rel))

    if rows is None:
        rows = max(1, math.ceil(per_page / cols))
    grid_capacity = cols * rows
    if grid_capacity < 1:
        raise ValueError("cols*rows must be >= 1")
    per_page = min(per_page, grid_capacity)

    n_pages = max(1, math.ceil(len(items) / per_page))
    out_dir = run_dir / SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"{run_dir.name}: {len(files)} ROIs → amostrando {take}; "
            f"{n_pages} página(s) {rows}×{cols} em {out_dir} (µm/px={um_pp:g})",
            flush=True,
        )

    for page in range(n_pages):
        chunk = items[page * per_page : (page + 1) * per_page]
        out_path = out_dir / f"page_{page + 1:02d}.png"
        render_page(
            chunk,
            out_path,
            um_per_pixel=um_pp,
            page_index=page + 1,
            n_pages=n_pages,
            cols=cols,
            rows=rows,
            run_name=run_dir.name,
            scales=scales,
        )
        if verbose:
            print(f"  wrote {out_path} ({len(chunk)} ROIs)", flush=True)

    index_path = out_dir / "sample_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "slot", "roi_image"])
        for i, (_p, _row, rel) in enumerate(items):
            page = i // per_page + 1
            slot = i % per_page + 1
            w.writerow([page, slot, rel])
    if verbose:
        print(f"  wrote {index_path}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera folhas de amostra (ROIs_samples/) com ROIs aleatórios, "
            "miniaturas, atributos e barra de escala."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPO / "outputs",
        help="Pasta outputs/ ou uma run com roi_crops/ (padrão: ./outputs).",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=180, help="ROIs por run (padrão 180).")
    parser.add_argument(
        "--per-page",
        type=int,
        default=18,
        help="ROIs por página (padrão 18).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Colunas no grid (padrão 6 → 3×6 para 18/página).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Linhas no grid (padrão: per-page/cols, ex. 3 para 18/página com 6 cols).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semente do RNG.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.n_samples < 1 or args.per_page < 1 or args.cols < 1:
        print("n-samples, per-page e cols devem ser >= 1", file=sys.stderr)
        sys.exit(1)
    if args.rows is not None and args.rows < 1:
        print("--rows deve ser >= 1", file=sys.stderr)
        sys.exit(1)
    if args.rows is None and args.per_page % args.cols != 0:
        print(
            f"Aviso: per-page={args.per_page} não é múltiplo de cols={args.cols}; "
            f"usando {math.ceil(args.per_page / args.cols)} linhas.",
            file=sys.stderr,
        )

    try:
        runs = discover_run_dirs(args.path)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
    if not runs:
        print(f"Nenhuma pasta com {ROI_SUBDIR}/ em {args.path}", file=sys.stderr)
        sys.exit(1)

    for run in runs:
        try:
            process_run(
                run,
                n_samples=int(args.n_samples),
                per_page=int(args.per_page),
                cols=int(args.cols),
                rows=None if args.rows is None else int(args.rows),
                seed=args.seed,
                verbose=True,
            )
        except Exception as e:
            print(f"Erro em {run}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
