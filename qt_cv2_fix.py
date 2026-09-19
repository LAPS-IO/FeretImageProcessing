"""Avoid OpenCV's bundled Qt plugins breaking PyQt5 GUIs.

``opencv-python`` ships Qt plugins under ``cv2/qt/plugins`` and, when imported,
may set ``QT_QPA_PLATFORM_PLUGIN_PATH`` to that path. PyQt5 then fails to load
``xcb`` (``Could not load the Qt platform plugin "xcb"``).

Call :func:`prefer_pyqt5_platform_plugins` **before** creating ``QApplication``,
and again **after** importing any module that pulls in ``cv2``.
"""

from __future__ import annotations

import os
from pathlib import Path


def prefer_pyqt5_platform_plugins() -> None:
    """Point Qt at PyQt5's plugin tree (or clear a cv2-hijacked path)."""
    # Always drop OpenCV's plugin dir — it is incompatible with PyQt5's xcb.
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "cv2" in current.replace("\\", "/"):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    try:
        import PyQt5
    except ImportError:
        return

    root = Path(PyQt5.__file__).resolve().parent
    for candidate in (
        root / "Qt5" / "plugins",
        root / "Qt" / "plugins",
    ):
        if (candidate / "platforms").is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(candidate)
            return
