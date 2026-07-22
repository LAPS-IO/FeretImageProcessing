#!/usr/bin/env python3
"""PyQt5 front-end for :mod:`run_full_pipeline`.

Pick the parent folder that holds the image subfolders, tune the parameters for
each stage (segmentation, Feret, ROI extraction), and run the whole pipeline
while watching the live log. The actual work is delegated to
``run_full_pipeline.py`` through a :class:`QProcess` so the UI stays responsive
and the run can be stopped at any time.

    python3 pipeline_gui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

REPO = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO / "segment_and_extract_rois.py"
FERET_CSV_NAME = "feret_diameters.csv"
RUN_RE = re.compile(r"^run(\d+)$")


class PipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Feret Image Processing - Segment + ROI + Feret")
        self.resize(820, 760)
        self._process: QProcess | None = None
        self._run_dir: Path | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_io_group())
        root_layout.addWidget(self._build_segmentation_group())
        root_layout.addWidget(self._build_feret_group())
        root_layout.addWidget(self._build_roi_group())
        root_layout.addLayout(self._build_action_bar())
        root_layout.addWidget(self._build_log_view(), stretch=1)

    # ------------------------------------------------------------------ UI
    def _build_io_group(self) -> QGroupBox:
        group = QGroupBox("Input / Output")
        form = QFormLayout(group)

        self.images_edit = QLineEdit()
        self.images_edit.setPlaceholderText(
            "Parent folder containing image subfolders"
        )
        images_button = QPushButton("Browse…")
        images_button.clicked.connect(self._pick_images_root)
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_edit, stretch=1)
        images_row.addWidget(images_button)
        form.addRow("Images folder:", self._wrap(images_row))

        self.output_edit = QLineEdit(str(REPO / "outputs"))
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._pick_output_base)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(output_button)
        form.addRow("Output base:", self._wrap(output_row))

        self.verbose_check = QCheckBox("Verbose output")
        self.verbose_check.setChecked(True)
        form.addRow("", self.verbose_check)
        return group

    def _build_segmentation_group(self) -> QGroupBox:
        group = QGroupBox("Segmentation")
        form = QFormLayout(group)

        self.rolling_width_spin = QSpinBox()
        self.rolling_width_spin.setRange(1, 999)
        self.rolling_width_spin.setValue(5)
        form.addRow("Rolling width:", self.rolling_width_spin)

        self.watershed_threshold_spin = QSpinBox()
        self.watershed_threshold_spin.setRange(0, 255)
        self.watershed_threshold_spin.setValue(4)
        form.addRow("Watershed threshold:", self.watershed_threshold_spin)

        self.watershed_min_area_spin = QSpinBox()
        self.watershed_min_area_spin.setRange(0, 10_000_000)
        self.watershed_min_area_spin.setValue(250)
        form.addRow("Watershed min area (px):", self.watershed_min_area_spin)

        self.circular_morph_spin = QSpinBox()
        self.circular_morph_spin.setRange(1, 999)
        self.circular_morph_spin.setValue(11)
        form.addRow("Circular morph radius:", self.circular_morph_spin)

        self.depth_check = QCheckBox("Filter by minimum depth")
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(-10_000_000, 10_000_000)
        self.depth_spin.setValue(500)
        self.depth_spin.setEnabled(False)
        self.depth_check.toggled.connect(self.depth_spin.setEnabled)
        depth_row = QHBoxLayout()
        depth_row.addWidget(self.depth_check)
        depth_row.addWidget(self.depth_spin, stretch=1)
        form.addRow("Depth filter:", self._wrap(depth_row))

        self.save_npz_check = QCheckBox(
            "Save .npz label maps (background_difference_watershed/)"
        )
        form.addRow("", self.save_npz_check)

        self.save_sbs_check = QCheckBox(
            "Save side-by-side preview (background_difference_watershed_side_by_side/)"
        )
        form.addRow("", self.save_sbs_check)
        return group

    def _build_feret_group(self) -> QGroupBox:
        group = QGroupBox("Feret")
        form = QFormLayout(group)

        self.feret_check = QCheckBox("Compute Feret diameters (CSV in crops folder)")
        self.feret_check.setChecked(True)
        form.addRow("", self.feret_check)

        self.um_per_pixel_spin = QDoubleSpinBox()
        self.um_per_pixel_spin.setDecimals(4)
        self.um_per_pixel_spin.setRange(0.0001, 1_000_000.0)
        self.um_per_pixel_spin.setValue(13.8)
        form.addRow("Micrometers per pixel:", self.um_per_pixel_spin)

        self.edge_strip_spin = QSpinBox()
        self.edge_strip_spin.setRange(0, 10_000)
        self.edge_strip_spin.setValue(2)
        form.addRow("Edge strip (px):", self.edge_strip_spin)
        return group

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("ROI extraction")
        form = QFormLayout(group)

        self.roi_border_spin = QSpinBox()
        self.roi_border_spin.setRange(0, 10_000)
        self.roi_border_spin.setValue(10)
        form.addRow("ROI border (px):", self.roi_border_spin)

        self.roi_max_coverage_spin = QDoubleSpinBox()
        self.roi_max_coverage_spin.setDecimals(4)
        self.roi_max_coverage_spin.setRange(0.0001, 1.0)
        self.roi_max_coverage_spin.setSingleStep(0.01)
        self.roi_max_coverage_spin.setValue(0.98)
        form.addRow("ROI max coverage:", self.roi_max_coverage_spin)
        return group

    def _build_action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.run_button = QPushButton("Run pipeline")
        self.run_button.clicked.connect(self._start_pipeline)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_pipeline)
        bar.addWidget(self.status_label, stretch=1)
        bar.addWidget(self.run_button)
        bar.addWidget(self.stop_button)
        return bar

    def _build_log_view(self) -> QPlainTextEdit:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Monospace", 9))
        self.log_view.setPlaceholderText("Pipeline output appears here…")
        return self.log_view

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        container = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(layout)
        return container

    # -------------------------------------------------------------- actions
    def _pick_images_root(self) -> None:
        start = self.images_edit.text().strip() or str(REPO)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select images parent folder", start
        )
        if chosen:
            self.images_edit.setText(chosen)

    def _pick_output_base(self) -> None:
        start = self.output_edit.text().strip() or str(REPO)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select crops output folder", start
        )
        if chosen:
            self.output_edit.setText(chosen)

    def _allocate_run_dir(self, base: Path) -> Path:
        """Create and return ``base/run{N}`` with N = highest existing run number + 1."""
        base.mkdir(parents=True, exist_ok=True)
        last = -1
        for child in base.iterdir():
            if child.is_dir():
                match = RUN_RE.match(child.name)
                if match is not None:
                    last = max(last, int(match.group(1)))
        run_dir = base / f"run{last + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_arguments(self) -> list[str]:
        images_root = self.images_edit.text().strip()
        assert self._run_dir is not None
        crops_dir = self._run_dir / "roi_crops"
        args = [str(PIPELINE_SCRIPT), images_root]
        args += ["--output", str(crops_dir), "--run-dir", str(self._run_dir)]
        args += [
            "--rolling",
            "--rolling-width",
            str(self.rolling_width_spin.value()),
            "--watershed-t",
            str(self.watershed_threshold_spin.value()),
            "--watershed-min-area",
            str(self.watershed_min_area_spin.value()),
            "--circular-morph",
            str(self.circular_morph_spin.value()),
        ]
        if self.depth_check.isChecked():
            args += ["--min-depth", str(self.depth_spin.value())]
        else:
            args.append("--no-depth-filter")
        if self.save_npz_check.isChecked():
            npz_dir = self._run_dir / "background_difference_watershed"
            args += ["--npz-dir", str(npz_dir)]
        if self.save_sbs_check.isChecked():
            sbs_dir = self._run_dir / "background_difference_watershed_side_by_side"
            args += ["--sbs-dir", str(sbs_dir)]
        args += [
            "--border",
            str(self.roi_border_spin.value()),
            "--max-coverage",
            f"{self.roi_max_coverage_spin.value():g}",
        ]
        if self.feret_check.isChecked():
            feret_csv = self._run_dir / FERET_CSV_NAME
            args += [
                "--feret-csv",
                str(feret_csv),
                "--um-per-pixel",
                f"{self.um_per_pixel_spin.value():g}",
                "--edge-strip",
                str(self.edge_strip_spin.value()),
            ]
        if self.verbose_check.isChecked():
            args.append("--verbose")
        return args

    def _start_pipeline(self) -> None:
        if self._process is not None:
            return
        images_root = self.images_edit.text().strip()
        if not images_root:
            QMessageBox.warning(
                self, "Missing folder", "Please choose an images parent folder."
            )
            return
        if not Path(images_root).is_dir():
            QMessageBox.warning(
                self, "Invalid folder", f"Not a directory:\n{images_root}"
            )
            return
        if not PIPELINE_SCRIPT.is_file():
            QMessageBox.critical(
                self, "Missing script", f"Cannot find:\n{PIPELINE_SCRIPT}"
            )
            return

        base_text = self.output_edit.text().strip()
        base = Path(base_text) if base_text else (REPO / "outputs")
        try:
            self._run_dir = self._allocate_run_dir(base)
        except OSError as e:
            QMessageBox.critical(
                self, "Output error", f"Could not create run folder in:\n{base}\n\n{e}"
            )
            return

        args = self._build_arguments()
        self.log_view.clear()
        self._append_log(f"Output folder: {self._run_dir}\n")
        self._append_log(f"$ {sys.executable} {' '.join(args)}\n")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(REPO))
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process

        self._set_running(True)
        self.status_label.setText(f"Running… → {self._run_dir.name}")
        process.start()

    def _stop_pipeline(self) -> None:
        if self._process is None:
            return
        self.status_label.setText("Stopping…")
        self._process.kill()

    # --------------------------------------------------------------- signals
    def _read_stdout(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._append_log(data)

    def _read_stderr(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        self._append_log(data)

    def _on_finished(self, exit_code: int, _status: object) -> None:
        run_name = self._run_dir.name if self._run_dir is not None else ""
        if exit_code == 0:
            self.status_label.setText(f"Finished successfully → {run_name}")
        else:
            self.status_label.setText(f"Failed (exit code {exit_code}) → {run_name}")
        self._append_log(f"\n[process finished with exit code {exit_code}]\n")
        self._process = None
        self._set_running(False)

    def _on_error(self, _error: object) -> None:
        if self._process is None:
            return
        self._append_log("\n[failed to start process]\n")

    # ----------------------------------------------------------------- helpers
    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(self.log_view.textCursor().End)
        self.log_view.insertPlainText(text)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._process is not None:
            self._process.kill()
            self._process.waitForFinished(3000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontUseNativeMenuBar, False)
    window = PipelineWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
