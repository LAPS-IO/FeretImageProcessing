#!/usr/bin/env python3
"""PyQt5 front-end for :mod:`segment_and_extract_rois`.

Pick a **campaign** folder whose layout is ``campaign/date/frames/images``, tune
the parameters for each stage, and run the pipeline for every date. Outputs go
to ``<output_base>/<campaign>/<date>/``. By default, if that folder already
exists, only images that are not yet present in the output are processed.

    python3 pipeline_gui.py
"""

from __future__ import annotations

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


def list_date_folders(campaign: Path) -> list[Path]:
    """Return date subfolders of a campaign that contain at least one image."""
    dates: list[Path] = []
    for date_dir in sorted(
        p for p in campaign.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        has_image = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in date_dir.rglob("*")
        )
        if has_image:
            dates.append(date_dir)
    return dates


class PipelineWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Feret Image Processing - Segment + ROI + Feret")
        self.resize(820, 780)
        self._process: QProcess | None = None
        self._run_dir: Path | None = None
        self._date_queue: list[tuple[Path, Path]] = []
        self._date_index = 0
        self._campaign_name = ""

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

        self.campaign_edit = QLineEdit()
        self.campaign_edit.setPlaceholderText(
            "Campaign folder (campaign/date/frames/images)"
        )
        campaign_button = QPushButton("Browse…")
        campaign_button.clicked.connect(self._pick_campaign)
        campaign_row = QHBoxLayout()
        campaign_row.addWidget(self.campaign_edit, stretch=1)
        campaign_row.addWidget(campaign_button)
        form.addRow("Campaign folder:", self._wrap(campaign_row))

        self.output_edit = QLineEdit(str(REPO / "outputs"))
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._pick_output_base)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(output_button)
        form.addRow("Output base:", self._wrap(output_row))

        self.skip_existing_check = QCheckBox(
            "Skip images already present in outputs/<campaign>/<date> (default)"
        )
        self.skip_existing_check.setChecked(True)
        form.addRow("", self.skip_existing_check)

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
        self.depth_check.setChecked(True)
        depth_row = QHBoxLayout()
        depth_row.addWidget(self.depth_check)
        depth_row.addWidget(self.depth_spin, stretch=1)
        form.addRow("Depth filter:", self._wrap(depth_row))

        self.save_npz_check = QCheckBox(
            "Save .npz label maps (background_difference_watershed/)"
        )
        self.save_npz_check.setChecked(True)
        form.addRow("", self.save_npz_check)

        self.save_sbs_check = QCheckBox(
            "Save side-by-side preview (background_difference_watershed_side_by_side/)"
        )
        self.save_sbs_check.setChecked(True)
        form.addRow("", self.save_sbs_check)
        return group

    def _build_feret_group(self) -> QGroupBox:
        group = QGroupBox("Feret")
        form = QFormLayout(group)

        self.feret_check = QCheckBox("Compute Feret diameters (CSV in date folder)")
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
    def _pick_campaign(self) -> None:
        start = self.campaign_edit.text().strip() or str(REPO)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select campaign folder", start
        )
        if chosen:
            self.campaign_edit.setText(chosen)

    def _pick_output_base(self) -> None:
        start = self.output_edit.text().strip() or str(REPO)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select output base folder", start
        )
        if chosen:
            self.output_edit.setText(chosen)

    def _build_arguments(
        self, date_dir: Path, run_dir: Path, *, output_existed: bool
    ) -> list[str]:
        crops_dir = run_dir / "roi_crops"
        args = [str(PIPELINE_SCRIPT), str(date_dir)]
        args += ["--output", str(crops_dir), "--run-dir", str(run_dir)]
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
            args += ["--npz-dir", str(run_dir / "background_difference_watershed")]
        if self.save_sbs_check.isChecked():
            args += [
                "--sbs-dir",
                str(run_dir / "background_difference_watershed_side_by_side"),
            ]
        # Only skip when the date output folder already existed before this run.
        if self.skip_existing_check.isChecked() and output_existed:
            args.append("--skip-existing")
        args += [
            "--border",
            str(self.roi_border_spin.value()),
            "--max-coverage",
            f"{self.roi_max_coverage_spin.value():g}",
        ]
        if self.feret_check.isChecked():
            args += [
                "--feret-csv",
                str(run_dir / FERET_CSV_NAME),
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
        campaign_text = self.campaign_edit.text().strip()
        if not campaign_text:
            QMessageBox.warning(
                self, "Missing folder", "Please choose a campaign folder."
            )
            return
        campaign = Path(campaign_text)
        if not campaign.is_dir():
            QMessageBox.warning(
                self, "Invalid folder", f"Not a directory:\n{campaign}"
            )
            return
        if not PIPELINE_SCRIPT.is_file():
            QMessageBox.critical(
                self, "Missing script", f"Cannot find:\n{PIPELINE_SCRIPT}"
            )
            return

        dates = list_date_folders(campaign)
        if not dates:
            QMessageBox.warning(
                self,
                "No dates found",
                f"No date subfolders with images under:\n{campaign}",
            )
            return

        base_text = self.output_edit.text().strip()
        base = Path(base_text) if base_text else (REPO / "outputs")
        campaign_out = base / campaign.name
        try:
            campaign_out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(
                self,
                "Output error",
                f"Could not create campaign output folder:\n{campaign_out}\n\n{e}",
            )
            return

        self._campaign_name = campaign.name
        self._date_queue = [(date_dir, campaign_out / date_dir.name) for date_dir in dates]
        self._date_index = 0
        self.log_view.clear()
        self._append_log(
            f"Campaign: {campaign}\n"
            f"Dates: {len(self._date_queue)} → {campaign_out}/\n"
        )
        self._set_running(True)
        self._start_next_date()

    def _start_next_date(self) -> None:
        if self._date_index >= len(self._date_queue):
            self.status_label.setText(
                f"Finished successfully → {self._campaign_name}"
            )
            self._append_log("\n[campaign finished]\n")
            self._process = None
            self._run_dir = None
            self._set_running(False)
            return

        date_dir, run_dir = self._date_queue[self._date_index]
        self._run_dir = run_dir
        output_existed = run_dir.is_dir()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._append_log(f"\nERROR creating {run_dir}: {e}\n")
            self.status_label.setText(f"Failed → {run_dir.name}")
            self._process = None
            self._set_running(False)
            return

        args = self._build_arguments(date_dir, run_dir, output_existed=output_existed)
        label = f"{self._campaign_name}/{date_dir.name}"
        progress = f"{self._date_index + 1}/{len(self._date_queue)}"
        self._append_log(f"\n=== {label} ({progress}) → {run_dir} ===\n")
        if "--skip-existing" in args:
            self._append_log("(skip-existing enabled)\n")
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
        self.status_label.setText(f"Running… → {label} ({progress})")
        process.start()

    def _stop_pipeline(self) -> None:
        if self._process is None:
            return
        self.status_label.setText("Stopping…")
        # Drop remaining dates so finish handler does not continue.
        self._date_queue = self._date_queue[: self._date_index + 1]
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
        date_name = self._run_dir.name if self._run_dir is not None else ""
        self._append_log(
            f"\n[date finished: {date_name}, exit code {exit_code}]\n"
        )
        self._process = None
        if exit_code != 0:
            self.status_label.setText(
                f"Failed (exit code {exit_code}) → "
                f"{self._campaign_name}/{date_name}"
            )
            self._set_running(False)
            return
        self._date_index += 1
        self._start_next_date()

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
            self._date_queue = []
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
