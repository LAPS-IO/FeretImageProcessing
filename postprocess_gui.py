#!/usr/bin/env python3
"""Interface PyQt5 para pós-processar runs e gerar ``atributos_por_roi.csv``.

O usuário escolhe a pasta ``outputs`` (padrão ``./outputs``) ou uma pasta de
run específica. Para cada run encontrada (com ``roi_crops/``), o programa
escreve ``atributos_por_roi.csv`` com o nome do ROI, o diâmetro de Feret
(inferido de ``feret_diameters.csv``) e várias medidas de contraste a partir
das segmentações ``.npz``.

    python3 postprocess_gui.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# OpenCV must not win the Qt plugin path before the GUI starts.
from qt_cv2_fix import prefer_pyqt5_platform_plugins

prefer_pyqt5_platform_plugins()

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
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
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

prefer_pyqt5_platform_plugins()

from atributos_por_roi import (
    OUTPUT_CSV_NAME,
    OUTPUT_LOG_NAME,
    ProcessingCancelled,
    discover_run_dirs,
    process_paths,
)

prefer_pyqt5_platform_plugins()

REPO = Path(__file__).resolve().parent


class Worker(QObject):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        path: Path,
        images_root: Path | None,
        border: int | None,
        skip_existing: bool,
        verbose: bool,
    ) -> None:
        super().__init__()
        self.path = path
        self.images_root = images_root
        self.border = border
        self.skip_existing = skip_existing
        self.verbose = verbose
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            results = process_paths(
                [self.path],
                images_root_override=self.images_root,
                border_override=self.border,
                skip_existing=self.skip_existing,
                verbose=False,
                progress=lambda msg: self.log.emit(msg + "\n"),
                should_stop=lambda: self._stop,
            )
            lines = []
            n_skipped = 0
            n_done = 0
            for run_dir, csv_path, n_rows, unmatched, skipped in results:
                if skipped:
                    n_skipped += 1
                    msg = (
                        f"{run_dir.name}: ignorado ({OUTPUT_LOG_NAME} marca completo, "
                        f"{n_rows} ROI(s))"
                    )
                else:
                    n_done += 1
                    msg = (
                        f"{run_dir.name}: {n_rows} ROI(s) → {csv_path.name}"
                        f" ({unmatched} sem match de Feret)"
                    )
                lines.append(msg)
                self.log.emit(msg + "\n")
            summary = (
                f"Concluído: {len(results)} run(s) "
                f"({n_done} processada(s), {n_skipped} ignorada(s))."
            )
            if lines:
                summary += "\n" + "\n".join(lines)
            self.finished_ok.emit(summary)
        except ProcessingCancelled as e:
            self.log.emit(f"\n{e}\n")
            self.cancelled.emit(str(e))
        except Exception as e:
            self.log.emit(traceback.format_exc() + "\n")
            self.failed.emit(str(e))


class PostprocessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pós-processamento — atributos por ROI")
        self.resize(720, 560)
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_io_group())
        layout.addWidget(self._build_options_group())
        layout.addLayout(self._build_action_bar())
        layout.addWidget(self._build_log_view(), stretch=1)

    def _build_io_group(self) -> QGroupBox:
        group = QGroupBox("Pasta a processar")
        form = QFormLayout(group)

        self.mode_outputs = QRadioButton("Pasta de outputs (todas as runs)")
        self.mode_run = QRadioButton("Pasta de uma run específica")
        self.mode_outputs.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_outputs)
        self._mode_group.addButton(self.mode_run)
        mode_row = QVBoxLayout()
        mode_row.addWidget(self.mode_outputs)
        mode_row.addWidget(self.mode_run)
        form.addRow("Modo:", self._wrap_v(mode_row))

        self.path_edit = QLineEdit(str(REPO / "outputs"))
        browse = QPushButton("Procurar…")
        browse.clicked.connect(self._pick_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse)
        form.addRow("Caminho:", self._wrap(path_row))

        hint = QLabel(
            "A pasta de outputs (padrão ./outputs) é varrida em busca de "
            "subpastas com roi_crops/. Uma run deve conter roi_crops/, "
            "feret_diameters.csv e background_difference_watershed/."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        form.addRow("", hint)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Opções")
        form = QFormLayout(group)

        self.images_edit = QLineEdit()
        self.images_edit.setPlaceholderText(
            "Opcional — sobrescreve o root do run_metadata.txt"
        )
        images_btn = QPushButton("Procurar…")
        images_btn.clicked.connect(self._pick_images_root)
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_edit, stretch=1)
        images_row.addWidget(images_btn)
        form.addRow("Pasta das imagens originais:", self._wrap(images_row))

        self.border_check = QCheckBox("Usar borda personalizada nos crops")
        self.border_spin = QSpinBox()
        self.border_spin.setRange(0, 10_000)
        self.border_spin.setValue(10)
        self.border_spin.setEnabled(False)
        self.border_check.toggled.connect(self.border_spin.setEnabled)
        border_row = QHBoxLayout()
        border_row.addWidget(self.border_check)
        border_row.addWidget(self.border_spin)
        form.addRow("Borda da ROI:", self._wrap(border_row))

        self.skip_existing_check = QCheckBox(
            f"Ignorar runs já concluídas ({OUTPUT_LOG_NAME} + CSV completo)"
        )
        self.skip_existing_check.setChecked(True)
        form.addRow("", self.skip_existing_check)

        self.verbose_check = QCheckBox("Saída detalhada no log")
        self.verbose_check.setChecked(True)
        form.addRow("", self.verbose_check)
        return group

    def _build_action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.status_label = QLabel("Pronto")
        self.run_button = QPushButton("Gerar atributos_por_roi.csv")
        self.run_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Parar")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        bar.addWidget(self.status_label, stretch=1)
        bar.addWidget(self.run_button)
        bar.addWidget(self.stop_button)
        return bar

    def _build_log_view(self) -> QPlainTextEdit:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Monospace", 9))
        self.log_view.setPlaceholderText("Progresso e resultados aparecem aqui…")
        return self.log_view

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    @staticmethod
    def _wrap_v(layout: QVBoxLayout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _pick_path(self) -> None:
        start = self.path_edit.text().strip() or str(REPO / "outputs")
        title = (
            "Selecionar pasta de outputs"
            if self.mode_outputs.isChecked()
            else "Selecionar pasta da run"
        )
        chosen = QFileDialog.getExistingDirectory(self, title, start)
        if chosen:
            self.path_edit.setText(chosen)

    def _pick_images_root(self) -> None:
        start = self.images_edit.text().strip() or str(REPO)
        chosen = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta das imagens originais", start
        )
        if chosen:
            self.images_edit.setText(chosen)

    def _start(self) -> None:
        if self._thread is not None:
            return
        text = self.path_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Pasta ausente", "Escolha uma pasta.")
            return
        path = Path(text)
        if not path.is_dir():
            QMessageBox.warning(
                self, "Pasta inválida", f"Não é um diretório:\n{path}"
            )
            return

        try:
            runs = discover_run_dirs(path)
        except Exception as e:
            QMessageBox.warning(self, "Erro", str(e))
            return
        if not runs:
            QMessageBox.warning(
                self,
                "Nenhuma run encontrada",
                f"Não há pastas com roi_crops/ em:\n{path}",
            )
            return

        if self.mode_run.isChecked() and not (path / "roi_crops").is_dir():
            QMessageBox.warning(
                self,
                "Não parece uma run",
                "No modo run específica, a pasta deve conter roi_crops/.",
            )
            return

        images_text = self.images_edit.text().strip()
        images_root = Path(images_text) if images_text else None
        if images_root is not None and not images_root.is_dir():
            QMessageBox.warning(
                self,
                "Pasta inválida",
                f"Pasta de imagens originais inválida:\n{images_root}",
            )
            return

        border = (
            int(self.border_spin.value()) if self.border_check.isChecked() else None
        )

        self.log_view.clear()
        self._append_log(
            f"Caminho: {path.resolve()}\n"
            f"Runs a processar: {len(runs)}\n"
            + "".join(f"  - {r}\n" for r in runs)
            + "\n"
        )
        self.status_label.setText("Em execução…")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        thread = QThread(self)
        worker = Worker(
            path,
            images_root,
            border,
            self.skip_existing_check.isChecked(),
            self.verbose_check.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.finished_ok.connect(self._on_ok)
        worker.failed.connect(self._on_fail)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished_ok.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _stop(self) -> None:
        if self._worker is None:
            return
        self.status_label.setText("Parando…")
        self.stop_button.setEnabled(False)
        self._append_log("Pedido de interrupção enviado…\n")
        self._worker.request_stop()

    def _on_ok(self, summary: str) -> None:
        self._append_log("\n" + summary + "\n")
        self.status_label.setText("Concluído com sucesso")
        QMessageBox.information(self, "Concluído", summary)

    def _on_fail(self, message: str) -> None:
        self._append_log(f"\nERRO: {message}\n")
        self.status_label.setText("Falhou")
        QMessageBox.critical(self, "Erro", message)

    def _on_cancelled(self, message: str) -> None:
        self._append_log(f"\n{message}\n")
        self.status_label.setText("Interrompido")
        QMessageBox.information(self, "Interrompido", message)

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.status_label.text() in ("Em execução…", "Parando…"):
            self.status_label.setText("Pronto")

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(self.log_view.textCursor().End)
        self.log_view.insertPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)


def main() -> None:
    prefer_pyqt5_platform_plugins()
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontUseNativeMenuBar, False)
    window = PostprocessWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
