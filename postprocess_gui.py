#!/usr/bin/env python3
"""Interface PyQt5 para pós-processar projetos e gerar ``atributos_por_roi.csv``.

A janela lista, com checkboxes, todos os projetos encontrados dentro da pasta
``outputs`` (padrão ``./outputs``, mas a pasta pode ser trocada). Cada projeto
segue a estrutura ``projeto/<run>/arquivos``, isto é, contém uma ou mais runs
com ``roi_crops/``. Para cada run dos projetos marcados o programa escreve
``atributos_por_roi.csv`` com o nome do ROI, o diâmetro de Feret (inferido de
``feret_diameters.csv``) e várias medidas de contraste a partir das
segmentações ``.npz``.

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
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

prefer_pyqt5_platform_plugins()

from atributos_por_roi import (
    OUTPUT_LOG_NAME,
    ProcessingCancelled,
    discover_run_dirs,
    process_paths,
)
from generate_roi_samples import (
    SAMPLES_DIR,
    process_run as generate_roi_sample_sheets,
)

prefer_pyqt5_platform_plugins()

REPO = Path(__file__).resolve().parent


def discover_projects(outputs_root: Path) -> list[tuple[Path, list[Path]]]:
    """Return ``(project_dir, runs)`` for each project under ``outputs_root``.

    Um projeto é uma subpasta imediata de ``outputs`` que contém runs
    (``projeto/<run>/roi_crops/``). Se a própria subpasta já for uma run, ela
    é listada como projeto de uma única run.
    """
    outputs_root = outputs_root.resolve()
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Não é um diretório: {outputs_root}")
    projects: list[tuple[Path, list[Path]]] = []
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            runs = discover_run_dirs(child)
        except Exception:
            continue
        if runs:
            projects.append((child.resolve(), runs))
    return projects


class Worker(QObject):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        paths: list[Path],
        images_root: Path | None,
        border: int | None,
        skip_existing: bool,
        generate_samples: bool,
        verbose: bool,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.images_root = images_root
        self.border = border
        self.skip_existing = skip_existing
        self.generate_samples = generate_samples
        self.verbose = verbose
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            results = process_paths(
                self.paths,
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
                label = f"{run_dir.parent.name}/{run_dir.name}"
                if self._stop:
                    raise ProcessingCancelled(
                        "Processamento interrompido pelo usuário."
                    )
                if skipped:
                    n_skipped += 1
                    msg = (
                        f"{label}: ignorado ({OUTPUT_LOG_NAME} marca completo, "
                        f"{n_rows} ROI(s))"
                    )
                else:
                    n_done += 1
                    msg = (
                        f"{label}: {n_rows} ROI(s) → {csv_path.name}"
                        f" ({unmatched} sem match de Feret)"
                    )
                lines.append(msg)
                self.log.emit(msg + "\n")

                if self.generate_samples:
                    self.log.emit(
                        f"{label}: gerando amostras em {SAMPLES_DIR}/ …\n"
                    )
                    out_samples = generate_roi_sample_sheets(
                        run_dir,
                        n_samples=180,
                        per_page=18,
                        cols=6,
                        rows=3,
                        seed=None,
                        verbose=False,
                    )
                    self.log.emit(f"  → {out_samples}\n")

            summary = (
                f"Concluído: {len(results)} run(s) "
                f"({n_done} processada(s), {n_skipped} ignorada(s))."
            )
            if self.generate_samples:
                summary += f"\nAmostras: {SAMPLES_DIR}/ (180 ROIs, 18/página)."
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
        self.resize(760, 680)
        self._thread: QThread | None = None
        self._worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self._build_io_group())
        layout.addWidget(self._build_options_group())
        layout.addLayout(self._build_action_bar())
        layout.addWidget(self._build_log_view(), stretch=1)

        self._refresh_projects()

    def _build_io_group(self) -> QGroupBox:
        group = QGroupBox("Projetos a processar")
        outer = QVBoxLayout(group)

        self.path_edit = QLineEdit(str(REPO / "outputs"))
        self.path_edit.editingFinished.connect(self._refresh_projects)
        browse = QPushButton("Procurar…")
        browse.clicked.connect(self._pick_path)
        refresh = QPushButton("Atualizar")
        refresh.clicked.connect(self._refresh_projects)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Pasta outputs:"))
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse)
        path_row.addWidget(refresh)
        outer.addLayout(path_row)

        self.project_list = QListWidget()
        self.project_list.setMinimumHeight(140)
        self.project_list.itemChanged.connect(self._update_selection_label)
        outer.addWidget(self.project_list, stretch=1)

        select_all = QPushButton("Marcar todos")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none = QPushButton("Desmarcar todos")
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #555;")
        sel_row = QHBoxLayout()
        sel_row.addWidget(select_all)
        sel_row.addWidget(select_none)
        sel_row.addWidget(self.selection_label, stretch=1)
        outer.addLayout(sel_row)

        hint = QLabel(
            "Cada projeto é uma subpasta de outputs com a estrutura "
            "projeto/<run>/arquivos. Uma run deve conter roi_crops/, "
            "feret_diameters.csv e background_difference_watershed/."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        outer.addWidget(hint)
        return group

    def _refresh_projects(self) -> None:
        """Re-scan the outputs folder and rebuild the checkbox list."""
        text = self.path_edit.text().strip()
        previously_checked = {p.name for p in self._checked_projects()}
        first_load = not previously_checked and self.project_list.count() == 0
        self.project_list.blockSignals(True)
        self.project_list.clear()
        self.project_list.blockSignals(False)

        if not text:
            self._update_selection_label()
            return
        try:
            projects = discover_projects(Path(text))
        except Exception as e:
            self.selection_label.setText(str(e))
            return

        # Trocar de pasta (nenhum nome em comum) volta a marcar tudo.
        if not any(d.name in previously_checked for d, _ in projects):
            first_load = True

        self.project_list.blockSignals(True)
        for project_dir, runs in projects:
            item = QListWidgetItem(f"{project_dir.name}  ({len(runs)} run(s))")
            item.setData(Qt.UserRole, str(project_dir))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = first_load or project_dir.name in previously_checked
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setToolTip(
                str(project_dir)
                + "\n"
                + "\n".join(f"  {r.name}" for r in runs)
            )
            self.project_list.addItem(item)
        self.project_list.blockSignals(False)
        self._update_selection_label()

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.project_list.blockSignals(True)
        for row in range(self.project_list.count()):
            self.project_list.item(row).setCheckState(state)
        self.project_list.blockSignals(False)
        self._update_selection_label()

    def _checked_projects(self) -> list[Path]:
        paths: list[Path] = []
        for row in range(self.project_list.count()):
            item = self.project_list.item(row)
            if item.checkState() == Qt.Checked:
                paths.append(Path(item.data(Qt.UserRole)))
        return paths

    def _update_selection_label(self, *_args) -> None:
        total = self.project_list.count()
        if total == 0:
            self.selection_label.setText("Nenhum projeto encontrado.")
            return
        n = len(self._checked_projects())
        self.selection_label.setText(f"{n} de {total} projeto(s) marcado(s)")

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

        self.samples_check = QCheckBox(
            f"Gerar 180 ROIs aleatórios em {SAMPLES_DIR}/ (18 por página)"
        )
        self.samples_check.setChecked(False)
        form.addRow("", self.samples_check)

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

    def _pick_path(self) -> None:
        start = self.path_edit.text().strip() or str(REPO / "outputs")
        chosen = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de outputs", start
        )
        if chosen:
            self.path_edit.setText(chosen)
            self._refresh_projects()

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
            QMessageBox.warning(self, "Pasta ausente", "Escolha a pasta outputs.")
            return
        outputs_root = Path(text)
        if not outputs_root.is_dir():
            QMessageBox.warning(
                self, "Pasta inválida", f"Não é um diretório:\n{outputs_root}"
            )
            return

        projects = self._checked_projects()
        if not projects:
            QMessageBox.warning(
                self,
                "Nenhum projeto marcado",
                "Marque ao menos um projeto para processar.",
            )
            return

        runs_by_project: list[tuple[Path, list[Path]]] = []
        for project in projects:
            try:
                runs = discover_run_dirs(project)
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"{project}:\n{e}")
                return
            if not runs:
                QMessageBox.warning(
                    self,
                    "Nenhuma run encontrada",
                    f"Não há pastas com roi_crops/ em:\n{project}",
                )
                return
            runs_by_project.append((project, runs))

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

        total_runs = sum(len(runs) for _, runs in runs_by_project)
        self.log_view.clear()
        lines = [
            f"Pasta outputs: {outputs_root.resolve()}",
            f"Projetos marcados: {len(runs_by_project)} "
            f"({total_runs} run(s) no total)",
        ]
        for project, runs in runs_by_project:
            lines.append(f"  {project.name}/")
            lines.extend(f"    - {r.name}" for r in runs)
        self._append_log("\n".join(lines) + "\n\n")
        self.status_label.setText("Em execução…")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        thread = QThread(self)
        worker = Worker(
            [project for project, _ in runs_by_project],
            images_root,
            border,
            self.skip_existing_check.isChecked(),
            self.samples_check.isChecked(),
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
