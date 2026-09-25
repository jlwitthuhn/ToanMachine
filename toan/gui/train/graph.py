# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from typing import Callable

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext
from toan.signal.analysis import generate_spectrogram, generate_sweep_frequency_response


# Widget to host a graph and keep it the right size
class FigurePanel(QtWidgets.QWidget):
    canvas: FigureCanvasQTAgg | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def set_figure(self, figure: Figure) -> None:
        if self.canvas is not None:
            self._layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.canvas = FigureCanvasQTAgg(figure)
        self._layout.addWidget(self.canvas)


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph_loss: FigurePanel
    graph_fr_sweep: FigurePanel

    combo_spec_source: QtWidgets.QComboBox
    stack_spec: QtWidgets.QStackedWidget

    signal_nam_big_sweep: np.ndarray | None = None
    signal_nam_small_sweep: np.ndarray | None = None

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Training Data")
        layout = QtWidgets.QVBoxLayout(self)

        self.tab_root = QtWidgets.QTabWidget()
        self.tab_root.currentChanged.connect(self.clicked_tab)

        # Lazy loaders populate graphs only when the tabs are displayed
        self._lazy_loaders: dict[int, Callable[[], None]] = {}
        self._loaded: set[int] = set()
        self._nam_tabs_built = False

        # Spectrogram sources are also populated only when selected
        self._spec_loaders: list[Callable[[], None]] = []
        self._spec_loaded: set[int] = set()

        loss_widget = QtWidgets.QWidget()
        loss_layout = QtWidgets.QVBoxLayout(loss_widget)
        self.graph_loss = FigurePanel()
        loss_layout.addWidget(self.graph_loss)
        self.tab_root.addTab(loss_widget, "Loss")

        spec_widget = QtWidgets.QWidget()
        spec_layout = QtWidgets.QVBoxLayout(spec_widget)

        source_row = QtWidgets.QWidget(spec_widget)
        source_row_layout = QtWidgets.QHBoxLayout(source_row)
        source_row_layout.setContentsMargins(0, 0, 0, 0)
        source_row_layout.addWidget(QtWidgets.QLabel("Source:", source_row))
        self.combo_spec_source = QtWidgets.QComboBox(source_row)
        source_row_layout.addWidget(self.combo_spec_source)
        source_row_layout.addStretch(1)
        spec_layout.addWidget(source_row)

        self.stack_spec = QtWidgets.QStackedWidget(spec_widget)
        spec_layout.addWidget(self.stack_spec)
        spec_index = self.tab_root.addTab(spec_widget, "Spectrogram")
        self._lazy_loaders[spec_index] = lambda: self._load_spectrogram_source(
            self.combo_spec_source.currentIndex()
        )

        self._add_spectrogram_source("Recording", lambda: self.context.signal_wet_sweep)
        # Connect after the first source so selecting it does not load it early
        self.combo_spec_source.currentIndexChanged.connect(
            self.selected_spectrogram_source
        )

        layout.addWidget(self.tab_root)

    def initializePage(self):
        self.graph_loss.set_figure(
            self.context.progress_context.summaries[-1].generate_loss_graph(5)
        )
        self._process_nam_sweeps()
        self._build_nam_tabs()

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Nam Files (*.nam)")

        if file_path == "":
            return False

        with open(file_path, "w") as file:
            file.write(self.context.progress_context.model.export_nam_json_str())

        return True

    def _process_nam_sweeps(self) -> None:
        if self.signal_nam_big_sweep is not None:
            return

        model = self.context.progress_context.model
        if model is None:
            return

        param_counts = [
            sum(p.numel() for p in submodel.parameters())
            for submodel in model.submodels
        ]
        order = sorted(
            range(len(param_counts)), key=lambda i: param_counts[i], reverse=True
        )

        device = next(model.parameters()).device
        input = np.concat(
            [np.zeros(model.receptive_field - 1), self.context.signal_dry_sweep]
        )
        input = torch.tensor(input.astype(np.float32)).to(device)

        # Outputs are stacked like (num_submodels, batch, length)
        with torch.no_grad():
            outputs = model(input.reshape(1, -1))
        outputs = outputs[:, 0, :].cpu().numpy()

        self.signal_nam_big_sweep = outputs[order[0]]
        self.signal_nam_small_sweep = outputs[order[-1]]

    def _build_nam_tabs(self) -> None:
        if self._nam_tabs_built:
            return

        if self.signal_nam_big_sweep is None or self.signal_nam_small_sweep is None:
            return

        self._add_spectrogram_source("NAM (Big)", lambda: self.signal_nam_big_sweep)
        self._add_spectrogram_source("NAM (Small)", lambda: self.signal_nam_small_sweep)

        fr_widget = QtWidgets.QWidget()
        fr_layout = QtWidgets.QVBoxLayout(fr_widget)
        self.graph_fr_sweep = FigurePanel()
        fr_layout.addWidget(self.graph_fr_sweep)
        fr_index = self.tab_root.addTab(fr_widget, "FR (Sweep)")
        self._lazy_loaders[fr_index] = self._load_sweep_frequency_response

        self._nam_tabs_built = True

    def _add_spectrogram_source(
        self, title: str, get_signal: Callable[[], np.ndarray]
    ) -> None:
        panel = FigurePanel()
        self.stack_spec.addWidget(panel)
        self._spec_loaders.append(lambda: self._load_spectrogram(panel, get_signal()))
        self.combo_spec_source.addItem(title)

    def _load_spectrogram_source(self, index: int) -> None:
        if index in self._spec_loaded:
            return
        self._spec_loaders[index]()
        self._spec_loaded.add(index)

    def _load_spectrogram(self, panel: FigurePanel, signal: np.ndarray) -> None:
        panel.set_figure(generate_spectrogram(self.context.sample_rate, signal))

    def _load_sweep_frequency_response(self) -> None:
        assert self.signal_nam_big_sweep is not None
        assert self.signal_nam_small_sweep is not None
        self.graph_fr_sweep.set_figure(
            generate_sweep_frequency_response(
                self.context.sample_rate,
                {
                    "Recording": self.context.signal_wet_sweep,
                    "NAM (Big)": self.signal_nam_big_sweep,
                    "NAM (Small)": self.signal_nam_small_sweep,
                },
            )
        )

    def selected_spectrogram_source(self, index: int) -> None:
        self._load_spectrogram_source(index)
        self.stack_spec.setCurrentIndex(index)

    def clicked_tab(self, index: int) -> None:
        if index in self._loaded:
            return
        loader = self._lazy_loaders.get(index)
        if loader is None:
            return
        loader()
        self._loaded.add(index)
