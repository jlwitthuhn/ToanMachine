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
from toan.model.nam_a2_wavenet_torch import NamA2WaveNetTorch
from toan.signal.analysis import (
    generate_noise_frequency_response,
    generate_spectrogram,
    generate_sweep_frequency_response,
)


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


class FigureSelector(QtWidgets.QWidget):
    combo: QtWidgets.QComboBox
    stack: QtWidgets.QStackedWidget

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._generators: list[Callable[[], Figure]] = []
        self._loaded: set[int] = set()
        # Nothing is generated until the selector is first displayed
        self._displayed = False

        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QtWidgets.QLabel(label, row))
        self.combo = QtWidgets.QComboBox(row)
        row_layout.addWidget(self.combo)
        row_layout.addStretch(1)
        layout.addWidget(row)

        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack)

        self.combo.currentIndexChanged.connect(self.selected_option)

    def add_option(self, title: str, generate: Callable[[], Figure]) -> None:
        self.stack.addWidget(FigurePanel())
        self._generators.append(generate)
        self.combo.addItem(title)

    def load_current(self) -> None:
        self._displayed = True
        self._load(self.combo.currentIndex())

    def _load(self, index: int) -> None:
        if index in self._loaded:
            return
        panel = self.stack.widget(index)
        assert isinstance(panel, FigurePanel)
        panel.set_figure(self._generators[index]())
        self._loaded.add(index)

    def selected_option(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if self._displayed:
            self._load(index)


def _run_nam_submodels(model: NamA2WaveNetTorch, signal: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    input = np.concat([np.zeros(model.receptive_field - 1), signal])
    input = torch.tensor(input.astype(np.float32)).to(device)

    # Outputs are stacked like (num_submodels, batch, length)
    with torch.no_grad():
        outputs = model(input.reshape(1, -1))
    return outputs[:, 0, :].cpu().numpy()


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph_loss: FigurePanel
    selector_spec: FigureSelector
    selector_fr: FigureSelector

    signal_nam_big_sweep: np.ndarray | None = None
    signal_nam_small_sweep: np.ndarray | None = None
    signal_nam_big_white_noise: np.ndarray | None = None
    signal_nam_small_white_noise: np.ndarray | None = None

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

        loss_widget = QtWidgets.QWidget()
        loss_layout = QtWidgets.QVBoxLayout(loss_widget)
        self.graph_loss = FigurePanel()
        loss_layout.addWidget(self.graph_loss)
        self.tab_root.addTab(loss_widget, "Loss")

        self.selector_spec = FigureSelector("Source:")
        spec_index = self.tab_root.addTab(self.selector_spec, "Spectrogram")
        self._lazy_loaders[spec_index] = self.selector_spec.load_current
        self._add_spectrogram_source("Recording", lambda: self.context.signal_wet_sweep)

        layout.addWidget(self.tab_root)

    def initializePage(self):
        self.graph_loss.set_figure(
            self.context.progress_context.summaries[-1].generate_loss_graph(5)
        )
        self._process_nam_signals()
        self._build_nam_tabs()

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Nam Files (*.nam)")

        if file_path == "":
            return False

        with open(file_path, "w") as file:
            file.write(self.context.progress_context.model.export_nam_json_str())

        return True

    def _process_nam_signals(self) -> None:
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

        sweep_outputs = _run_nam_submodels(model, self.context.signal_dry_sweep)
        self.signal_nam_big_sweep = sweep_outputs[order[0]]
        self.signal_nam_small_sweep = sweep_outputs[order[-1]]

        white_noise_outputs = _run_nam_submodels(
            model, self.context.signal_dry_white_noise
        )
        self.signal_nam_big_white_noise = white_noise_outputs[order[0]]
        self.signal_nam_small_white_noise = white_noise_outputs[order[-1]]

    def _build_nam_tabs(self) -> None:
        if self._nam_tabs_built:
            return

        if self.signal_nam_big_sweep is None or self.signal_nam_small_sweep is None:
            return

        self._add_spectrogram_source("NAM Full", lambda: self.signal_nam_big_sweep)
        self._add_spectrogram_source("NAM Lite", lambda: self.signal_nam_small_sweep)

        self.selector_fr = FigureSelector("Signal:")
        self.selector_fr.add_option("Sweep", self._generate_sweep_frequency_response)
        self.selector_fr.add_option("Noise", self._generate_noise_frequency_response)
        self.selector_fr.combo.setCurrentText("Noise")
        fr_index = self.tab_root.addTab(self.selector_fr, "Frequency Response")
        self._lazy_loaders[fr_index] = self.selector_fr.load_current

        self._nam_tabs_built = True

    def _add_spectrogram_source(
        self, title: str, get_signal: Callable[[], np.ndarray]
    ) -> None:
        self.selector_spec.add_option(
            title, lambda: generate_spectrogram(self.context.sample_rate, get_signal())
        )

    def _generate_sweep_frequency_response(self) -> Figure:
        assert self.signal_nam_big_sweep is not None
        assert self.signal_nam_small_sweep is not None
        return generate_sweep_frequency_response(
            self.context.sample_rate,
            {
                "Recording": self.context.signal_wet_sweep,
                "NAM Full": self.signal_nam_big_sweep,
                "NAM Lite": self.signal_nam_small_sweep,
            },
        )

    def _generate_noise_frequency_response(self) -> Figure:
        assert self.signal_nam_big_white_noise is not None
        assert self.signal_nam_small_white_noise is not None
        return generate_noise_frequency_response(
            self.context.sample_rate,
            self.context.signal_dry_white_noise,
            {
                "Recording": self.context.signal_wet_white_noise,
                "NAM Full": self.signal_nam_big_white_noise,
                "NAM Lite": self.signal_nam_small_white_noise,
            },
        )

    def clicked_tab(self, index: int) -> None:
        if index in self._loaded:
            return
        loader = self._lazy_loaders.get(index)
        if loader is None:
            return
        loader()
        self._loaded.add(index)
