# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from typing import Callable

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext
from toan.signal.analysis import generate_spectrogram, generate_sweep_frequency_response


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph_loss: FigureCanvasQTAgg
    graph_spec_real: FigureCanvasQTAgg
    graph_fr_sweep: FigureCanvasQTAgg

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

        loss_widget = QtWidgets.QWidget()
        loss_layout = QtWidgets.QVBoxLayout(loss_widget)
        self.graph_loss = FigureCanvasQTAgg()
        loss_layout.addWidget(self.graph_loss)
        self.tab_root.addTab(loss_widget, "Loss")

        spec_real_widget = QtWidgets.QWidget()
        spec_real_layout = QtWidgets.QVBoxLayout(spec_real_widget)
        self.graph_spec_real = FigureCanvasQTAgg()
        spec_real_layout.addWidget(self.graph_spec_real)
        real_index = self.tab_root.addTab(spec_real_widget, "Spectrogram (Real)")
        self._lazy_loaders[real_index] = self._load_real_spectrogram

        layout.addWidget(self.tab_root)

    def initializePage(self):
        self.graph_loss.figure = self.context.progress_context.summaries[
            -1
        ].generate_loss_graph(5)
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

        self._add_nam_tab("NAM (Big)", self.signal_nam_big_sweep)
        self._add_nam_tab("NAM (Small)", self.signal_nam_small_sweep)

        fr_widget = QtWidgets.QWidget()
        fr_layout = QtWidgets.QVBoxLayout(fr_widget)
        self.graph_fr_sweep = FigureCanvasQTAgg()
        fr_layout.addWidget(self.graph_fr_sweep)
        fr_index = self.tab_root.addTab(fr_widget, "FR (Sweep)")
        self._lazy_loaders[fr_index] = self._load_sweep_frequency_response

        self._nam_tabs_built = True

    def _add_nam_tab(self, title: str, signal: np.ndarray) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg()
        layout.addWidget(canvas)
        index = self.tab_root.addTab(widget, title)
        self._lazy_loaders[index] = lambda: self._load_nam_spectrogram(canvas, signal)

    def _load_real_spectrogram(self) -> None:
        self.graph_spec_real.figure = generate_spectrogram(
            self.context.sample_rate, self.context.signal_wet_sweep
        )
        self.graph_spec_real.draw_idle()
        self.graph_spec_real.flush_events()

    def _load_nam_spectrogram(
        self, canvas: FigureCanvasQTAgg, signal: np.ndarray
    ) -> None:
        canvas.figure = generate_spectrogram(self.context.sample_rate, signal)
        canvas.draw_idle()
        canvas.flush_events()

    def _load_sweep_frequency_response(self) -> None:
        assert self.signal_nam_big_sweep is not None
        assert self.signal_nam_small_sweep is not None
        self.graph_fr_sweep.figure = generate_sweep_frequency_response(
            self.context.sample_rate,
            {
                "Real": self.context.signal_wet_sweep,
                "NAM (Big)": self.signal_nam_big_sweep,
                "NAM (Small)": self.signal_nam_small_sweep,
            },
        )
        self.graph_fr_sweep.draw_idle()
        self.graph_fr_sweep.flush_events()

    def clicked_tab(self, index: int) -> None:
        if index in self._loaded:
            return
        loader = self._lazy_loaders.get(index)
        if loader is None:
            return
        loader()
        self._loaded.add(index)
