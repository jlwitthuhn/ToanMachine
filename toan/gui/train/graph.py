# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from typing import Callable

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext
from toan.model.nam_a2_wavenet_torch import NamA2WaveNetTorch
from toan.signal.analysis import generate_spectrogram


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph_loss: FigureCanvasQTAgg
    graph_spec_real: FigureCanvasQTAgg

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
        self.graph_loss.figure = (
            self.context.progress_context.summary.generate_loss_graph(5)
        )
        self._build_nam_tabs()

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Nam Files (*.nam)")

        if file_path == "":
            return False

        with open(file_path, "w") as file:
            file.write(self.context.progress_context.model.export_nam_json_str())

        return True

    def _build_nam_tabs(self) -> None:
        if self._nam_tabs_built:
            return

        model = self.context.progress_context.model
        if model is None:
            return

        if isinstance(model, NamA2WaveNetTorch):
            param_counts = [
                sum(p.numel() for p in submodel.parameters())
                for submodel in model.submodels
            ]
            order = sorted(
                range(len(param_counts)), key=lambda i: param_counts[i], reverse=True
            )
            titles = ["NAM (Big)", "NAM (Small)"]
            for title, sub_index in zip(titles, order):
                self._add_nam_tab(title, sub_index)
        else:
            self._add_nam_tab("Spectrogram (NAM)", None)

        self._nam_tabs_built = True

    def _add_nam_tab(self, title: str, sub_index: int | None) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        canvas = FigureCanvasQTAgg()
        layout.addWidget(canvas)
        index = self.tab_root.addTab(widget, title)
        self._lazy_loaders[index] = lambda: self._load_nam_spectrogram(
            canvas, sub_index
        )

    def _load_real_spectrogram(self) -> None:
        self.graph_spec_real.figure = generate_spectrogram(
            self.context.sample_rate, self.context.signal_wet_sweep
        )
        self.graph_spec_real.draw_idle()
        self.graph_spec_real.flush_events()

    def _load_nam_spectrogram(
        self, canvas: FigureCanvasQTAgg, sub_index: int | None
    ) -> None:
        the_model = self.context.progress_context.model
        assert the_model is not None
        input = np.concat(
            [np.zeros(the_model.receptive_field - 1), self.context.signal_dry_sweep]
        )
        input = torch.tensor(input.astype(np.float32)).to(torch.device("mps"))

        with torch.no_grad():
            if sub_index is None:
                output = the_model(input.reshape(1, -1))
            else:
                output = the_model.submodels[sub_index](input.reshape(1, -1))
        output = output.squeeze().cpu().detach().numpy()

        canvas.figure = generate_spectrogram(self.context.sample_rate, output)
        canvas.draw_idle()
        canvas.flush_events()

    def clicked_tab(self, index: int) -> None:
        if index in self._loaded:
            return
        loader = self._lazy_loaders.get(index)
        if loader is None:
            return
        loader()
        self._loaded.add(index)
