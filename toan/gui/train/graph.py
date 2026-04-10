# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext
from toan.signal.analysis import generate_spectrogram


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph_loss: FigureCanvasQTAgg
    graph_spec_real: FigureCanvasQTAgg
    graph_spec_real_loaded: bool = False
    graph_spec_nam: FigureCanvasQTAgg
    graph_spec_nam_loaded: bool = False

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Training Data")
        layout = QtWidgets.QVBoxLayout(self)

        tab_root = QtWidgets.QTabWidget()
        tab_root.currentChanged.connect(self.clicked_tab)

        loss_widget = QtWidgets.QWidget()
        loss_layout = QtWidgets.QVBoxLayout(loss_widget)
        self.graph_loss = FigureCanvasQTAgg()
        loss_layout.addWidget(self.graph_loss)
        tab_root.addTab(loss_widget, "Loss")

        spec_real_widget = QtWidgets.QWidget()
        spec_real_layout = QtWidgets.QVBoxLayout(spec_real_widget)
        self.graph_spec_real = FigureCanvasQTAgg()
        spec_real_layout.addWidget(self.graph_spec_real)
        tab_root.addTab(spec_real_widget, "Spectrogram (Real)")

        spec_nam_widget = QtWidgets.QWidget()
        spec_nam_layout = QtWidgets.QVBoxLayout(spec_nam_widget)
        self.graph_spec_nam = FigureCanvasQTAgg()
        spec_nam_layout.addWidget(self.graph_spec_nam)
        tab_root.addTab(spec_nam_widget, "Spectrogram (NAM)")

        layout.addWidget(tab_root)

    def initializePage(self):
        self.graph_loss.figure = (
            self.context.progress_context.summary.generate_loss_graph(5)
        )

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Nam Files (*.nam)")

        if file_path == "":
            return False

        with open(file_path, "w") as file:
            file.write(self.context.progress_context.model.export_nam_json_str())

        return True

    def clicked_tab(self, index: int) -> None:
        if index == 1 and not self.graph_spec_real_loaded:
            self.graph_spec_real.figure = generate_spectrogram(
                self.context.sample_rate, self.context.signal_wet_sweep
            )
            self.graph_spec_real.draw_idle()
            self.graph_spec_real.flush_events()
            self.graph_spec_real_loaded = True
        elif index == 2 and not self.graph_spec_nam_loaded:
            self.graph_spec_nam.figure = generate_spectrogram(
                self.context.sample_rate, self.context.signal_dry_sweep
            )
            self.graph_spec_nam.draw_idle()
            self.graph_spec_nam.flush_events()
            self.graph_spec_nam_loaded = True
