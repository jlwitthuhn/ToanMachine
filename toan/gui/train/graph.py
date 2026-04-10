# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    graph: FigureCanvasQTAgg

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Training Data")
        layout = QtWidgets.QVBoxLayout(self)

        tab_root = QtWidgets.QTabWidget()

        loss_widget = QtWidgets.QWidget()
        loss_layout = QtWidgets.QVBoxLayout(loss_widget)
        self.graph = FigureCanvasQTAgg()
        loss_layout.addWidget(self.graph)

        tab_root.addTab(loss_widget, "Loss")

        layout.addWidget(tab_root)

    def initializePage(self):
        figure = self.context.progress_context.summary.generate_loss_graph(5)
        self.graph.figure = figure

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Nam Files (*.nam)")

        if file_path == "":
            return False

        with open(file_path, "w") as file:
            file.write(self.context.progress_context.model.export_nam_json_str())

        return True
