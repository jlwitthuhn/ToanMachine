# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets

from toan.gui.train import TrainingContext


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingContext

    graph: FigureCanvasQTAgg

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Training Graphs")
        layout = QtWidgets.QVBoxLayout(self)

        self.graph = FigureCanvasQTAgg()
        layout.addWidget(self.graph)

    def initializePage(self):
        figure = self.context.training_summary.generate_loss_graph(5)
        self.graph.figure = figure
