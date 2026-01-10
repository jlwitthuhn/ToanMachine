# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.train import TrainingContext


class TrainGraphPage(QtWidgets.QWizardPage):
    context: TrainingContext

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Training Graphs")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Training Loss")
        layout.addWidget(label)

    def initializePage(self):
        self.context.training_summary.generate_loss_graph(7)
