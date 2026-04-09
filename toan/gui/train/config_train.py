# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.train.context import TrainingGuiContext


class TrainTrainConfigPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setCommitPage(True)
        self.setTitle("Training Configuration")
