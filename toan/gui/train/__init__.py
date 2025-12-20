# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.train.context import TrainingContext
from toan.gui.train.input import TrainInputFilePage
from toan.gui.train.intro import TrainIntroPage
from toan.gui.train.train import TrainTrainPage
from toan.gui.train.validate import TrainValidatePage


class TrainingWizard(QtWidgets.QWizard):
    context: TrainingContext

    def __init__(self, parent):
        super().__init__(parent)
        self.context = TrainingContext()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.addPage(TrainIntroPage(self))
        self.addPage(TrainInputFilePage(self, self.context))
        self.addPage(TrainValidatePage(self, self.context))
        self.addPage(TrainTrainPage(self))

        self.setWindowTitle("Training Wizard")
        self.setModal(True)
