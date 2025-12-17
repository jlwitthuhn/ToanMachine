# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.train.intro import TrainIntroPage


class TrainingWizard(QtWidgets.QWizard):
    def __init__(self, parent):
        super().__init__(parent)

        self.addPage(TrainIntroPage(self))

        self.setWindowTitle("Training Wizard")
        self.setModal(True)
