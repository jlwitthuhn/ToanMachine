# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

INTRO_TEXT = [
    "Welcome to the training wizard.",
    "This wizard will guide you through the process of creating a NAM model from a recording of a device.",
    'Before you begin, you will need to have already recorded your device with the "Record Device" button from the main window.',
]


class TrainIntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Introduction")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(INTRO_TEXT), self)
        label.setWordWrap(True)

        layout.addWidget(label)
