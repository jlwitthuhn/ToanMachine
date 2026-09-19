# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

INTRO_TEXT = [
    "Welcome to the recording wizard.",
    "This wizard will guide you through a step-by-step process to capture a recording of your pedal or amp.",
    "To begin, connect your pedal such that you can send a signal out an interface, through the pedal, and back into your interface.",
    "If you are recording an amp, be sure you have it connected through a load box and not directly plugged in to your interface.",
]


class RecordIntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Introduction")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(INTRO_TEXT), self)
        label.setWordWrap(True)

        layout.addWidget(label)
