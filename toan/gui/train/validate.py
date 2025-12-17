# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets


class TrainValidatePage(QtWidgets.QWizardPage):
    text_edit: QtWidgets.QTextEdit

    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Checking Input")
        layout = QtWidgets.QVBoxLayout(self)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def initializePage(self):
        self.text_edit.clear()
        self.text_edit.append("Analyzing file: PATH HERE\n")
