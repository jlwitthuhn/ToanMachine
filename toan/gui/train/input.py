# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from pathlib import Path

from PySide6 import QtWidgets

from toan.gui.train import TrainingGuiContext


class TrainInputFilePage(QtWidgets.QWizardPage):
    context: TrainingGuiContext
    file_edit: QtWidgets.QLineEdit

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Select Recording")
        layout = QtWidgets.QVBoxLayout(self)

        layout.addStretch(1)

        input_label = QtWidgets.QLabel("Recording Zip File:", self)
        layout.addWidget(input_label)

        file_widget = QtWidgets.QWidget(self)
        file_layout = QtWidgets.QHBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)

        self.file_edit = QtWidgets.QLineEdit(self)
        self.file_edit.textChanged.connect(self.completeChanged)
        file_layout.addWidget(self.file_edit)

        file_browse_button = QtWidgets.QPushButton("Browse...")
        file_browse_button.clicked.connect(self._pressed_browse)
        file_layout.addWidget(file_browse_button)

        layout.addWidget(file_widget)

        layout.addStretch(1)

    def isComplete(self):
        file_path = self.file_edit.text()
        return Path(file_path).is_file()

    def validatePage(self):
        self.context.input_path = self.file_edit.text()
        return True

    def _pressed_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", filter="*.zip"
        )
        if path != "":
            self.file_edit.setText(path)
