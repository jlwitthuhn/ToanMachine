# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record import RecordingContext
from toan.zip import create_training_zip

SAVE_TEXT = [
    "You did et.",
]


class RecordSavePage(QtWidgets.QWizardPage):
    context: RecordingContext

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Save Recording")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(SAVE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

    def validatePage(self) -> bool:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(filter="Zip Files (*.zip)")

        if file_path == "":
            return False

        zip_buffer = create_training_zip(
            self.context.sample_rate,
            self.context.signal_dry,
            self.context.signal_recorded,
            self.context.device_make,
            self.context.device_model,
            self.context.test_signal_offset,
        )

        with open(file_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        return True
