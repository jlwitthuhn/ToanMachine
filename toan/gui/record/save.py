# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import io
import zipfile

import numpy as np
import scipy
from PySide6 import QtWidgets

from toan.gui.record import RecordingContext

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

        dry_wav = io.BytesIO()
        scipy.io.wavfile.write(
            dry_wav,
            self.context.sample_rate,
            self.context.signal_dry.astype(np.float32),
        )
        dry_wav.seek(0)

        wet_wav = io.BytesIO()
        scipy.io.wavfile.write(
            wet_wav, self.context.sample_rate, self.context.signal_recorded
        )
        wet_wav.seek(0)

        with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr("dry.wav", dry_wav.getvalue())
            zip.writestr("wet.wav", wet_wav.getvalue())

        return True
