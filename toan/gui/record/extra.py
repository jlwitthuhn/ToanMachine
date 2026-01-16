# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record import RecordingContext

EXTRA_AUDIO_TEXT = [
    "Select any extra audio you would like to include in the recorded signal.",
    "You can import audio to be used here with 'Sound Manager' from the main menu.",
]


class RecordExtraPage(QtWidgets.QWizardPage):
    context: RecordingContext

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Extra Audio")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(EXTRA_AUDIO_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)
