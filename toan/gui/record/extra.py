# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record import RecordingContext
from toan.persistence import get_user_wav_list
from toan.qt import WavFileModel

EXTRA_AUDIO_TEXT = [
    "Select any extra audio you would like to include in the recorded signal.",
    "You can import audio to be used here with 'Sound Manager' from the main menu.",
]


class RecordExtraPage(QtWidgets.QWizardPage):
    context: RecordingContext

    table: QtWidgets.QTableView

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Extra Audio")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(EXTRA_AUDIO_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.table = QtWidgets.QTableView(self)
        layout.addWidget(self.table)

    def initializePage(self):
        wav_files = get_user_wav_list()
        model = WavFileModel(self, wav_files, True)
        model._select_all()
        self.table.setModel(model)
