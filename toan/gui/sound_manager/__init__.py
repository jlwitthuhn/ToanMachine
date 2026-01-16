# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.persistence import UserWavDesc, get_user_wav_list
from toan.qt import WavFileModel

SOUND_TEXT = [
    "This is a list of user-imported wav files that can be used to enhance the training process.",
    "Any files listed here can optionally be added to the recording signal and then used as extra training data.",
]


class SoundManager(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Sound Manager")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(SOUND_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        wav_files: list[UserWavDesc] = get_user_wav_list()
        wav_file_model = WavFileModel(self, wav_files)

        table = QtWidgets.QTableView(self)
        table.setModel(wav_file_model)
        table.resizeColumnsToContents()
        layout.addWidget(table)

        refresh_button = QtWidgets.QPushButton("Refresh")
        layout.addWidget(refresh_button)
