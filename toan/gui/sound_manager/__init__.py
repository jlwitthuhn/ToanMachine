# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import os
from pathlib import Path

from PySide6 import QtWidgets
from scipy.io import wavfile

from toan.persistence.user_wav import UserWavDesc, get_user_wav_dir, get_user_wav_list
from toan.qt import WavFileModel

SOUND_TEXT = [
    "This is a list of user-imported wav files that can be used to enhance the training process.",
    "Any files listed here can optionally be added to the recording signal and then used as extra training data.",
]


class SoundManager(QtWidgets.QDialog):
    the_table: QtWidgets.QTableView | None = None

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

        self.the_table = QtWidgets.QTableView(self)
        self.the_table.setModel(wav_file_model)
        self.the_table.resizeColumnsToContents()
        layout.addWidget(self.the_table)

        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self._pressed_refresh)
        layout.addWidget(refresh_button)

        import_button = QtWidgets.QPushButton("Import wav...")
        import_button.clicked.connect(self._pressed_import)
        layout.addWidget(import_button)

    def _pressed_import(self):
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import WAV file", filter="*.wav"
        )
        if path_str == "":
            return

        path = Path(path_str)
        if not path.is_file():
            return

        sample_rate, signal = wavfile.read(path)

        out_dir = get_user_wav_dir()
        out_file = path.name
        if not out_file.endswith(".wav"):
            out_file += ".wav"
        out_path = os.path.join(out_dir, out_file)

        wavfile.write(out_path, sample_rate, signal)

        self._pressed_refresh()

    def _pressed_refresh(self):
        wav_files: list[UserWavDesc] = get_user_wav_list()
        wav_file_model = WavFileModel(self, wav_files)
        self.the_table.setModel(wav_file_model)
