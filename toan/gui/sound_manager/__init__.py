# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QModelIndex, Qt

from toan.persistence import UserWavDesc, get_user_wav_list

SOUND_TEXT = [
    "This is a list of user-imported wav files that can be used to enhance the training process.",
    "Any files listed here can optionally be added to the recording signal and then used as extra training data.",
]


class WavFileModel(QtCore.QAbstractTableModel):
    file_list: list[UserWavDesc]

    def __init__(self, parent, file_list: list[UserWavDesc]):
        super().__init__(parent)
        self.file_list = file_list

    def rowCount(self, parent=QModelIndex()):
        return len(self.file_list)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def data(self, index, /, role=...):
        if not index.isValid():
            return None
        if role != Qt.DisplayRole:
            return None
        if index.column() == 0:
            return self.file_list[index.row()].filename
        if index.column() == 1:
            return f"{self.file_list[index.row()].sample_rate}"
        if index.column() == 2:
            return f"{self.file_list[index.row()].duration:.2f}s"
        return None

    def headerData(self, section, orientation, /, role=...):
        if role != Qt.DisplayRole:
            return None
        if orientation != Qt.Horizontal:
            return None
        if section == 0:
            return "Filename"
        if section == 1:
            return "Samp. Rate"
        if section == 2:
            return "Duration"
        return None


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
