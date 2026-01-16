# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore
from PySide6.QtGui import Qt

from toan.persistence import UserWavDesc


class WavFileModel(QtCore.QAbstractTableModel):
    file_list: list[UserWavDesc]

    def __init__(self, parent, file_list: list[UserWavDesc]):
        super().__init__(parent)
        self.file_list = file_list

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.file_list)

    def columnCount(self, parent=QtCore.QModelIndex()):
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
