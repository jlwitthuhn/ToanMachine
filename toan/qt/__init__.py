# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore
from PySide6.QtGui import Qt

from toan.persistence import UserWavDesc


class WavFileModel(QtCore.QAbstractTableModel):
    file_list: list[UserWavDesc]
    with_checkbox: bool
    selected_wavs: set[UserWavDesc]

    def __init__(
        self, parent, file_list: list[UserWavDesc], with_checkbox: bool = False
    ):
        super().__init__(parent)
        self.file_list = file_list
        self.with_checkbox = with_checkbox

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.file_list)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if self.with_checkbox:
            return 4
        else:
            return 3

    def data(self, index, /, role=...):
        if not index.isValid():
            return None

        if self.with_checkbox:
            offset = 1
        else:
            offset = 0

        if role == Qt.DisplayRole:
            if index.column() == 0 + offset:
                return self.file_list[index.row()].filename
            if index.column() == 1 + offset:
                return f"{self.file_list[index.row()].sample_rate}"
            if index.column() == 2 + offset:
                return f"{self.file_list[index.row()].duration:.2f}s"

        if role == Qt.CheckStateRole and self.with_checkbox and index.column() == 0:
            return Qt.Unchecked

        return None

    def headerData(self, section, orientation, /, role=...):
        if role != Qt.DisplayRole:
            return None
        if orientation != Qt.Horizontal:
            return None

        if self.with_checkbox:
            offset = 1
        else:
            offset = 0

        if self.with_checkbox and section == 0:
            return "Enabled"
        if section == 0 + offset:
            return "Filename"
        if section == 1 + offset:
            return "Samp. Rate"
        if section == 2 + offset:
            return "Duration"
        return None

    def flags(self, index):
        flags = super().flags(index)
        if self.with_checkbox and index.column() == 0:
            flags |= Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
        return flags
