# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore
from PySide6.QtGui import Qt

from toan.formatting import format_seconds_as_mmss
from toan.persistence import UserWavDesc


class WavFileModel(QtCore.QAbstractTableModel):
    file_list: list[UserWavDesc]
    with_checkbox: bool
    selected_wavs: set[str]

    def __init__(
        self, parent, file_list: list[UserWavDesc], with_checkbox: bool = False
    ):
        super().__init__(parent)
        self.file_list = file_list
        self.with_checkbox = with_checkbox
        self.selected_wavs = set()
        self.file_list.sort(key=lambda x: x.filename)

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
                return format_seconds_as_mmss(self.file_list[index.row()].duration)

        if role == Qt.CheckStateRole and self.with_checkbox and index.column() == 0:
            the_file = self.file_list[index.row()]
            if the_file.path in self.selected_wavs:
                return Qt.Checked
            else:
                return Qt.Unchecked

        return None

    def setData(self, index, value, /, role=...):
        if not index.isValid():
            return False
        if role != Qt.CheckStateRole or index.column() != 0:
            return False
        this_path = self.file_list[index.row()].path
        if this_path in self.selected_wavs:
            self.selected_wavs.remove(this_path)
        else:
            self.selected_wavs.add(this_path)
        self.dataChanged.emit(index, index, [role])
        return True

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

    def get_selected_wavs(self) -> list[UserWavDesc]:
        result = []
        for maybe_wav in self.file_list:
            if maybe_wav.path in self.selected_wavs:
                result.append(maybe_wav)
        return result

    def _select_all(self):
        for file in self.file_list:
            self.selected_wavs.add(file.path)
        index_start = self.createIndex(0, 0)
        index_end = self.createIndex(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(index_start, index_end)
