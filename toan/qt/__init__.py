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
    selected_test_wavs: set[str]
    selected_train_wavs: set[str]

    def __init__(
        self, parent, file_list: list[UserWavDesc], with_checkbox: bool = False
    ):
        super().__init__(parent)
        self.file_list = file_list
        self.with_checkbox = with_checkbox
        self.selected_test_wavs = set()
        self.selected_train_wavs = set()
        self.file_list.sort(key=lambda x: x.filename)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.file_list)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if self.with_checkbox:
            return 5
        else:
            return 3

    def data(self, index, /, role=...):
        if not index.isValid():
            return None

        if self.with_checkbox:
            offset = 2
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
            if the_file.path in self.selected_train_wavs:
                return Qt.Checked
            else:
                return Qt.Unchecked

        if role == Qt.CheckStateRole and self.with_checkbox and index.column() == 1:
            the_file = self.file_list[index.row()]
            if the_file.path in self.selected_test_wavs:
                return Qt.Checked
            else:
                return Qt.Unchecked

        return None

    def setData(self, index, value, /, role=...):
        if not index.isValid():
            return False
        if role != Qt.CheckStateRole or index.column() > 1:
            return False
        this_path = self.file_list[index.row()].path
        if index.column() == 0:
            if this_path in self.selected_train_wavs:
                self.selected_train_wavs.remove(this_path)
            else:
                self.selected_test_wavs.remove(this_path)
                self.selected_train_wavs.add(this_path)
        elif index.column() == 1:
            if this_path in self.selected_test_wavs:
                self.selected_test_wavs.remove(this_path)
            else:
                self.selected_train_wavs.remove(this_path)
                self.selected_test_wavs.add(this_path)
        self.emit_all_changed()
        return True

    def headerData(self, section, orientation, /, role=...):
        if role != Qt.DisplayRole:
            return None
        if orientation != Qt.Horizontal:
            return None

        if self.with_checkbox:
            offset = 2
        else:
            offset = 0

        if self.with_checkbox and section == 0:
            return "Train"
        if self.with_checkbox and section == 1:
            return "Test"
        if section == 0 + offset:
            return "Filename"
        if section == 1 + offset:
            return "Samp. Rate"
        if section == 2 + offset:
            return "Duration"
        return None

    def flags(self, index):
        flags = super().flags(index)
        if self.with_checkbox and 0 <= index.column() <= 1:
            flags |= Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
        return flags

    def get_selected_test_wavs(self) -> list[UserWavDesc]:
        result = []
        for maybe_wav in self.file_list:
            if maybe_wav.path in self.selected_test_wavs:
                result.append(maybe_wav)
        return result

    def get_selected_train_wavs(self) -> list[UserWavDesc]:
        result = []
        for maybe_wav in self.file_list:
            if maybe_wav.path in self.selected_train_wavs:
                result.append(maybe_wav)
        return result

    def deselect_all(self):
        self.selected_test_wavs.clear()
        self.selected_train_wavs.clear()
        self.emit_all_changed()

    def emit_all_changed(self):
        index_start = self.createIndex(0, 0)
        index_end = self.createIndex(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(index_start, index_end)

    def select_all_test(self):
        self.deselect_all()
        for file in self.file_list:
            self.selected_test_wavs.add(file.path)
        self.emit_all_changed()

    def select_all_train(self):
        self.deselect_all()
        for file in self.file_list:
            self.selected_train_wavs.add(file.path)
        self.emit_all_changed()
