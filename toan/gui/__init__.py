# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.main_window import MainWindow
from toan.persistence import create_user_wav_dir


def run_qt_gui() -> None:
    app = QtWidgets.QApplication([])

    create_user_wav_dir()

    window = MainWindow()
    window.show()

    app.exec()
