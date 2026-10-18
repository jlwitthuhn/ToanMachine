# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import sys

from PySide6 import QtWidgets

from toan.gui.main_window import MainWindow
from toan.persistence.user_wav import create_user_wav_dir


def run_qt_gui(app: QtWidgets.QApplication) -> None:
    show_debug = False
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        show_debug = True

    create_user_wav_dir()

    window = MainWindow(show_debug)
    window.show()

    app.exec()
