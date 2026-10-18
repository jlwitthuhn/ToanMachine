# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from collections.abc import Callable

# Only PySide6 is imported up front, everything else (torch, numpy, scipy, ...)
# is imported by ImportThread while the loading window is visible.
from PySide6 import QtCore, QtGui, QtWidgets


class LoadingWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None, QtCore.Qt.WindowType.FramelessWindowHint)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(48, 32, 48, 32)
        layout.addWidget(QtWidgets.QLabel("Toan Machine is loading...", self))

        self.adjustSize()
        center = self.screen().availableGeometry().center()
        self.move(center - self.rect().center())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # The window can't be dismissed, it's hidden once loading finishes
        event.ignore()


class ImportThread(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.run_qt_gui: Callable[[QtWidgets.QApplication], None] | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            from toan.gui import run_qt_gui

            self.run_qt_gui = run_qt_gui
        except BaseException as e:
            self.error = e


def main() -> None:
    app = QtWidgets.QApplication([])

    loading_window = LoadingWindow()
    loading_window.show()

    loop = QtCore.QEventLoop()
    import_thread = ImportThread()
    import_thread.finished.connect(loop.quit)
    import_thread.start()
    loop.exec()
    import_thread.wait()

    loading_window.hide()

    if import_thread.error is not None:
        raise import_thread.error
    import_thread.run_qt_gui(app)


if __name__ == "__main__":
    main()
