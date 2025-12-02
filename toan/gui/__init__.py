from PySide6 import QtWidgets

from toan.gui.main_window import MainWindow


def run_qt_gui() -> None:
    app = QtWidgets.QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
