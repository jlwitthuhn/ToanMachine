# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import sys
import threading
import traceback
from types import TracebackType

from PySide6 import QtCore, QtGui, QtWidgets


def _format_exception_type(exc_type: type[BaseException]) -> str:
    if exc_type.__module__ == "builtins":
        return exc_type.__qualname__
    return f"{exc_type.__module__}.{exc_type.__qualname__}"


class ExceptionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ):
        super().__init__(None)
        self.setWindowTitle("Unhandled Exception")
        self.setModal(True)
        self.resize(720, 420)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("An unhandled exception has occurred", self))

        type_label = QtWidgets.QLabel(_format_exception_type(exc_type), self)
        type_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        type_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(type_label)

        text_edit = QtWidgets.QPlainTextEdit(self)
        text_edit.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(
            "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        )
        layout.addWidget(text_edit)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, self
        )
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class _ExceptionReporter(QtCore.QObject):
    exception_raised = QtCore.Signal(object, object, object)

    def __init__(self):
        super().__init__()
        self.dialog_open = False
        self.exception_raised.connect(self._show_dialog)

    @QtCore.Slot(object, object, object)
    def _show_dialog(self, exc_type, exc_value, exc_traceback) -> None:
        # Only one error dialog at a time so they don't stack up on errors in a loop
        if self.dialog_open:
            return
        self.dialog_open = True
        try:
            ExceptionDialog(exc_type, exc_value, exc_traceback).exec()
        finally:
            self.dialog_open = False


def install_exception_hook() -> None:
    reporter = _ExceptionReporter()
    previous_sys_hook = sys.excepthook
    previous_threading_hook = threading.excepthook

    def sys_hook(exc_type, exc_value, exc_traceback) -> None:
        previous_sys_hook(exc_type, exc_value, exc_traceback)
        if not issubclass(exc_type, KeyboardInterrupt):
            reporter.exception_raised.emit(exc_type, exc_value, exc_traceback)

    def threading_hook(args: threading.ExceptHookArgs) -> None:
        previous_threading_hook(args)
        # The default threading hook silently ignores SystemExit, do the same
        if not issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            reporter.exception_raised.emit(
                args.exc_type, args.exc_value, args.exc_traceback
            )

    sys.excepthook = sys_hook
    threading.excepthook = threading_hook
