# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading
import zipfile

from PySide6 import QtGui, QtWidgets

from toan.gui.train import TrainingContext


class _ValidateThreadContext:
    text_edit: QtWidgets.QTextEdit


class TrainValidatePage(QtWidgets.QWizardPage):
    context: TrainingContext
    thread_context: _ValidateThreadContext | None = None
    text_edit: QtWidgets.QTextEdit

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Checking Input")
        layout = QtWidgets.QVBoxLayout(self)

        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setFont(font)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def initializePage(self):
        self.text_edit.clear()
        self.text_edit.append(f"Analyzing input file: {self.context.input_path}")
        self.thread_context = _ValidateThreadContext()
        self.thread_context.text_edit = self.text_edit

        def thread_func():
            _run_thread(self.thread_context, self.context.input_path)

        threading.Thread(target=thread_func).start()


def _run_thread(context: _ValidateThreadContext, input_path: str):
    context.text_edit.append("Loading as zip archive...")
    try:
        with zipfile.ZipFile(input_path, "r") as zip_file:
            context.text_edit.append("Loading file: config.json")
            try:
                config_json_bytes = zip_file.read("config.json")
            except KeyError:
                context.text_edit.append("Error: config.json not found")
                return
    except zipfile.BadZipFile:
        context.text_edit.append("Error: File is not a valid zip archive")
        return
    except:
        context.text_edit.append("Error: Unknown error occurred")
        return
