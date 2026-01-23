# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading

from PySide6 import QtCore, QtGui, QtWidgets

from toan.gui.train import TrainingGuiContext
from toan.training.zip_loader import ZipLoaderContext, run_zip_loader


class TrainValidatePage(QtWidgets.QWizardPage):
    context: TrainingGuiContext
    thread_context: ZipLoaderContext | None = None
    timer_refresh: QtCore.QTimer

    text_edit: QtWidgets.QTextEdit

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context
        self.timer_refresh = QtCore.QTimer(self)
        self.timer_refresh.setInterval(100)
        self.timer_refresh.timeout.connect(self._do_refresh)

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
        self.thread_context = ZipLoaderContext()
        self.thread_context.text_edit = self.text_edit

        self.timer_refresh.start()

        def thread_func():
            run_zip_loader(self.thread_context, self.context.input_path)

        threading.Thread(target=thread_func).start()

    def isComplete(self):
        return (
            self.context.signal_dry is not None and self.context.signal_wet is not None
        )

    def _do_refresh(self):
        if self.thread_context is not None:
            with self.thread_context.messages_lock:
                for message in self.thread_context.messages_queue:
                    self.text_edit.append(message)
                self.thread_context.messages_queue.clear()
            if self.thread_context.complete:
                self.context.signal_dry = self.thread_context.signal_dry
                self.context.signal_wet = self.thread_context.signal_wet
                self.context.signal_dry_test = self.thread_context.signal_dry_test
                self.context.signal_wet_test = self.thread_context.signal_wet_test
                self.context.loaded_metadata = self.thread_context.metadata
                self.context.sample_rate = self.thread_context.sample_rate
                self.thread_context = None
                self.completeChanged.emit()
