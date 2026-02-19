# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import datetime
import threading

from PySide6 import QtCore, QtWidgets

from toan.formatting import format_seconds_as_mmss
from toan.gui.train import TrainingGuiContext
from toan.training.config import TrainingConfig
from toan.training.loop import run_training_loop

TRAIN_TEXT = [
    "Your model is now training. After training has finished you will be asked to choose a location for the NAM file."
]


class TrainTrainPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext
    refresh_timer: QtCore.QTimer

    progress_bar: QtWidgets.QProgressBar
    progress_desc_test: QtWidgets.QLabel
    progress_desc_train: QtWidgets.QLabel

    timestamp_begin: datetime.datetime | None = None
    timer_label: QtWidgets.QLabel

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context
        self.context.progress_lock = threading.Lock()
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.setInterval(150)
        self.refresh_timer.setSingleShot(False)
        self.refresh_timer.timeout.connect(self.refresh_page)
        self.refresh_timer.start()

        self.setCommitPage(True)
        self.setTitle("Train")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(TRAIN_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.timer_label = QtWidgets.QLabel(
            f"Time spent: {format_seconds_as_mmss(0)}", self
        )
        layout.addWidget(self.timer_label)

        self.progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.progress_desc_test = QtWidgets.QLabel(f"Test loss: ?")
        layout.addWidget(self.progress_desc_test)

        self.progress_desc_train = QtWidgets.QLabel("Training loss:", self)
        layout.addWidget(self.progress_desc_train)

    def initializePage(self):
        # Copy all needed data from gui context to thread context before starting
        self.context.progress_context.model_config = self.context.model_config
        self.context.progress_context.metadata = self.context.loaded_metadata
        self.context.progress_context.sample_rate = self.context.sample_rate

        self.context.progress_context.signal_dry_test = self.context.signal_dry_test
        self.context.progress_context.signal_wet_test = self.context.signal_wet_test
        self.context.progress_context.signal_dry_train = self.context.signal_dry
        self.context.progress_context.signal_wet_train = self.context.signal_wet

        def thread_func():
            run_training_loop(self.context.progress_context, TrainingConfig())

        self.context.quit_training = False
        self.timestamp_begin = datetime.datetime.now()
        threading.Thread(target=thread_func).start()

    def cleanupPage(self):
        self.refresh_timer.stop()
        self.context.quit_training = True

    def isComplete(self) -> bool:
        return self.context.progress_context.model is not None

    def validatePage(self) -> bool:
        return self.context.progress_context.model is not None

    def refresh_page(self):
        with self.context.progress_context.lock:
            self.progress_bar.setMaximum(self.context.progress_context.iters_total)
            self.progress_bar.setValue(self.context.progress_context.iters_done)
            self.progress_bar.repaint()
            if self.context.progress_context.loss_train is not None:
                self.progress_desc_train.setText(
                    f"Training loss: {self.context.progress_context.loss_train:.5f}"
                )
            if (
                self.context.progress_context.summary is not None
                and self.context.progress_context.summary.losses_test is not None
                and len(self.context.progress_context.summary.losses_test) > 0
            ):
                self.progress_desc_test.setText(
                    f"Test loss: {self.context.progress_context.summary.losses_test[-1]:.5f}"
                )
            if self.context.progress_context.model is not None:
                self.refresh_timer.stop()
                self.completeChanged.emit()
        if self.timestamp_begin is not None:
            current_time = datetime.datetime.now()
            time_elapsed = current_time - self.timestamp_begin
            formatted_time = format_seconds_as_mmss(time_elapsed.total_seconds())
            self.timer_label.setText(f"Time spent: {formatted_time}")
