# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets

from toan.gui.record import RecordingContext
from toan.signal import generate_capture_signal
from toan.soundio import SdPlayrecController, prepare_play_record

RECORD_TEXT = [
    "In this section you will send a signal through your pedal and record the result.",
    "Once you have started, allow the full recording to complete before proceeding. Do not change any settings on your pedal while recording.",
]


class RecordWetSignalPage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_record: QtWidgets.QPushButton
    bar_progress: QtWidgets.QProgressBar
    bar_update_timer: QtCore.QTimer
    io_controller: SdPlayrecController | None = None

    signal_out_index: int

    recorded_samples: list[np.ndarray]

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context
        self.signal_out_index = 0
        self.recorded_samples = []

        self.bar_update_timer = QtCore.QTimer()
        self.bar_update_timer.setInterval(50)
        self.bar_update_timer.setSingleShot(False)
        self.bar_update_timer.timeout.connect(self._update_status)

        self.setTitle("Record")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(RECORD_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.button_record = QtWidgets.QPushButton("Record", self)
        self.button_record.clicked.connect(self._clicked_record)
        layout.addWidget(self.button_record)

        label_progress = QtWidgets.QLabel("Progress:", self)
        layout.addWidget(label_progress)

        self.bar_progress = QtWidgets.QProgressBar(self)
        layout.addWidget(self.bar_progress)

    def cleanupPage(self):
        if self.io_controller is not None:
            self.io_controller.close()

    def isComplete(self):
        if self.context.signal_recorded is not None:
            self.cleanupPage()
            return True
        return False

    def _clicked_record(self):
        self.button_record.setEnabled(False)

        self.context.signal_dry = (
            generate_capture_signal(self.context.sample_rate) * 0.94
        )

        self.signal_out_index = 0
        self.recorded_samples = []

        self.io_controller = prepare_play_record(
            self.context.sample_rate,
            self.context.input_channel,
            self.context.output_channel,
            self._input_callback,
            self._output_callback,
        )
        self.io_controller.start()

        self.bar_update_timer.start()

    def _input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        self.recorded_samples.append(
            indata[:, self.context.input_channel.channel_index - 1].copy()
        )

    def _output_callback(
        self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        channel = self.context.output_channel.channel_index - 1
        outdata.fill(0)
        if self.signal_out_index >= len(self.context.signal_dry):
            return
        segment = self.context.signal_dry[
            self.signal_out_index : self.signal_out_index + frames
        ]
        self.signal_out_index += frames
        outdata[0 : len(segment), channel] = segment

    def _update_status(self):
        self.bar_progress.setMaximum(len(self.context.signal_dry))
        self.bar_progress.setValue(self.signal_out_index)
        if self.signal_out_index >= len(self.context.signal_dry):
            self._complete()
            self.bar_update_timer.stop()

    def _complete(self):
        self.io_controller.close()
        self.context.signal_recorded = np.concat(self.recorded_samples).squeeze()
        self.completeChanged.emit()
