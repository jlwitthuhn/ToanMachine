# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets

from toan.gui.record import RecordingContext
from toan.mix import concat_signals
from toan.signal import generate_chirp, generate_tone
from toan.soundio import SdPlayrecController, prepare_play_record

VOLUME_TEXT = [
    "Set the input gain on your interface so that you are able to record to full range of your pedal's output without clipping.",
    "With the audio signal running through your pedal, press 'Play Test Sound' below and adjust your input gain so that the volume is around 95.",
]

BAR_PRECISION = 1000


class RecordVolumePage(QtWidgets.QWizardPage):
    context: RecordingContext

    play_button: QtWidgets.QPushButton
    play_active: bool = False

    output_callback_signal: np.ndarray
    output_callback_signal_index: int = 0

    bar_input_level: QtWidgets.QProgressBar
    bar_progress: int = 0
    bar_update_timer: QtCore.QTimer
    text_volume: QtWidgets.QLineEdit

    io_controller: SdPlayrecController | None = None
    volume_samples: np.ndarray

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        single_sweep = generate_chirp(context.sample_rate, 18, 22000, 1.0, 0.8)
        single_sweep_samples = len(single_sweep)
        volume_buffer_samples = math.floor(single_sweep_samples * 1.10)
        self.volume_samples = np.zeros(volume_buffer_samples)

        self.output_callback_signal = concat_signals([single_sweep] * 30, 0)

        self.bar_update_timer = QtCore.QTimer()
        self.bar_update_timer.setInterval(50)
        self.bar_update_timer.setSingleShot(False)
        self.bar_update_timer.timeout.connect(self._update_status)

        self.setTitle("Set Volume")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(VOLUME_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.play_button = QtWidgets.QPushButton("Play Test Tone", self)
        self.play_button.clicked.connect(self._clicked_play_test)
        layout.addWidget(self.play_button)

        label_input_level = QtWidgets.QLabel("Input Level:", self)
        layout.addWidget(label_input_level)

        self.bar_input_level = QtWidgets.QProgressBar(self)
        self.bar_input_level.setMinimum(0)
        self.bar_input_level.setMaximum(BAR_PRECISION)
        layout.addWidget(self.bar_input_level)

        self.text_volume = QtWidgets.QLineEdit(self)
        self.text_volume.setFixedWidth(80)
        self.text_volume.setReadOnly(True)
        layout.addWidget(self.text_volume)

    def cleanupPage(self, /) -> None:
        if self.play_active:
            self._clicked_play_test()
        assert self.play_active == False

    def validatePage(self, /) -> bool:
        self.cleanupPage()
        return True

    def _clicked_play_test(self):
        if self.play_active:
            self.play_active = False
            self.play_button.setText("Play Test Sound")
            self.bar_update_timer.stop()
            if self.io_controller is not None:
                self.io_controller.close()
                self.io_controller = None
            self.bar_progress = 0
            self._update_status()
            return
        self.play_active = True
        self.play_button.setText("Stop Test Sound")
        self.bar_update_timer.start()
        self._setup_io_streams()

    def _setup_io_streams(self):
        self.io_controller = prepare_play_record(
            self.context.sample_rate,
            self.context.input_channel,
            self.context.output_channel,
            self._input_callback,
            self._output_callback,
        )
        self.io_controller.start()

    def _update_status(self):
        self.bar_input_level.setValue(self.bar_progress)
        self.text_volume.setText(f"{self.bar_progress / 10}")

    def _input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        buffer_length = len(self.volume_samples)
        self.volume_samples = np.append(self.volume_samples, indata)[-buffer_length:]
        out_val = self.volume_samples.max()
        self.bar_progress = min(BAR_PRECISION, math.floor(out_val * 1000))

    def _output_callback(
        self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        outdata.fill(0)

        if self.output_callback_signal_index + frames >= len(
            self.output_callback_signal
        ):
            self.output_callback_signal_index = 0
        segment = self.output_callback_signal[
            self.output_callback_signal_index : self.output_callback_signal_index
            + frames
        ]
        self.output_callback_signal_index += frames

        channel = self.context.output_channel.channel_index - 1
        outdata[:, channel] = segment
