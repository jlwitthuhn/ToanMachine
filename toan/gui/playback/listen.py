# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx
import numpy as np
import sounddevice as sd
from PySide6 import QtWidgets

from toan.gui.playback import PlaybackContext
from toan.model.nam_wavenet import NamWaveNet
from toan.signal import generate_capture_signal

LISTEN_TEXT = [
    "This page allows you to asynchronously record a signal, then listen to how the model responds to that signal.",
]


class PlaybackListenPage(QtWidgets.QWizardPage):
    context: PlaybackContext

    def __init__(self, parent, context: PlaybackContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Listen")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(LISTEN_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.record_group = QtWidgets.QGroupBox("Record", self)
        record_group_layout = QtWidgets.QFormLayout(self.record_group)

        device_combo = QtWidgets.QComboBox(self.record_group)
        record_group_layout.addRow("Device:", device_combo)

        record_button = QtWidgets.QPushButton("Record", self.record_group)
        record_group_layout.addRow("", record_button)

        self.play_group = QtWidgets.QGroupBox("Playback", self)
        play_group_layout = QtWidgets.QVBoxLayout(self.play_group)

        self.play_button = QtWidgets.QPushButton("Play", self.play_group)
        self.play_button.clicked.connect(self._clicked_play)
        play_group_layout.addWidget(self.play_button)

        self.stop_button = QtWidgets.QPushButton("Stop Audio", self.play_group)
        self.stop_button.clicked.connect(self._clicked_stop)
        play_group_layout.addWidget(self.stop_button)

        layout.addWidget(self.record_group)
        layout.addWidget(self.play_group)

    def cleanupPage(self):
        self._stop_all()

    def validatePage(self):
        self.cleanupPage()
        return True

    def _clicked_play(self):
        raw_signal = generate_capture_signal(self.context.sample_rate, 0.8)
        signal = _convert_complete_signal(raw_signal, self.context.nam_model)
        sd.play(signal, self.context.sample_rate)

    def _clicked_stop(self):
        self._stop_all()

    def _stop_all(self):
        sd.stop()


def _convert_complete_signal(raw_signal: np.ndarray, model: NamWaveNet) -> np.ndarray:
    output = model(mx.array(raw_signal).reshape(1, -1))
    return np.array(output.squeeze())
