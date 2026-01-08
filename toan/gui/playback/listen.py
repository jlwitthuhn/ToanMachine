# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx
import numpy as np
import sounddevice as sd
from PySide6 import QtWidgets

from toan.gui.playback import PlaybackContext
from toan.model.nam_wavenet import NamWaveNet
from toan.soundio import SdChannel, generate_descriptions, get_input_devices

LISTEN_TEXT = [
    "This page allows you to asynchronously record a signal, then listen to how the model responds to that signal.",
]


class PlaybackListenPage(QtWidgets.QWizardPage):
    context: PlaybackContext
    indev_desc_map: dict[str, SdChannel]
    indev_combo: QtWidgets.QComboBox

    recorded_samples_label: QtWidgets.QLabel
    record_button: QtWidgets.QPushButton

    record_active: bool = False
    record_channel: SdChannel | None = None
    record_stream: sd.InputStream | None = None
    recorded_samples: list[np.ndarray]

    wet_signal: np.ndarray | None = None

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

        input_devices = get_input_devices()
        descriptions, self.indev_desc_map = generate_descriptions(
            input_devices, include_out=False
        )

        self.indev_combo = QtWidgets.QComboBox(self.record_group)
        self.indev_combo.addItems(descriptions)
        record_group_layout.addRow("Device:", self.indev_combo)

        self.recorded_samples_label = QtWidgets.QLabel("0", self.record_group)
        record_group_layout.addRow("Recorded Samples:", self.recorded_samples_label)

        self.record_button = QtWidgets.QPushButton("Start Recording", self.record_group)
        self.record_button.clicked.connect(self._clicked_record)
        record_group_layout.addRow("", self.record_button)

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
        if self.wet_signal is None or len(self.wet_signal) == 0:
            return
        signal = _convert_complete_signal(self.wet_signal, self.context.nam_model)
        sd.play(signal, self.context.sample_rate)

    def _clicked_record(self):
        if not self.record_active:
            desc = self.indev_combo.currentText()
            if desc not in self.indev_desc_map:
                return
            self.record_active = True
            self.record_channel = self.indev_desc_map[desc]
            self.recorded_samples = []
            self.record_button.setText("Stop Recording")
            self.record_stream = sd.InputStream(
                samplerate=self.context.sample_rate,
                device=self.record_channel.device_index,
                callback=self._record_input_callback,
            )
            self.record_stream.start()
        else:
            self.wet_signal = np.concat(self.recorded_samples).squeeze()
            self.recorded_samples_label.setText(f"{len(self.wet_signal)}")
            self._stop_all()
            self.record_active = False
            self.record_button.setText("Start Recording")

    def _clicked_stop(self):
        self._stop_all()

    def _stop_all(self):
        if self.record_active:
            self.record_stream.stop()
        sd.stop()

    def _record_input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        self.recorded_samples.append(
            indata[:, self.record_channel.channel_index - 1].copy()
        )


def _convert_complete_signal(raw_signal: np.ndarray, model: NamWaveNet) -> np.ndarray:
    output = model(mx.array(raw_signal).reshape(1, -1))
    return np.array(output.squeeze())
