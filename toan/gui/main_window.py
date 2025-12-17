# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import sounddevice as sd
from PySide6 import QtWidgets

from toan.gui.record import RecordWizard
from toan.signal import generate_capture_signal


def _clicked_play_training_signal():
    playback_sample_rate = 44100
    signal = generate_capture_signal(playback_sample_rate, 0.5)
    sd.play(signal, playback_sample_rate)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout()

        create_group_box = QtWidgets.QGroupBox("Create Model", self)
        create_layout = QtWidgets.QVBoxLayout()

        record_button = QtWidgets.QPushButton("Record Device", self)
        record_button.clicked.connect(self._clicked_record_device)
        create_layout.addWidget(record_button)

        train_button = QtWidgets.QPushButton("Train Model", self)
        create_layout.addWidget(train_button)

        create_group_box.setLayout(create_layout)
        layout.addWidget(create_group_box)

        debug_group_box = QtWidgets.QGroupBox("Debug", self)
        debug_layout = QtWidgets.QVBoxLayout()

        play_training_signal_button = QtWidgets.QPushButton(
            "Play Training Signal", self
        )
        play_training_signal_button.clicked.connect(_clicked_play_training_signal)
        debug_layout.addWidget(play_training_signal_button)

        debug_group_box.setLayout(debug_layout)
        layout.addWidget(debug_group_box)

        self.setLayout(layout)

    def _clicked_record_device(self):
        wizard = RecordWizard(self)
        wizard.show()
