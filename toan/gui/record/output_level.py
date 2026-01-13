# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
from PySide6 import QtWidgets

from toan.gui.record import RecordingContext

OUTPUT_LEVEL_TEXT = [
    "In this section you will set the output level of your interface. For overdrive and similar effects it is very important to get this right to ensure that the input signal is loud enough to trigger the desired clipping effect.",
    "Click the 'Record' button to play some synthetic chords through your pedal and record the result. Then press 'Play' and the recorded result will be played back out.",
    "If the recording clips, you will need to turn down your input gain. Don't worry about setting it too precisely because you will calibrate it on the next screen.",
]


class RecordOutputLevelPage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_record: QtWidgets.QPushButton
    button_play: QtWidgets.QPushButton

    recorded_buffer: np.ndarray | None = None
    recording: bool = False

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)

        self.setTitle("Output Level")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(OUTPUT_LEVEL_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        button_panel = QtWidgets.QWidget(self)
        button_panel_layout = QtWidgets.QHBoxLayout(button_panel)
        button_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.button_record = QtWidgets.QPushButton("Record", button_panel)
        self.button_record.clicked.connect(self._pressed_record)
        button_panel_layout.addWidget(self.button_record)

        self.button_play = QtWidgets.QPushButton("Play", button_panel)
        button_panel_layout.addWidget(self.button_play)

        layout.addWidget(button_panel)

        progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(progress_bar)

        self._refresh_buttons()

    def _pressed_record(self):
        if self.recording:
            return

        self.recording = True
        self.recorded_buffer = None
        self._refresh_buttons()

    def _refresh_buttons(self):
        play_enabled = True
        record_enabled = True

        if self.recording:
            play_enabled = False
            record_enabled = False

        if self.recorded_buffer is None:
            play_enabled = False

        self.button_play.setEnabled(play_enabled)
        self.button_record.setEnabled(record_enabled)
