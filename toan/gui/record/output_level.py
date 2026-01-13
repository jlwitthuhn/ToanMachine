# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import sounddevice as sd
from PySide6 import QtWidgets

from toan.gui.record import RecordingContext
from toan.music import get_note_frequency_by_name
from toan.signal.chord import generate_generic_chord_pluck
from toan.soundio import SdPlayrecController, prepare_play_record

OUTPUT_LEVEL_TEXT = [
    "In this section you will set the output level of your interface. For overdrive and similar effects it is very important to get this right to ensure that the input signal is loud enough to trigger the desired clipping effect.",
    "Click the 'Record' button to play some synthetic chords through your pedal and record the result. Then press 'Play' and the recorded result will be played back out.",
    "If the recording clips, you will need to turn down your input gain. Don't worry about setting it too precisely because you will calibrate it on the next screen.",
]


class RecordOutputLevelPage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_record: QtWidgets.QPushButton
    button_play: QtWidgets.QPushButton

    generated_chords: np.ndarray | None = None
    generated_chord_index: int = 0

    recorded_buffer: np.ndarray | None = None
    recorded_buffer_partial: np.ndarray | None = None

    io_controller: SdPlayrecController | None = None

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
        if self.recorded_buffer_partial is not None:
            return

        self.recorded_buffer = None
        self.recorded_buffer_partial = np.ndarray([0])
        self._refresh_buttons()

        if self.generated_chords is None:
            d_root = get_note_frequency_by_name("D", 3, 440.0)
            self.generated_chords = generate_generic_chord_pluck(
                self.context.sample_rate, [4, 7], d_root, 2.0
            )
        self.generated_chord_index = 0

        self.io_controller = prepare_play_record(
            self.context.sample_rate,
            self.context.input_channel,
            self.context.output_channel,
            self._record_input_callback,
            self._record_output_callback,
        )

    def _record_input_callback(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        pass

    def _record_output_callback(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        data.fill(0)

    def _refresh_buttons(self):
        play_enabled = True
        record_enabled = True

        if self.recorded_buffer is None:
            play_enabled = False

        if self.recorded_buffer_partial is not None:
            play_enabled = False
            record_enabled = False

        self.button_play.setEnabled(play_enabled)
        self.button_record.setEnabled(record_enabled)
