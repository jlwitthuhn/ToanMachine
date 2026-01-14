# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets

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

    refresh_timer: QtCore.QTimer

    button_record: QtWidgets.QPushButton
    button_play: QtWidgets.QPushButton
    progress_bar: QtWidgets.QProgressBar

    generated_chords: np.ndarray | None = None
    generated_chord_index: int = 0

    recorded_buffer: np.ndarray | None = None
    recorded_buffer_partial: np.ndarray | None = None
    recorded_playback_index: int = 0

    record_controller: SdPlayrecController | None = None
    play_controller: sd.OutputStream | None = None

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self._refresh_timer_triggered)
        self.refresh_timer.start()

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
        self.button_play.clicked.connect(self._pressed_play)
        button_panel_layout.addWidget(self.button_play)

        layout.addWidget(button_panel)

        self.progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self._refresh_buttons()

    def _pressed_play(self):
        if self.recorded_buffer is None:
            return
        if self.record_controller is not None:
            return
        if self.play_controller is not None:
            return

        self.recorded_playback_index = 0
        self.play_controller = sd.OutputStream(
            samplerate=self.context.sample_rate,
            device=self.context.output_channel.device_index,
            callback=self._play_output_callback,
        )
        self.play_controller.start()

    def _pressed_record(self):
        if self.recorded_buffer_partial is not None:
            return
        if self.record_controller is not None:
            return
        if self.play_controller is not None:
            return

        self.recorded_buffer = None
        self.recorded_buffer_partial = np.ndarray([0])
        self._refresh_buttons()

        if self.generated_chords is None:
            d_root = get_note_frequency_by_name("D", 3, 440.0)
            d_chord_raw = generate_generic_chord_pluck(
                self.context.sample_rate, [4, 7], d_root, 0.9
            )
            d_chord_2 = d_chord_raw * 0.80
            d_chord_3 = d_chord_2 * 0.80
            self.generated_chords = np.concat(
                (
                    np.zeros(self.context.sample_rate // 2),
                    d_chord_3,
                    d_chord_2,
                    d_chord_raw,
                    np.zeros(self.context.sample_rate // 2),
                )
            )
        self.generated_chord_index = 0

        self.record_controller = prepare_play_record(
            self.context.sample_rate,
            self.context.input_channel,
            self.context.output_channel,
            self._record_input_callback,
            self._record_output_callback,
        )
        self.record_controller.start()

    def _play_output_callback(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        channel = self.context.output_channel.channel_index - 1
        data.fill(0)
        if self.recorded_playback_index >= len(self.recorded_buffer):
            return
        segment = self.recorded_buffer[
            self.recorded_playback_index : self.recorded_playback_index + frames
        ]
        self.recorded_playback_index += frames
        data[0 : len(segment), channel] = segment

    def _record_input_callback(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        if self.recorded_buffer_partial is None:
            # We are in the brief period between when recording is complete and when the callback closes audio IO
            return
        channel_data = data[:, self.context.input_channel.channel_index - 1]
        self.recorded_buffer_partial = np.concat(
            (self.recorded_buffer_partial, channel_data)
        )
        if len(self.recorded_buffer_partial) >= len(self.generated_chords):
            self.recorded_buffer = self.recorded_buffer_partial[
                : len(self.generated_chords)
            ]
            self.recorded_buffer_partial = None

    def _record_output_callback(
        self, data: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        channel = self.context.output_channel.channel_index - 1
        data.fill(0)
        if self.generated_chord_index >= len(self.generated_chords):
            # Playback is complete, only send zeros now
            return
        segment = self.generated_chords[
            self.generated_chord_index : self.generated_chord_index + frames
        ]
        self.generated_chord_index += frames
        data[0 : len(segment), channel] = segment

    def _refresh_timer_triggered(self):
        # Progress bar first
        if self.generated_chords is not None:
            self.progress_bar.setMaximum(len(self.generated_chords))
            if self.recorded_buffer_partial is not None:
                self.progress_bar.setValue(len(self.recorded_buffer_partial))
            else:
                self.progress_bar.setValue(0)

        if (
            self.recorded_buffer_partial is None
            and self.recorded_buffer is not None
            and self.record_controller is not None
        ):
            # Recording has ended, kill the stream
            self.record_controller.close()
            self.record_controller = None

        if self.play_controller is not None and self.recorded_playback_index >= len(
            self.recorded_buffer
        ):
            # Playback has ended, kill the stream
            self.play_controller.close()
            self.play_controller = None

        self._refresh_buttons()

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
