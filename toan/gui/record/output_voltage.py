# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import sounddevice as sd
from PySide6 import QtGui, QtWidgets

from toan.gui.record.context import RecordingContext
from toan.gui.record.voltage import (
    TONE_FREQUENCY,
    UNIT_MILLIVOLTS_RMS,
    VOLTAGE_UNITS,
    dbu_to_millivolts_rms,
    to_dbu,
)
from toan.signal.generator.trig import generate_sine_wave

OUTPUT_VOLTAGE_TEXT = [
    "This step requires that you own a multimeter or otherwise can figure out your interface's output voltage.",
    "Measure the voltage going in to the device you are capturing. If your signal chain includes a reamp box, be sure to measure what is coming out of the reamp rather than what is going in to it.",
]

TONE_TEXT = "Press 'Play Test Tone' to output a 300Hz sine wave, then measure your interface's output with a multimeter."


class RecordOutputVoltagePage(QtWidgets.QWizardPage):
    context: RecordingContext

    play_button: QtWidgets.QPushButton
    play_active: bool = False

    output_stream: sd.OutputStream | None = None
    tone_signal: np.ndarray
    tone_signal_index: int = 0

    text_voltage: QtWidgets.QLineEdit
    combo_unit: QtWidgets.QComboBox

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.tone_signal = generate_sine_wave(
            context.sample_rate * 10, context.sample_rate // TONE_FREQUENCY
        )

        self.setTitle("Output Voltage")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(OUTPUT_VOLTAGE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        tone_label = QtWidgets.QLabel(TONE_TEXT, self)
        tone_label.setWordWrap(True)
        layout.addWidget(tone_label)

        self.play_button = QtWidgets.QPushButton("Play Test Tone", self)
        self.play_button.clicked.connect(self._clicked_play_tone)
        layout.addWidget(self.play_button)

        hline2 = QtWidgets.QFrame(self)
        hline2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline2)

        form_panel = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_panel)

        voltage_row = QtWidgets.QWidget(form_panel)
        voltage_row_layout = QtWidgets.QHBoxLayout(voltage_row)
        voltage_row_layout.setContentsMargins(0, 0, 0, 0)

        self.text_voltage = QtWidgets.QLineEdit(voltage_row)
        self.text_voltage.setFixedWidth(80)
        self.text_voltage.setValidator(QtGui.QDoubleValidator(self.text_voltage))
        self.text_voltage.textChanged.connect(self.completeChanged)
        voltage_row_layout.addWidget(self.text_voltage)

        self.combo_unit = QtWidgets.QComboBox(voltage_row)
        self.combo_unit.addItems(VOLTAGE_UNITS)
        self.combo_unit.currentTextChanged.connect(self.completeChanged)
        voltage_row_layout.addWidget(self.combo_unit)

        form_layout.addRow("Output Voltage:", voltage_row)

        layout.addWidget(form_panel)

        layout.addStretch(1)

    def initializePage(self):
        self.combo_unit.setCurrentText(UNIT_MILLIVOLTS_RMS)
        if self.context.input_level_dbu is not None:
            millivolts = dbu_to_millivolts_rms(self.context.input_level_dbu)
            self.text_voltage.setText(f"{millivolts:.4g}")
        else:
            self.text_voltage.clear()

    def isComplete(self) -> bool:
        return self._entered_dbu() is not None

    def _entered_dbu(self) -> float | None:
        try:
            value = float(self.text_voltage.text())
        except ValueError:
            return None
        return to_dbu(value, self.combo_unit.currentText())

    def _clicked_play_tone(self):
        if self.play_active:
            self._stop_tone()
            return
        self.play_active = True
        self.play_button.setText("Stop Test Tone")
        self.tone_signal_index = 0
        self.output_stream = sd.OutputStream(
            samplerate=self.context.sample_rate,
            device=self.context.output_channel.device_index,
            callback=self._output_callback,
        )
        self.output_stream.start()

    def _stop_tone(self):
        self.play_active = False
        self.play_button.setText("Play Test Tone")
        if self.output_stream is not None:
            self.output_stream.close()
            self.output_stream = None

    def _output_callback(
        self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        outdata.fill(0)

        if self.tone_signal_index + frames >= len(self.tone_signal):
            self.tone_signal_index = 0
        segment = self.tone_signal[
            self.tone_signal_index : self.tone_signal_index + frames
        ]
        self.tone_signal_index += frames

        channel = self.context.output_channel.channel_index - 1
        outdata[:, channel] = segment

    def cleanupPage(self):
        if self.play_active:
            self._stop_tone()

    def validatePage(self) -> bool:
        dbu = self._entered_dbu()
        if dbu is None:
            return False
        self.cleanupPage()
        self.context.input_level_dbu = dbu
        return True
