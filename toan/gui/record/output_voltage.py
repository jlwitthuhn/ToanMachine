# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np
import sounddevice as sd
from PySide6 import QtGui, QtWidgets

from toan.gui.record.context import RecordingContext
from toan.signal.generator.trig import generate_sine_wave

OUTPUT_VOLTAGE_TEXT = [
    "Set you interface's output voltage here.",
    "This step requires that you own a multimeter or otherwise can figure out your interface's output level. If you cannot determine this, skip this step.",
    "Be sure to measure straight out of your interface, after the reamp box if you are using one, before it goes through any of the pedals/amps you want to capture.",
]

TONE_TEXT = "Press 'Play Test Tone' to output a 300Hz signal, then measure your interface's output with a multimeter."

TONE_FREQUENCY = 300

# 0 dBu is 0.775 V RMS
DBU_REFERENCE_VOLTS = math.sqrt(0.6)

UNIT_MILLIVOLTS_RMS = "Millivolts RMS"
UNIT_VOLTS_RMS = "Volts RMS"
UNIT_DBU = "dBu"


def _to_dbu(value: float, unit: str) -> float | None:
    if unit == UNIT_DBU:
        return value
    if unit == UNIT_MILLIVOLTS_RMS:
        volts = value / 1000.0
    elif unit == UNIT_VOLTS_RMS:
        volts = value
    else:
        return None
    if volts <= 0:
        return None
    return 20.0 * math.log10(volts / DBU_REFERENCE_VOLTS)


def _dbu_to_millivolts_rms(dbu: float) -> float:
    return DBU_REFERENCE_VOLTS * (10.0 ** (dbu / 20.0)) * 1000.0


class RecordOutputVoltagePage(QtWidgets.QWizardPage):
    context: RecordingContext

    play_button: QtWidgets.QPushButton
    play_active: bool = False

    output_stream: sd.OutputStream | None = None
    tone_signal: np.ndarray
    tone_signal_index: int = 0

    checkbox_calibration: QtWidgets.QCheckBox
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

        self.checkbox_calibration = QtWidgets.QCheckBox("Use Voltage Calibration", self)
        self.checkbox_calibration.toggled.connect(self._calibration_toggled)
        layout.addWidget(self.checkbox_calibration)

        form_panel = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_panel)

        voltage_row = QtWidgets.QWidget(form_panel)
        voltage_row_layout = QtWidgets.QHBoxLayout(voltage_row)
        voltage_row_layout.setContentsMargins(0, 0, 0, 0)

        self.text_voltage = QtWidgets.QLineEdit(voltage_row)
        self.text_voltage.setFixedWidth(80)
        self.text_voltage.setValidator(QtGui.QDoubleValidator(self.text_voltage))
        voltage_row_layout.addWidget(self.text_voltage)

        self.combo_unit = QtWidgets.QComboBox(voltage_row)
        self.combo_unit.addItems([UNIT_MILLIVOLTS_RMS, UNIT_VOLTS_RMS, UNIT_DBU])
        voltage_row_layout.addWidget(self.combo_unit)

        form_layout.addRow("Output Voltage:", voltage_row)

        layout.addWidget(form_panel)

        layout.addStretch(1)

        self._calibration_toggled(self.checkbox_calibration.isChecked())

    def initializePage(self):
        self.combo_unit.setCurrentText(UNIT_MILLIVOLTS_RMS)
        if self.context.dbu is not None:
            self.checkbox_calibration.setChecked(True)
            millivolts = _dbu_to_millivolts_rms(self.context.dbu)
            self.text_voltage.setText(f"{millivolts:.4g}")
        else:
            self.checkbox_calibration.setChecked(False)
            self.text_voltage.clear()
        self._calibration_toggled(self.checkbox_calibration.isChecked())

    def _calibration_toggled(self, checked: bool):
        self.text_voltage.setEnabled(checked)
        self.combo_unit.setEnabled(checked)

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
        self.cleanupPage()
        if self.checkbox_calibration.isChecked():
            try:
                value = float(self.text_voltage.text())
            except ValueError:
                self.context.dbu = None
            else:
                self.context.dbu = _to_dbu(value, self.combo_unit.currentText())
        else:
            self.context.dbu = None
        return True
