# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
from typing import Callable

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtGui, QtWidgets

from toan.gui.record.context import RecordingContext
from toan.gui.record.voltage import (
    TONE_FREQUENCY,
    UNIT_MILLIVOLTS_RMS,
    VOLTAGE_UNITS,
    to_dbu,
)
from toan.signal.generator.trig import generate_sine_wave
from toan.soundio import SdChannel, SdIoController

INPUT_VOLTAGE_TEXT = [
    "Here you will find the voltage level that corresponds to 0 dBFS on your interface's input. Do not touch your interface's input or output gain.",
    "First, connect the output directly to the input and play the test tone to measure the input dBFS. Once that is recorded below, unplug the cable from your interface's input and measure the voltage across its terminals.",
    "Plug your pedal back in again when you have entered both values.",
]

# The meter bar is stored in tenths of a dB
METER_FLOOR_DB = -60.0
METER_PRECISION = 10

# Peaks at or above this are treated as clipping
CLIPPING_PEAK = 0.999

OUTPUT_SCALE_MIN_DB = -40
OUTPUT_SCALE_MAX_DB = 0


def _peak_to_dbfs(peak: float) -> float:
    if peak <= 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


class InputVoltageWidget(QtWidgets.QWidget):
    # Emitted when the entered level, voltage, or unit changes
    measurement_changed = QtCore.Signal()

    sample_rate: int
    # Returns the (input, output) channels to use when the tone starts
    get_channels: Callable[[], tuple[SdChannel, SdChannel]]
    input_channel: SdChannel
    output_channel: SdChannel

    play_button: QtWidgets.QPushButton
    play_active: bool = False

    tone_signal: np.ndarray
    tone_signal_index: int = 0
    output_scale: float = 1.0

    slider_output_scale: QtWidgets.QSlider
    label_output_scale: QtWidgets.QLabel

    bar_input_level: QtWidgets.QProgressBar
    bar_update_timer: QtCore.QTimer
    text_input_level: QtWidgets.QLineEdit
    label_clipping: QtWidgets.QLabel

    io_controller: SdIoController | None = None
    peak_samples: np.ndarray
    input_peak: float = 0.0

    text_level: QtWidgets.QLineEdit
    text_voltage: QtWidgets.QLineEdit
    combo_unit: QtWidgets.QComboBox

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        sample_rate: int,
        get_channels: Callable[[], tuple[SdChannel, SdChannel]],
    ):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.get_channels = get_channels

        self.tone_signal = generate_sine_wave(
            sample_rate * 10, sample_rate // TONE_FREQUENCY
        )
        self.peak_samples = np.zeros(sample_rate // 4)

        self.bar_update_timer = QtCore.QTimer()
        self.bar_update_timer.setInterval(100)
        self.bar_update_timer.setSingleShot(False)
        self.bar_update_timer.timeout.connect(self._update_status)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tone controls and the meter share a grid so the slider lines up with
        # the bar and the scale value lines up with the level reading
        meter_panel = QtWidgets.QWidget(self)
        meter_layout = QtWidgets.QGridLayout(meter_panel)
        meter_layout.setContentsMargins(0, 0, 0, 0)
        meter_layout.setColumnStretch(2, 1)

        self.play_button = QtWidgets.QPushButton("Play Test Tone", meter_panel)
        self.play_button.clicked.connect(self._clicked_play_tone)
        meter_layout.addWidget(self.play_button, 0, 0)

        label_output_scale_name = QtWidgets.QLabel("Output Scale:", meter_panel)
        meter_layout.addWidget(
            label_output_scale_name, 0, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.slider_output_scale = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, meter_panel
        )
        self.slider_output_scale.setRange(OUTPUT_SCALE_MIN_DB, OUTPUT_SCALE_MAX_DB)
        self.slider_output_scale.setValue(OUTPUT_SCALE_MAX_DB)
        self.slider_output_scale.valueChanged.connect(self._output_scale_changed)
        meter_layout.addWidget(self.slider_output_scale, 0, 2)

        self.label_output_scale = QtWidgets.QLabel("", meter_panel)
        meter_layout.addWidget(self.label_output_scale, 0, 3)

        label_input_level = QtWidgets.QLabel("Input Level (dBFS peak):", meter_panel)
        meter_layout.addWidget(
            label_input_level, 1, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.bar_input_level = QtWidgets.QProgressBar(meter_panel)
        self.bar_input_level.setMinimum(round(METER_FLOOR_DB * METER_PRECISION))
        self.bar_input_level.setMaximum(0)
        self.bar_input_level.setTextVisible(False)
        meter_layout.addWidget(self.bar_input_level, 1, 2)

        self.text_input_level = QtWidgets.QLineEdit(meter_panel)
        self.text_input_level.setFixedWidth(60)
        self.text_input_level.setReadOnly(True)
        meter_layout.addWidget(self.text_input_level, 1, 3)

        layout.addWidget(meter_panel)

        self.label_clipping = QtWidgets.QLabel(
            "Clipping, reduce the output scale", self
        )
        self.label_clipping.setVisible(False)
        layout.addWidget(self.label_clipping)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        form_panel = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_panel)

        level_row = QtWidgets.QWidget(form_panel)
        level_row_layout = QtWidgets.QHBoxLayout(level_row)
        level_row_layout.setContentsMargins(0, 0, 0, 0)

        self.text_level = QtWidgets.QLineEdit(level_row)
        self.text_level.setFixedWidth(80)
        self.text_level.setValidator(QtGui.QDoubleValidator(self.text_level))
        self.text_level.textChanged.connect(self.measurement_changed)
        level_row_layout.addWidget(self.text_level)
        level_row_layout.addWidget(QtWidgets.QLabel("dBFS peak", level_row))
        level_row_layout.addStretch(1)

        form_layout.addRow("Level:", level_row)

        voltage_row = QtWidgets.QWidget(form_panel)
        voltage_row_layout = QtWidgets.QHBoxLayout(voltage_row)
        voltage_row_layout.setContentsMargins(0, 0, 0, 0)

        self.text_voltage = QtWidgets.QLineEdit(voltage_row)
        self.text_voltage.setFixedWidth(80)
        self.text_voltage.setValidator(QtGui.QDoubleValidator(self.text_voltage))
        self.text_voltage.textChanged.connect(self.measurement_changed)
        voltage_row_layout.addWidget(self.text_voltage)

        self.combo_unit = QtWidgets.QComboBox(voltage_row)
        self.combo_unit.addItems(VOLTAGE_UNITS)
        self.combo_unit.currentTextChanged.connect(self.measurement_changed)
        voltage_row_layout.addWidget(self.combo_unit)
        voltage_row_layout.addStretch(1)

        form_layout.addRow("Voltage:", voltage_row)

        layout.addWidget(form_panel)

        self._output_scale_changed(self.slider_output_scale.value())
        self._update_status()

    def clear(self):
        self.text_level.clear()
        self.text_voltage.clear()
        self.combo_unit.setCurrentText(UNIT_MILLIVOLTS_RMS)

    def stop_tone(self):
        if self.play_active:
            self._clicked_play_tone()
        assert self.play_active == False

    def computed_dbu(self) -> float | None:
        level_dbfs = self._entered_level_dbfs()
        voltage_dbu = self._entered_voltage_dbu()
        if level_dbfs is None or voltage_dbu is None:
            return None
        # A sine's RMS voltage scales with its peak level, so extrapolate the
        # measured tone up to one whose peaks reach 0 dBFS
        return voltage_dbu - level_dbfs

    def _entered_level_dbfs(self) -> float | None:
        try:
            value = float(self.text_level.text())
        except ValueError:
            return None
        if value > 0.0:
            return None
        return value

    def _entered_voltage_dbu(self) -> float | None:
        try:
            value = float(self.text_voltage.text())
        except ValueError:
            return None
        return to_dbu(value, self.combo_unit.currentText())

    def _output_scale_changed(self, value_db: int):
        self.output_scale = 10.0 ** (value_db / 20.0)
        self.label_output_scale.setText(f"{value_db} dB")

    def _clicked_play_tone(self):
        if self.play_active:
            self.play_active = False
            self.play_button.setText("Play Test Tone")
            self.bar_update_timer.stop()
            if self.io_controller is not None:
                self.io_controller.close()
                self.io_controller = None
            self.input_peak = 0.0
            self._update_status()
            return
        self.input_channel, self.output_channel = self.get_channels()
        self.play_active = True
        self.play_button.setText("Stop Test Tone")
        self.tone_signal_index = 0
        self.peak_samples.fill(0)
        self.input_peak = 0.0
        self.bar_update_timer.start()
        self._setup_io_streams()

    def _setup_io_streams(self):
        self.io_controller = SdIoController.from_callbacks(
            self.sample_rate,
            self.input_channel,
            self.output_channel,
            self._input_callback,
            self._output_callback,
        )
        self.io_controller.start()

    def _update_status(self):
        peak_db = _peak_to_dbfs(self.input_peak)
        bar_db = max(peak_db, METER_FLOOR_DB)
        self.bar_input_level.setValue(round(bar_db * METER_PRECISION))

        if self.play_active:
            self.text_input_level.setText(f"{peak_db:.1f}")
        else:
            self.text_input_level.clear()

        self.label_clipping.setVisible(
            self.play_active and self.input_peak >= CLIPPING_PEAK
        )

    def _input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        buffer_length = len(self.peak_samples)
        self.peak_samples = np.concat(
            (
                self.peak_samples,
                indata[:, self.input_channel.channel_index - 1],
            )
        )[-buffer_length:]
        self.input_peak = float(np.max(np.abs(self.peak_samples)))

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

        channel = self.output_channel.channel_index - 1
        outdata[:, channel] = segment * self.output_scale


class RecordInputVoltagePage(QtWidgets.QWizardPage):
    context: RecordingContext
    voltage_widget: InputVoltageWidget

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Input Voltage")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(INPUT_VOLTAGE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.voltage_widget = InputVoltageWidget(
            self,
            context.sample_rate,
            lambda: (self.context.input_channel, self.context.output_channel),
        )
        self.voltage_widget.measurement_changed.connect(self.completeChanged)
        layout.addWidget(self.voltage_widget)

        layout.addStretch(1)

    def initializePage(self):
        # Any earlier measurement was taken before the input gain page was
        # revisited, so it may no longer be valid
        self.voltage_widget.clear()

    def isComplete(self) -> bool:
        return self.voltage_widget.computed_dbu() is not None

    def cleanupPage(self):
        self.voltage_widget.stop_tone()

    def validatePage(self) -> bool:
        dbu = self.voltage_widget.computed_dbu()
        if dbu is None:
            return False
        self.cleanupPage()
        self.context.output_level_dbu = dbu
        return True
