# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.record.input_voltage import InputVoltageWidget
from toan.soundio import (
    SdChannel,
    generate_descriptions,
    get_input_devices,
    get_output_devices,
)

SAMPLE_RATE = 48000


class InputLevelHelper(QtWidgets.QDialog):
    input_channels: dict[str, SdChannel]
    output_channels: dict[str, SdChannel]

    combo_output: QtWidgets.QComboBox
    combo_input: QtWidgets.QComboBox

    voltage_widget: InputVoltageWidget
    text_dbu: QtWidgets.QLineEdit

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.setWindowTitle("Input Level Helper")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        device_panel = QtWidgets.QWidget(self)
        device_layout = QtWidgets.QFormLayout(device_panel)

        self.combo_output = QtWidgets.QComboBox(device_panel)
        output_labels, self.output_channels = generate_descriptions(
            get_output_devices(), include_in=False, include_out=True
        )
        self.combo_output.addItems(output_labels)
        self.combo_output.currentTextChanged.connect(self._device_changed)
        device_layout.addRow("Output Device:", self.combo_output)

        self.combo_input = QtWidgets.QComboBox(device_panel)
        input_labels, self.input_channels = generate_descriptions(
            get_input_devices(), include_in=True, include_out=False
        )
        self.combo_input.addItems(input_labels)
        self.combo_input.currentTextChanged.connect(self._device_changed)
        device_layout.addRow("Input Device:", self.combo_input)

        layout.addWidget(device_panel)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.voltage_widget = InputVoltageWidget(
            self, SAMPLE_RATE, self._selected_channels
        )
        self.voltage_widget.measurement_changed.connect(self._update_result)
        layout.addWidget(self.voltage_widget)

        hline2 = QtWidgets.QFrame(self)
        hline2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline2)

        result_panel = QtWidgets.QWidget(self)
        result_layout = QtWidgets.QFormLayout(result_panel)

        result_row = QtWidgets.QWidget(result_panel)
        result_row_layout = QtWidgets.QHBoxLayout(result_row)
        result_row_layout.setContentsMargins(0, 0, 0, 0)

        self.text_dbu = QtWidgets.QLineEdit(result_row)
        self.text_dbu.setFixedWidth(80)
        self.text_dbu.setReadOnly(True)
        result_row_layout.addWidget(self.text_dbu)
        result_row_layout.addWidget(QtWidgets.QLabel("dBu", result_row))
        result_row_layout.addStretch(1)

        result_layout.addRow("Max Input Level:", result_row)

        layout.addWidget(result_panel)

        layout.addStretch(1)

    def done(self, result: int):
        self.voltage_widget.stop_tone()
        super().done(result)

    def _selected_channels(self) -> tuple[SdChannel, SdChannel]:
        return (
            self.input_channels[self.combo_input.currentText()],
            self.output_channels[self.combo_output.currentText()],
        )

    def _device_changed(self):
        # A playing tone is still using the previously selected devices
        self.voltage_widget.stop_tone()

    def _update_result(self):
        dbu = self.voltage_widget.computed_dbu()
        if dbu is None:
            self.text_dbu.clear()
        else:
            self.text_dbu.setText(f"{dbu:.2f}")
