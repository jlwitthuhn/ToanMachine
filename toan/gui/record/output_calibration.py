# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record.context import RecordingContext

OUTPUT_CALIBRATION_CHOICE_TEXT = [
    "To record your gear, you will need to ensure that 0 dBFS on your interface produces a signal at a reasonable volume. If this is too quiet, the recording won't fully capture the gain behavior of your pedal. If the signal is too loud the model won't be accurate for lower guitar-level inputs.",
    "The best way to calibrate is to measure the RMS voltage of a 0 dBFS signal using a multimeter, but if you don't have a multimeter you can also try to calibrate it by ear.",
]


class RecordOutputCalibrationChoicePage(QtWidgets.QWizardPage):
    context: RecordingContext

    radio_by_ear: QtWidgets.QRadioButton
    radio_by_voltage: QtWidgets.QRadioButton

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Output Calibration")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(OUTPUT_CALIBRATION_CHOICE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        form_panel = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_panel)

        radio_widget = QtWidgets.QWidget(form_panel)
        radio_layout = QtWidgets.QVBoxLayout(radio_widget)
        radio_layout.setContentsMargins(0, 0, 0, 0)

        self.radio_by_ear = QtWidgets.QRadioButton("Calibrate by ear", radio_widget)
        self.radio_by_ear.setChecked(True)
        radio_layout.addWidget(self.radio_by_ear)
        self.radio_by_voltage = QtWidgets.QRadioButton(
            "Calibrate by voltage (requires multimeter)", radio_widget
        )
        radio_layout.addWidget(self.radio_by_voltage)

        form_layout.addRow("Calibration Mode:", radio_widget)

        layout.addWidget(form_panel)

        layout.addStretch(1)

    def is_voltage_selected(self) -> bool:
        return self.radio_by_voltage.isChecked()

    def validatePage(self) -> bool:
        if not self.is_voltage_selected():
            # Drop any voltage entered on an earlier pass through the wizard
            self.context.dbu = None
        return True
