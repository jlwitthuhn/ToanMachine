# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record.context import RecordingContext

OUTPUT_CALIBRATION_CHOICE_TEXT = [
    "To record your gear, you will need to ensure that 0 dBFS on your interface produces a signal at a volume high enough to cover what a guitar pickup might output.",
    "The easiest way to do this is to set your interface's output level to maximum. This will ensure the signal is strong enough to capture the full behavior of your equipment.",
    "If you have a multimeter, you can make this a calibrated NAM capture by measuring this output voltage.",
]


class RecordOutputCalibrationChoicePage(QtWidgets.QWizardPage):
    context: RecordingContext

    checkbox_output_level_set: QtWidgets.QCheckBox
    checkbox_calibrated: QtWidgets.QCheckBox

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

        self.checkbox_output_level_set = QtWidgets.QCheckBox(
            "I have set my output level high enough", self
        )
        self.checkbox_output_level_set.toggled.connect(self.completeChanged)
        layout.addWidget(self.checkbox_output_level_set)

        self.checkbox_calibrated = QtWidgets.QCheckBox(
            "Calibrated capture (requires multimeter)", self
        )
        layout.addWidget(self.checkbox_calibrated)

        layout.addStretch(1)

    def isComplete(self) -> bool:
        return self.checkbox_output_level_set.isChecked()

    def is_voltage_selected(self) -> bool:
        return self.checkbox_calibrated.isChecked()

    def validatePage(self) -> bool:
        if not self.is_voltage_selected():
            # Drop any voltages entered on an earlier pass through the wizard
            self.context.input_level_dbu = None
            self.context.output_level_dbu = None
        return True
