# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtGui, QtWidgets

from toan.gui.record.context import RecordingContext

OUTPUT_VOLTAGE_TEXT = [
    "Set you interface's output voltage here.",
    "This step requires that you own a multimeter or otherwise can figure out your interface's output level. If you cannot determine this, skip this step.",
]


class RecordOutputVoltagePage(QtWidgets.QWizardPage):
    context: RecordingContext

    checkbox_calibration: QtWidgets.QCheckBox
    text_dbu: QtWidgets.QLineEdit

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Output Voltage")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(OUTPUT_VOLTAGE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.checkbox_calibration = QtWidgets.QCheckBox("Use Voltage Calibration", self)
        self.checkbox_calibration.toggled.connect(self._calibration_toggled)
        layout.addWidget(self.checkbox_calibration)

        form_panel = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_panel)

        self.text_dbu = QtWidgets.QLineEdit(form_panel)
        self.text_dbu.setFixedWidth(80)
        self.text_dbu.setValidator(QtGui.QDoubleValidator(self.text_dbu))
        form_layout.addRow("dBu:", self.text_dbu)

        layout.addWidget(form_panel)

        layout.addStretch(1)

        self._calibration_toggled(self.checkbox_calibration.isChecked())

    def initializePage(self):
        if self.context.dbu is not None:
            self.checkbox_calibration.setChecked(True)
            self.text_dbu.setText(str(self.context.dbu))
        else:
            self.checkbox_calibration.setChecked(False)
        self._calibration_toggled(self.checkbox_calibration.isChecked())

    def _calibration_toggled(self, checked: bool):
        self.text_dbu.setEnabled(checked)

    def validatePage(self) -> bool:
        if self.checkbox_calibration.isChecked():
            try:
                self.context.dbu = float(self.text_dbu.text())
            except ValueError:
                self.context.dbu = None
        else:
            self.context.dbu = None
        return True
