# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.record.config import RecordConfigPage
from toan.gui.record.context import RecordingContext
from toan.gui.record.device import RecordDevicePage
from toan.gui.record.extra import RecordExtraPage
from toan.gui.record.input_gain import RecordInputGainPage
from toan.gui.record.input_voltage import RecordInputVoltagePage
from toan.gui.record.intro import RecordIntroPage
from toan.gui.record.output_calibration import RecordOutputCalibrationChoicePage
from toan.gui.record.output_level import RecordOutputLevelPage
from toan.gui.record.output_voltage import RecordOutputVoltagePage
from toan.gui.record.save import RecordSavePage
from toan.gui.record.wet import RecordWetSignalPage


class RecordWizard(QtWidgets.QWizard):
    context: RecordingContext

    page_output_calibration_choice: RecordOutputCalibrationChoicePage
    id_output_calibration_choice: int
    id_output_level: int
    id_output_voltage: int
    id_input_gain: int
    id_input_voltage: int
    id_wet_signal: int

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.context = RecordingContext()

        self.page_output_calibration_choice = RecordOutputCalibrationChoicePage(
            self, self.context
        )

        self.addPage(RecordIntroPage(self))
        self.addPage(RecordConfigPage(self, self.context))
        self.addPage(RecordExtraPage(self, self.context))
        self.addPage(RecordDevicePage(self, self.context))
        self.id_output_calibration_choice = self.addPage(
            self.page_output_calibration_choice
        )
        self.id_output_level = self.addPage(RecordOutputLevelPage(self, self.context))
        self.id_output_voltage = self.addPage(
            RecordOutputVoltagePage(self, self.context)
        )
        self.id_input_gain = self.addPage(RecordInputGainPage(self, self.context))
        self.id_input_voltage = self.addPage(RecordInputVoltagePage(self, self.context))
        self.id_wet_signal = self.addPage(RecordWetSignalPage(self, self.context))
        self.addPage(RecordSavePage(self, self.context))

        self.setWindowTitle("Recording Wizard")
        self.setModal(True)

    def nextId(self) -> int:
        # Output calibration branches to exactly one of the level or voltage
        # pages, and both rejoin the main sequence at input gain. Input
        # voltage is only measured when calibrating by voltage, and must come
        # after input gain because changing the gain invalidates it.
        current_id = self.currentId()
        if current_id == self.id_output_calibration_choice:
            if self.page_output_calibration_choice.is_voltage_selected():
                return self.id_output_voltage
            return self.id_output_level
        if current_id in (self.id_output_level, self.id_output_voltage):
            return self.id_input_gain
        if current_id == self.id_input_gain:
            if self.page_output_calibration_choice.is_voltage_selected():
                return self.id_input_voltage
            return self.id_wet_signal
        return super().nextId()
