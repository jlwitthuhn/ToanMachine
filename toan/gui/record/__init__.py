# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.record.config import RecordConfigPage
from toan.gui.record.context import RecordingContext
from toan.gui.record.device import RecordDevicePage
from toan.gui.record.extra import RecordExtraPage
from toan.gui.record.input_gain import RecordInputGainPage
from toan.gui.record.intro import RecordIntroPage
from toan.gui.record.output_level import RecordOutputLevelPage
from toan.gui.record.save import RecordSavePage
from toan.gui.record.wet import RecordWetSignalPage


class RecordWizard(QtWidgets.QWizard):
    context: RecordingContext

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.context = RecordingContext()

        self.addPage(RecordIntroPage(self))
        self.addPage(RecordConfigPage(self, self.context))
        self.addPage(RecordExtraPage(self, self.context))
        self.addPage(RecordDevicePage(self, self.context))
        self.addPage(RecordOutputLevelPage(self, self.context))
        self.addPage(RecordInputGainPage(self, self.context))
        self.addPage(RecordWetSignalPage(self, self.context))
        self.addPage(RecordSavePage(self, self.context))

        self.setWindowTitle("Recording Wizard")
        self.setModal(True)
