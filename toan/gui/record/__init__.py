from PySide6 import QtWidgets

from toan.gui.record.context import RecordingContext
from toan.gui.record.device import RecordDevicePage
from toan.gui.record.intro import RecordIntroPage
from toan.gui.record.volume import RecordVolumePage
from toan.gui.record.wet import RecordWetSignalPage


class RecordWizard(QtWidgets.QWizard):
    context: RecordingContext

    def __init__(self, parent):
        super().__init__(parent)
        self.context = RecordingContext()

        self.addPage(RecordIntroPage(self))
        self.addPage(RecordDevicePage(self, self.context))
        self.addPage(RecordVolumePage(self, self.context))
        self.addPage(RecordWetSignalPage(self, self.context))

        self.setWindowTitle("Recording Wizard")
        self.setModal(True)
