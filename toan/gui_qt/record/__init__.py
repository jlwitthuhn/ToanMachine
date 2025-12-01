from PySide6 import QtWidgets

from toan.gui_qt.record.context import RecordingContext
from toan.gui_qt.record.device import RecordDevicePage
from toan.gui_qt.record.intro import RecordIntroPage


class RecordWizard(QtWidgets.QWizard):
    context: RecordingContext

    def __init__(self, parent):
        super().__init__(parent)

        self.addPage(RecordIntroPage(self))
        self.addPage(RecordDevicePage(self, context))

        self.setWindowTitle("Recording Wizard")
        self.setModal(True)
