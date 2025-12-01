from PySide6 import QtWidgets

from toan.gui_qt.record.device import RecordDevicePage
from toan.gui_qt.record.intro import RecordIntroPage


class RecordWizard(QtWidgets.QWizard):
    def __init__(self, parent):
        super().__init__(parent)

        self.addPage(RecordIntroPage(self))
        self.addPage(RecordDevicePage(self))

        self.setWindowTitle("Recording Wizard")
        self.setModal(True)
