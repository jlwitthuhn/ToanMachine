from PySide6 import QtWidgets

INTRO_TEXT = [
    "Welcome to the recording wizard.",
    "This will guide you through a step-by-step process to capture a recording of your pedal or amp.",
    "To begin: Ensure you have pedal connected such that you can send a signal out of this computer, through your pedal, and back into this computer.",
]


class RecordIntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Introduction")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(INTRO_TEXT), self)
        label.setWordWrap(True)

        layout.addWidget(label)
