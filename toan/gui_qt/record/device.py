from PySide6 import QtWidgets

DEVICE_TEXT = [
    "Choose both the output device that will send a signal to your pedal as well as the input device that will record the signal coming back from your pedal."
]


class RecordDevicePage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Choose Device")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(DEVICE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        label_output = QtWidgets.QLabel("Output Device:", self)
        layout.addWidget(label_output)

        combo_output = QtWidgets.QComboBox(self)
        layout.addWidget(combo_output)

        label_input = QtWidgets.QLabel("Input Device:", self)
        layout.addWidget(label_input)

        combo_input = QtWidgets.QComboBox(self)
        layout.addWidget(combo_input)
