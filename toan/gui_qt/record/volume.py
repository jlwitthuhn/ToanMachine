from PySide6 import QtWidgets

from toan.gui_qt.record import RecordingContext

VOLUME_TEXT = [
    "Set the input gain on your interface so that you are able to record to full range of your pedal's output without clipping.",
    "With the audio signal running through your pedal, press 'Play Test Tone' below and adjust your input gain so that no clipping occurs.",
]


class RecordVolumePage(QtWidgets.QWizardPage):
    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Set Volume")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(VOLUME_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        button_test = QtWidgets.QPushButton("Play Test Tone", self)
        layout.addWidget(button_test)

        label_input_level = QtWidgets.QLabel("Input Level:", self)
        layout.addWidget(label_input_level)

        bar_input_level = QtWidgets.QProgressBar(self)
        bar_input_level.setMinimum(0)
        bar_input_level.setMaximum(100)
        layout.addWidget(bar_input_level)
