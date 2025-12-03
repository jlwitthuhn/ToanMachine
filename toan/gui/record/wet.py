# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record import RecordingContext

RECORD_TEXT = [
    "In this section you will send a signal through your pedal and record the result.",
    "Once you have started, allow the full recording to complete before proceeding. Do not change any settings on your pedal while recording.",
]


class RecordWetSignalPage(QtWidgets.QWizardPage):
    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)

        self.setTitle("Record")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(RECORD_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        button_record = QtWidgets.QPushButton("Record", self)
        button_record.clicked.connect(self._clicked_record)
        layout.addWidget(button_record)

        label_progress = QtWidgets.QLabel("Progress:", self)
        layout.addWidget(label_progress)

        progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(progress_bar)

    def isComplete(self, /):
        return False

    def _clicked_record(self):
        print("Clicked record")
