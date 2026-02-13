# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.record import RecordingContext
from toan.mix import concat_signals
from toan.signal import generate_capture_signal
from toan.soundio.record_wet import RecordWetController, RecordWetProgress

RECORD_TEXT = [
    "In this section you will send a signal through your pedal and record the result.",
    "Once you have started, allow the full recording to complete before proceeding. Do not change any settings on your pedal while recording.",
]


class RecordWetSignalPage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_record: QtWidgets.QPushButton
    bar_progress: QtWidgets.QProgressBar
    bar_update_timer: QtCore.QTimer

    record_controller: RecordWetController | None = None
    record_progress: RecordWetProgress | None = None

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.bar_update_timer = QtCore.QTimer()
        self.bar_update_timer.setInterval(50)
        self.bar_update_timer.setSingleShot(False)
        self.bar_update_timer.timeout.connect(self._update_status)

        self.setTitle("Record")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(RECORD_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.button_record = QtWidgets.QPushButton("Record", self)
        self.button_record.clicked.connect(self._clicked_record)
        layout.addWidget(self.button_record)

        label_progress = QtWidgets.QLabel("Progress:", self)
        layout.addWidget(label_progress)

        self.bar_progress = QtWidgets.QProgressBar(self)
        layout.addWidget(self.bar_progress)

    def cleanupPage(self):
        if self.record_controller is not None:
            self.record_controller.close()
            self.record_controller = None

    def isComplete(self):
        if self.context.signal_recorded is not None:
            self.cleanupPage()
            return True
        return False

    def _clicked_record(self):
        self.button_record.setEnabled(False)

        capture_signal_raw = generate_capture_signal(self.context.sample_rate)
        capture_signal_train = concat_signals(
            [capture_signal_raw, self.context.extra_signal_dry_train],
            self.context.sample_rate // 2,
        )

        if self.context.extra_signal_dry_test is not None:
            self.context.test_signal_offset = len(capture_signal_train)
            self.context.signal_dry = concat_signals(
                [capture_signal_train, self.context.extra_signal_dry_test],
                self.context.sample_rate // 2,
            )
        else:
            self.context.test_signal_offset = 0
            self.context.signal_dry = capture_signal_train

        self.record_controller = RecordWetController(
            self.context.sample_rate,
            self.context.signal_dry,
            self.context.input_channel,
            self.context.output_channel,
        )
        self.record_progress = self.record_controller.progress
        self.record_controller.start()

        self.bar_update_timer.start()

    def _update_status(self):
        self.bar_progress.setMaximum(len(self.context.signal_dry))
        if self.record_progress is not None:
            self.bar_progress.setValue(self.record_progress.samples_played)
            if self.record_progress.samples_recorded >= len(self.context.signal_dry):
                self._complete()
                self.bar_update_timer.stop()

    def _complete(self):
        assert self.record_controller is not None
        self.record_controller.close()
        self.context.signal_recorded = self.record_controller.get_recorded_signal()
        self.record_controller = None
        self.completeChanged.emit()
