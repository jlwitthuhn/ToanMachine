# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtGui, QtWidgets

from toan.formatting import format_seconds_as_mmss
from toan.gui.record import RecordingContext
from toan.signal.capture_signal import generate_capture_signal
from toan.signal.mix import concat_signals
from toan.soundio.record_wet import RecordWetController, RecordWetProgress

RECORD_TEXT = [
    "Configuration complete. Click 'Record' below to begin recording.",
    "Do not change any settings on your pedal or interface while recording.",
]


class RecordWetSignalPage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_record: QtWidgets.QPushButton
    bar_progress: QtWidgets.QProgressBar
    label_time: QtWidgets.QLabel
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

        progress_layout = QtWidgets.QHBoxLayout()
        self.bar_progress = QtWidgets.QProgressBar(self)
        progress_layout.addWidget(self.bar_progress, 1)

        self.label_time = QtWidgets.QLabel("00:00 / 00:00", self)
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.label_time.setFont(font)
        self.label_time.setToolTip("Time elapsed / Total duration")
        progress_layout.addWidget(self.label_time)
        layout.addLayout(progress_layout)

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

        capture_signal_details = generate_capture_signal(self.context.sample_rate)

        capture_signal_train = concat_signals(
            [capture_signal_details.signal, self.context.extra_signal_dry_train],
            self.context.sample_rate // 2,
        )

        train_begin, _ = capture_signal_details.segments["train"]
        if self.context.extra_signal_dry_test is not None:
            self.context.signal_dry = concat_signals(
                [capture_signal_train, self.context.extra_signal_dry_test],
                self.context.sample_rate // 2,
            )
            segment_test = (len(capture_signal_train), len(self.context.signal_dry))
        else:
            self.context.signal_dry = capture_signal_train
            segment_test = (0, 0)

        self.context.segments_dry = {
            "clicks": capture_signal_details.segments["clicks"],
            "train": (train_begin, len(capture_signal_train)),
            "test": segment_test,
            "sweep": capture_signal_details.segments["sweep"],
            "white_noise": capture_signal_details.segments["white_noise"],
        }

        self.record_controller = RecordWetController(
            self.context.sample_rate,
            self.context.signal_dry,
            self.context.input_channel,
            self.context.output_channel,
        )
        self.record_progress = self.record_controller.progress
        self._update_status()
        self.record_controller.start()

        self.bar_update_timer.start()

    def _update_status(self):
        self.bar_progress.setMaximum(len(self.context.signal_dry))
        if self.record_progress is not None:
            samples_played = min(
                self.record_progress.samples_played, len(self.context.signal_dry)
            )
            self.bar_progress.setValue(samples_played)
            elapsed = samples_played / self.context.sample_rate
            total = len(self.context.signal_dry) / self.context.sample_rate
            self.label_time.setText(
                f"{format_seconds_as_mmss(elapsed)} / {format_seconds_as_mmss(total)}"
            )
            if self.record_progress.samples_recorded >= len(self.context.signal_dry):
                self._complete()
                self.bar_update_timer.stop()

    def _complete(self):
        assert self.record_controller is not None
        self.record_controller.close()
        self.context.signal_recorded = self.record_controller.get_recorded_signal()
        self.record_controller = None
        self.completeChanged.emit()
