# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from concurrent.futures import Future, ThreadPoolExecutor

from PySide6 import QtCore, QtWidgets

from toan.gui.record.context import RecordingContext
from toan.gui.record.voltage import TONE_FREQUENCY, dbu_to_millivolts_rms
from toan.soundio.input_calibration import calibrate_input_level
from toan.soundio.tone_loopback import LoopbackCancelled, ToneLoopback

INPUT_VOLTAGE_TEXT = [
    "Here you will find the voltage level that corresponds to 0 dBFS on your interface's input. Do not touch your interface's input or output gain.",
    "Unplug the cable going in to your pedal, the same one you measured the output voltage on, and plug it directly in to your interface's input. Then click 'Calibrate' to play a series of test tones through it.",
    "Plug your pedal back in once calibration succeeds.",
]


def _run_calibration(loopback: ToneLoopback, output_max_dbu: float) -> float:
    with loopback:
        return calibrate_input_level(loopback, output_max_dbu)


class RecordInputVoltagePage(QtWidgets.QWizardPage):
    context: RecordingContext

    button_calibrate: QtWidgets.QPushButton
    label_status: QtWidgets.QLabel
    poll_timer: QtCore.QTimer

    executor: ThreadPoolExecutor
    loopback: ToneLoopback | None = None
    calibration: Future | None = None
    input_max_dbu: float | None = None

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll_calibration)

        self.setTitle("Input Voltage")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(INPUT_VOLTAGE_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.button_calibrate = QtWidgets.QPushButton("Calibrate", self)
        self.button_calibrate.clicked.connect(self._clicked_calibrate)
        layout.addWidget(self.button_calibrate)

        self.label_status = QtWidgets.QLabel("", self)
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)

        layout.addStretch(1)

    def initializePage(self):
        # Revisiting input gain invalidates calibration.
        self.input_max_dbu = None
        self.context.output_level_dbu = None
        self.label_status.clear()
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self.input_max_dbu is not None

    def cleanupPage(self):
        self._cancel_calibration()
        self.input_max_dbu = None
        self.context.output_level_dbu = None
        self.completeChanged.emit()

    def shutdown(self):
        self._cancel_calibration()
        self.executor.shutdown(wait=True)

    def _cancel_calibration(self):
        if self.calibration is not None:
            self.loopback.cancel()
            self._finish_calibration()

    def validatePage(self) -> bool:
        if not self.isComplete():
            return False
        self.context.output_level_dbu = self.input_max_dbu
        return True

    def _clicked_calibrate(self):
        self.input_max_dbu = None
        self.context.output_level_dbu = None
        self.completeChanged.emit()
        self.button_calibrate.setEnabled(False)
        self.label_status.setText("Calibrating...")

        self.loopback = ToneLoopback(
            self.context.sample_rate,
            TONE_FREQUENCY,
            self.context.input_channel,
            self.context.output_channel,
        )
        self.calibration = self.executor.submit(
            _run_calibration, self.loopback, self.context.input_level_dbu
        )
        self.poll_timer.start()

    def _poll_calibration(self):
        if self.calibration is not None and self.calibration.done():
            self._finish_calibration()

    def _finish_calibration(self):
        self.poll_timer.stop()
        try:
            input_max_dbu = self.calibration.result()
            millivolts = dbu_to_millivolts_rms(input_max_dbu)
            self.label_status.setText(
                f"Calibration succeeded: 0 dBFS on your input is {input_max_dbu:.2f} dBu ({millivolts:.4g} mV RMS)."
            )
            self.input_max_dbu = input_max_dbu
        except LoopbackCancelled:
            self.label_status.clear()
        except Exception as error:
            self.label_status.setText(f"Calibration failed: {error}")
        self.loopback = None
        self.calibration = None
        self.button_calibrate.setEnabled(True)
        self.completeChanged.emit()
