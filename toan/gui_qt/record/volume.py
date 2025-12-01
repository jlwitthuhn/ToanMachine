import math

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets

from toan.gui_qt.record import RecordingContext

VOLUME_TEXT = [
    "Set the input gain on your interface so that you are able to record to full range of your pedal's output without clipping.",
    "With the audio signal running through your pedal, press 'Play Test Tone' below and adjust your input gain so that no clipping occurs.",
]

BAR_PRECISION = 1000
VOLUME_SAMPLE_COUNT = 3


class RecordVolumePage(QtWidgets.QWizardPage):
    context: RecordingContext

    play_button: QtWidgets.QPushButton
    play_active: bool = False

    bar_input_level: QtWidgets.QProgressBar
    bar_progress: int = 0
    bar_update_timer: QtCore.QTimer

    input_stream: sd.InputStream | None = None
    volume_samples: np.ndarray
    volume_samples_index: int = 0

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context
        self.volume_samples = np.zeros(VOLUME_SAMPLE_COUNT)

        self.bar_update_timer = QtCore.QTimer()
        self.bar_update_timer.setInterval(50)
        self.bar_update_timer.setSingleShot(False)
        self.bar_update_timer.timeout.connect(self._update_bar)

        self.setTitle("Set Volume")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(VOLUME_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        self.play_button = QtWidgets.QPushButton("Play Test Tone", self)
        self.play_button.clicked.connect(self._clicked_play_test)
        layout.addWidget(self.play_button)

        label_input_level = QtWidgets.QLabel("Input Level:", self)
        layout.addWidget(label_input_level)

        self.bar_input_level = QtWidgets.QProgressBar(self)
        self.bar_input_level.setMinimum(0)
        self.bar_input_level.setMaximum(BAR_PRECISION)
        layout.addWidget(self.bar_input_level)

    def cleanupPage(self, /) -> None:
        if self.play_active:
            self._clicked_play_test()
        assert self.play_active == False

    def validatePage(self, /) -> bool:
        self.cleanupPage()
        return True

    def _clicked_play_test(self):
        if self.play_active:
            self.play_active = False
            self.play_button.setText("Play Test Tone")
            self.bar_update_timer.stop()
            if self.input_stream is not None:
                self.input_stream.close()
                self.input_stream = None
            return
        self.play_active = True
        self.play_button.setText("Stop Test Tone")
        self.bar_update_timer.start()
        self._setup_io_streams()

    def _setup_io_streams(self):
        sample_rate = self.context.sample_rate
        input_channel = self.context.input_channel
        self.input_stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=2048,
            device=input_channel.device_index,
            callback=self._input_callback,
        )
        self.input_stream.start()

    def _update_bar(self):
        self.bar_input_level.setValue(self.bar_progress)

    def _input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ) -> None:
        sample = np.abs(indata).max()
        self.volume_samples[self.volume_samples_index] = sample
        self.volume_samples_index = (
            self.volume_samples_index + 1
        ) % VOLUME_SAMPLE_COUNT
        avg = self.volume_samples.mean()
        self.bar_progress = min(BAR_PRECISION, math.floor(avg * 1000))
