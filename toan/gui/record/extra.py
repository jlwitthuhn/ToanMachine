# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
from PySide6 import QtWidgets
from scipy.io import wavfile
from scipy.signal import resample

from toan.gui.record import RecordingContext
from toan.mix import concat_signals
from toan.persistence import UserWavDesc, get_user_wav_list
from toan.qt import WavFileModel

EXTRA_AUDIO_TEXT = [
    "Select any extra audio you would like to include in the recorded signal.",
    "You can import audio to be used here with 'Sound Manager' from the main menu.",
]


class RecordExtraPage(QtWidgets.QWizardPage):
    context: RecordingContext

    table: QtWidgets.QTableView

    def __init__(self, parent, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Extra Audio")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(EXTRA_AUDIO_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.table = QtWidgets.QTableView(self)
        layout.addWidget(self.table)

    def initializePage(self):
        wav_files = get_user_wav_list()
        model = WavFileModel(self, wav_files, True)
        model._select_all()
        self.table.setModel(model)

    def validatePage(self):
        extra_wavs: list[UserWavDesc] = self.table.model().get_selected_wavs()
        ready_to_concat: list[np.ndarray] = []
        for this_wav in extra_wavs:
            this_sample_rate, this_signal = wavfile.read(this_wav.path)
            if len(this_signal.shape) == 2:
                this_signal = this_signal[:, 0]
            if this_sample_rate != self.context.sample_rate:
                this_sample_count = len(this_signal)
                desired_sample_count = int(
                    this_sample_count * (self.context.sample_rate / this_sample_rate)
                )
                this_signal = resample(this_signal, desired_sample_count)
            ready_to_concat.append(this_signal)

        if len(ready_to_concat) > 0:
            full_signal = concat_signals(ready_to_concat, self.context.sample_rate // 4)
            full_signal = full_signal.astype(np.float32) / np.abs(full_signal).max()
            self.context.extra_signal_dry = full_signal
        else:
            self.context.extra_signal_dry = np.zeros(1)
        return True
