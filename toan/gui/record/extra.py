# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import warnings

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
        def load_and_resample_wav(desc: UserWavDesc):
            with warnings.catch_warnings():
                this_sample_rate, this_signal = wavfile.read(desc.path)
            if len(this_signal.shape) == 2:
                this_signal = this_signal[:, 0]
            if this_sample_rate != self.context.sample_rate:
                this_sample_count = len(this_signal)
                desired_sample_count = int(
                    this_sample_count * (self.context.sample_rate / this_sample_rate)
                )
                this_signal = resample(this_signal, desired_sample_count)
            return this_signal

        train_wavs: list[UserWavDesc] = self.table.model().get_selected_train_wavs()
        train_ready_to_concat: list[np.ndarray] = []
        for this_wav in train_wavs:
            this_signal = load_and_resample_wav(this_wav)
            train_ready_to_concat.append(this_signal)

        if len(train_ready_to_concat) > 0:
            train_signal = concat_signals(
                train_ready_to_concat, self.context.sample_rate // 4
            )
            train_signal = train_signal.astype(np.float32) / np.abs(train_signal).max()
            self.context.extra_signal_dry_train = train_signal
        else:
            self.context.extra_signal_dry_train = np.zeros(1)

        test_wavs: list[UserWavDesc] = self.table.model().get_selected_test_wavs()
        test_ready_to_concat: list[np.ndarray] = []
        for this_wav in test_wavs:
            this_signal = load_and_resample_wav(this_wav)
            test_ready_to_concat.append(this_signal)

        if len(test_ready_to_concat) > 0:
            test_signal = concat_signals(
                test_ready_to_concat, self.context.sample_rate // 4
            )
            # TODO: This scales the signal too loud in many cases, it should not do that
            test_signal = test_signal.astype(np.float32) / np.abs(test_signal).max()
            self.context.extra_signal_dry_test = test_signal

        return True
