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
from toan.wav import load_and_resample_wav

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

        panel_button = QtWidgets.QWidget(self)
        panel_button_layout = QtWidgets.QHBoxLayout(panel_button)
        panel_button_layout.setContentsMargins(0, 0, 0, 0)

        def pressed_train_all():
            model = self.table.model()
            if isinstance(model, WavFileModel):
                model.select_all_train()

        button_train_all = QtWidgets.QPushButton("Train with all")
        button_train_all.clicked.connect(pressed_train_all)
        panel_button_layout.addWidget(button_train_all)

        def pressed_test_all():
            model = self.table.model()
            if isinstance(model, WavFileModel):
                model.select_all_test()

        button_test_all = QtWidgets.QPushButton("Test with all")
        button_test_all.clicked.connect(pressed_test_all)
        panel_button_layout.addWidget(button_test_all)

        def pressed_deselect():
            model = self.table.model()
            if isinstance(model, WavFileModel):
                model.deselect_all()

        button_deselect_all = QtWidgets.QPushButton("Deselect all")
        button_deselect_all.clicked.connect(pressed_deselect)
        panel_button_layout.addWidget(button_deselect_all)

        layout.addWidget(panel_button)

    def initializePage(self):
        wav_files = get_user_wav_list()
        model = WavFileModel(self, wav_files, True)
        model.select_all_train()
        self.table.setModel(model)
        self.table.resizeColumnsToContents()

    def validatePage(self):
        train_wavs: list[UserWavDesc] = self.table.model().get_selected_train_wavs()
        train_ready_to_concat: list[np.ndarray] = []
        for this_wav in train_wavs:
            this_signal = load_and_resample_wav(self.context.sample_rate, this_wav.path)
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
            this_signal = load_and_resample_wav(self.context.sample_rate, this_wav.path)
            test_ready_to_concat.append(this_signal)

        if len(test_ready_to_concat) > 0:
            test_signal = concat_signals(
                test_ready_to_concat, self.context.sample_rate // 4
            )
            # TODO: This scales the signal too loud in many cases, it should not do that
            test_signal = test_signal.astype(np.float32) / np.abs(test_signal).max()
            self.context.extra_signal_dry_test = test_signal

        return True
