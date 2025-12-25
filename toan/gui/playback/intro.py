# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

PLAYBACK_TEXT = [
    "Welcome to the playback wizard.",
    "This allows you to load a NAM model, record some input audio, then hear how that model sounds.",
    "This is intended as a simple and quick way to test that your model works and does not support real-time playback. For more extensive testing you should load the model into a DAW.",
]


class PlaybackIntroPage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Playback")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(PLAYBACK_TEXT), self)
        label.setWordWrap(True)

        layout.addWidget(label)
