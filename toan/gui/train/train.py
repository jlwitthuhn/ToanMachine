# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.model.nam_wavenet import NamWaveNet
from toan.model.nam_wavenet_config import default_wavenet_config

TRAIN_TEXT = [
    "Your model is now training. After training has finished you will be asked to choose a location for the NAM file."
]


class TrainTrainPage(QtWidgets.QWizardPage):
    def __init__(self, parent):
        super().__init__(parent)

        self.setTitle("Train")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(TRAIN_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        label_progress = QtWidgets.QLabel("Progress:", self)
        layout.addWidget(label_progress)

        self.bar_progress = QtWidgets.QProgressBar(self)
        layout.addWidget(self.bar_progress)

        _run_training()


def _run_training():
    model_config = default_wavenet_config()
    model = NamWaveNet(model_config)
    assert model is not None
    print(f"input width: {model.receptive_field}")
