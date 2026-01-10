# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

from PySide6 import QtGui, QtWidgets

from toan.gui.playback import PlaybackContext
from toan.model.metadata import ModelMetadata
from toan.model.nam_wavenet import NamWaveNet
from toan.model.nam_wavenet_config import json_wavenet_config


class PlaybackValidatePage(QtWidgets.QWizardPage):
    context: PlaybackContext

    def __init__(self, parent, context: PlaybackContext):
        super().__init__(parent)
        self.context = context

        self.setCommitPage(True)
        self.setTitle("Checking Input")
        layout = QtWidgets.QVBoxLayout(self)

        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setFont(font)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def initializePage(self):
        self.text_edit.clear()
        self.text_edit.append(f"Analyzing NAM file: {self.context.nam_model_path}")

        try:
            with open(self.context.nam_model_path, "r") as file:
                nam_json = json.load(file)
                self.text_edit.append("Parsed file")
        except FileNotFoundError:
            self.text_edit.append("Error: Failed to open file")
            return
        except json.decoder.JSONDecodeError:
            self.text_edit.append("Error: Failed to decode json")
            return

        if not isinstance(nam_json, dict):
            self.text_edit.append("Error: Root json object must be dict")
            return

        if "sample_rate" not in nam_json or not isinstance(
            nam_json["sample_rate"], int
        ):
            self.text_edit.append("Error: Root key 'sample_rate' must be int")
            return
        self.context.sample_rate = nam_json["sample_rate"]

        if "architecture" not in nam_json:
            self.text_edit.append("Error: Architecture is not specified")
            return

        arch = nam_json["architecture"]
        if not isinstance(arch, str):
            self.text_edit.append("Error: Architecture is not a string")
            return
        if arch != "WaveNet":
            self.text_edit.append(f"Error: Unsupported architecture: {arch}")
            return

        if "config" not in nam_json:
            self.text_edit.append("Error: Model config is not specified")
            return

        self.text_edit.append("Loading config...")

        try:
            model_config = json_wavenet_config(nam_json["config"])
        except:
            self.text_edit.append("Error: wavenet config is not valid")
            return

        self.text_edit.append("Creating model...")

        metadata = ModelMetadata(
            "Playback NAM model", "Playback make", "Playback model"
        )

        model = NamWaveNet(model_config, metadata, self.context.sample_rate)
        model_params = model.parameter_count
        self.text_edit.append(f"Model parameters: {model_params}")

        if "weights" not in nam_json or not isinstance(nam_json["weights"], list):
            self.text_edit.append("Error: Model weights are not specified")
            return

        weights = nam_json["weights"]
        weight_count = len(weights)
        self.text_edit.append(f"Profile parameters: {weight_count}")

        model.import_nam_linear_weights(weights)
        self.text_edit.append("Model successfully loaded")

        self.context.nam_model = model
        self.completeChanged.emit()

    def isComplete(self):
        return self.context.nam_model is not None
