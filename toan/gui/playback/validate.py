# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

from PySide6 import QtGui, QtWidgets

from toan.gui.playback import PlaybackContext
from toan.model.metadata import ModelA1Metadata, ModelA2Metadata
from toan.model.nam_a1_wavenet_config import json_a1_wavenet_config
from toan.model.nam_a1_wavenet_torch import NamA1WaveNetTorch
from toan.model.nam_a2_wavenet_config import json_a2_wavenet_container_config
from toan.model.nam_a2_wavenet_torch import NamA2WaveNetTorch


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

        if "version" not in nam_json or not isinstance(nam_json["version"], str):
            self.text_edit.append("Error: Root key 'version' must be str")
            return

        if "sample_rate" not in nam_json or not isinstance(
            nam_json["sample_rate"], (int, float)
        ):
            self.text_edit.append("Error: Root key 'sample_rate' must be a number")
            return
        # Always store the sample rate as an int internally, truncate float on load
        self.context.sample_rate = int(nam_json["sample_rate"])

        json_version = nam_json["version"]
        if json_version == "0.5.4":
            self._init_a1(nam_json)
        elif json_version == "0.7.0":
            self._init_a2(nam_json)
        else:
            self.text_edit.append(f"Error: Unsupported version: {json_version}")

    def isComplete(self):
        return self.context.nam_model is not None

    def _init_a1(self, nam_json: dict):
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
            model_config = json_a1_wavenet_config(nam_json["config"])
        except:
            self.text_edit.append("Error: a1 wavenet config is not valid")
            return

        self.text_edit.append("Creating model...")

        metadata = ModelA1Metadata(
            "Playback A1 NAM model", "Toan Machine", "Test model"
        )

        model = NamA1WaveNetTorch(model_config, metadata, self.context.sample_rate)
        self.text_edit.append(f"Model parameters: {model.parameter_count}")

        if "weights" not in nam_json or not isinstance(nam_json["weights"], list):
            self.text_edit.append("Error: Model weights are not specified")
            return

        weights = nam_json["weights"]
        weight_count = len(weights)
        self.text_edit.append(f"Profile parameters: {weight_count}")

        model.import_nam_linear_weights(weights)
        self.text_edit.append("A1 model successfully loaded")

        self.context.nam_model = model
        self.completeChanged.emit()

    def _init_a2(self, nam_json: dict):
        if "architecture" not in nam_json:
            self.text_edit.append("Error: Architecture is not specified")
            return

        arch = nam_json["architecture"]
        if not isinstance(arch, str):
            self.text_edit.append("Error: Architecture is not a string")
            return
        if arch != "SlimmableContainer":
            self.text_edit.append(f"Error: Unsupported architecture: {arch}")
            return

        if "config" not in nam_json:
            self.text_edit.append("Error: Model config is not specified")
            return

        self.text_edit.append("Loading config...")

        try:
            model_config = json_a2_wavenet_container_config(nam_json["config"])
        except:
            self.text_edit.append("Error: a2 wavenet config is not valid")
            return

        self.text_edit.append("Creating model...")

        metadata = ModelA2Metadata(
            "Playback A2 NAM model", "Toan Machine", "Test model"
        )

        model = NamA2WaveNetTorch(model_config, metadata, self.context.sample_rate)
        self.text_edit.append(f"Model parameters: {model.parameter_count}")
        self.text_edit.append(f"Submodels: {len(model.submodels)}")

        submodel_weights: list[list[float]] = []
        for submodel in nam_json["config"]["submodels"]:
            if (
                not isinstance(submodel, dict)
                or "model" not in submodel
                or not isinstance(submodel["model"], dict)
                or "weights" not in submodel["model"]
                or not isinstance(submodel["model"]["weights"], list)
            ):
                self.text_edit.append("Error: Submodel weights are not specified")
                return
            submodel_weights.append(submodel["model"]["weights"])

        weight_count = sum(len(weights) for weights in submodel_weights)
        self.text_edit.append(f"Profile parameters: {weight_count}")

        model.import_nam_linear_weights(submodel_weights)
        self.text_edit.append("A2 model successfully loaded")

        self.context.nam_model = model
        self.completeChanged.emit()
