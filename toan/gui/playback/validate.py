# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

from PySide6 import QtGui, QtWidgets

from toan.gui.playback import PlaybackContext
from toan.model.metadata import ModelMetadata
from toan.model.nam_a1_wavenet import NamA1WaveNet
from toan.model.nam_a1_wavenet_config import json_a1_wavenet_config
from toan.model.nam_a2_wavenet import NamA2WaveNet
from toan.model.nam_a2_wavenet_config import json_a2_wavenet_config


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
            nam_json["sample_rate"], int
        ):
            self.text_edit.append("Error: Root key 'sample_rate' must be int")
            return
        self.context.sample_rate = nam_json["sample_rate"]

        json_version = nam_json["version"]
        if json_version == "0.5.4":
            self._init_a1(nam_json)
        elif json_version == "0.6.0":
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

        metadata = ModelMetadata("Playback A1 NAM model", "Toan Machine", "Test model")

        model = NamA1WaveNet(model_config, metadata, self.context.sample_rate)
        self.text_edit.append(f"Model parameters: {model.parameter_count}")

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

    def _init_a2(self, nam_json: dict):
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
            model_config = json_a2_wavenet_config(nam_json["config"])
        except TypeError as e:
            self.text_edit.append("Error: a2 wavenet config has invalid contents")
            self.text_edit.append(f"> {e.__str__()}")
            return
        except:
            self.text_edit.append(
                "Error: encountered unknown error while parsing a2 wavenet config"
            )
            return

        if "weights" not in nam_json or not isinstance(nam_json["weights"], list):
            self.text_edit.append("Error: model weights are not specified")
            return
        self.text_edit.append(f"Found {len(nam_json["weights"])} weights")

        for layer in model_config.layers:
            if layer.gating_mode not in ["none", "gated"]:
                self.text_edit.append(
                    f"Error: unsupported gating mode: {layer.gating_mode}"
                )
                return
            film_active = (
                layer.conv_pre_film.active
                or layer.conv_post_film
                or layer.input_mixin_pre_film.active
                or layer.input_mixin_post_film.active
                or layer.activation_pre_film.active
                or layer.activation_post_film.active
                or layer.layer1x1_post_film.active
                or layer.head1x1_post_film.active
            )
            if film_active:
                self.text_edit.append("Error: film is not supported")

        self.text_edit.append("Creating model...")

        metadata = ModelMetadata("Playback A2 NAM model", "Toan Machine", "Test model")
        model = NamA2WaveNet(model_config, metadata, self.context.sample_rate)
        self.text_edit.append(f"Model parameters: {model.parameter_count}")

        if model.parameter_count != len(nam_json["weights"]):
            self.text_edit.append(
                "Error: Mismatch between number of loaded weights and model parameters"
            )
            model.debug_print_size()
            return

        raise NotImplemented
