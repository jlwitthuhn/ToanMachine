# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

from PySide6 import QtGui, QtWidgets

from toan.gui.playback import PlaybackContext


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

        model_config = nam_json["config"]
        valid = _validate_wavenet_config(model_config)
        if not valid:
            self.text_edit.append("Error: wavenet config is not valid")
            return

        self.text_edit.append("Success")


def _validate_wavenet_config(config: dict) -> bool:
    if "layers" not in config or not isinstance(config["layers"], list):
        return False

    layers = config["layers"]
    for layer in layers:
        if not isinstance(layer, dict):
            return False
        if "input_size" not in layer or not isinstance(layer["input_size"], int):
            return False
        if "condition_size" not in layer or not isinstance(
            layer["condition_size"], int
        ):
            return False
        if "head_size" not in layer or not isinstance(layer["head_size"], int):
            return False
        if "channels" not in layer or not isinstance(layer["channels"], int):
            return False
        if "kernel_size" not in layer or not isinstance(layer["kernel_size"], int):
            return False
        if "dilations" not in layer or not isinstance(layer["dilations"], list):
            return False
        if "activation" not in layer or not isinstance(layer["activation"], str):
            return False
        if "gated" not in layer or not isinstance(layer["gated"], bool):
            return False
        if "head_bias" not in layer or not isinstance(layer["head_bias"], bool):
            return False

    return True
