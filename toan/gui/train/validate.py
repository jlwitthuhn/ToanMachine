# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import io
import json
import threading
import zipfile

import numpy as np
from PySide6 import QtGui, QtWidgets
from scipy.io import wavfile

from toan.gui.train import TrainingContext


class _ValidateThreadContext:
    text_edit: QtWidgets.QTextEdit


class TrainValidatePage(QtWidgets.QWizardPage):
    context: TrainingContext
    thread_context: _ValidateThreadContext | None = None
    text_edit: QtWidgets.QTextEdit

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

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
        self.text_edit.append(f"Analyzing input file: {self.context.input_path}")
        self.thread_context = _ValidateThreadContext()
        self.thread_context.text_edit = self.text_edit

        def thread_func():
            _run_thread(self.thread_context, self.context.input_path)

        threading.Thread(target=thread_func).start()


def _find_clicks(signal: np.ndarray, raw_noise_floor: float) -> list[int]:
    noise_threshold = raw_noise_floor * 3.5
    print("noise_threshold", noise_threshold)
    click_indices = []
    silence_samples_remaining = 0
    for i in range(len(signal)):
        this_sample = signal[i]
        if np.abs(this_sample) > noise_threshold:
            if silence_samples_remaining <= 0:
                click_indices.append(i)
            silence_samples_remaining = 100
        else:
            silence_samples_remaining -= 1
    return click_indices


def _run_thread(context: _ValidateThreadContext, input_path: str):
    context.text_edit.append("Loading as zip archive...")
    try:
        with zipfile.ZipFile(input_path, "r") as zip_file:
            context.text_edit.append("Loading config...")

            try:
                config_json_bytes = zip_file.read("config.json")
                config_json = json.loads(config_json_bytes.decode("utf-8"))
            except KeyError:
                context.text_edit.append("Error: config.json not found")
                return

            if "version" not in config_json:
                context.text_edit.append(
                    "Error: config.json does not contain key 'version'"
                )
                return
            if (
                not isinstance(config_json["version"], int)
                or config_json["version"] != 0
            ):
                context.text_edit.append("Error: config.json has unknown version")
                return

            if "device_name" not in config_json or not isinstance(
                config_json["device_name"], str
            ):
                context.text_edit.append(
                    "Error: config.json does not contain key 'device_name'"
                )
                return

            context.text_edit.append(f"Device name: {config_json["device_name"]}")

            if "sample_rate" not in config_json or not isinstance(
                config_json["sample_rate"], int
            ):
                context.text_edit.append(
                    "Error: config.json does not contain key 'sample_rate'"
                )
                return

            context.text_edit.append(f"Sample rate: {config_json["sample_rate"]}")

            if "dry_signal" not in config_json or not isinstance(
                config_json["dry_signal"], str
            ):
                context.text_edit.append(
                    "Error: config.json does not contain key 'dry_signal'"
                )
                return
            if "wet_signal" not in config_json or not isinstance(
                config_json["wet_signal"], str
            ):
                context.text_edit.append(
                    "Error: config.json does not contain key 'wet_signal'"
                )
                return

            try:
                context.text_edit.append(
                    f"Loading dry signal: {config_json["dry_signal"]}"
                )
                dry_signal_bytes = zip_file.read(config_json["dry_signal"])
                with io.BytesIO(dry_signal_bytes) as dry_bytes_io:
                    dry_sample_rate, dry_signal = wavfile.read(dry_bytes_io)
                    if dry_sample_rate != config_json["sample_rate"]:
                        context.text_edit.append(
                            "Error: dry signal has unexpected sample rate"
                        )
                        return
            except:
                context.text_edit.append(
                    f"Error: Failed to load dry signal from {config_json["dry_signal"]}"
                )
                return

            try:
                context.text_edit.append(
                    f"Loading wet signal: {config_json["wet_signal"]}"
                )
                wet_signal_bytes = zip_file.read(config_json["wet_signal"])
                with io.BytesIO(wet_signal_bytes) as wet_bytes_io:
                    wet_sample_rate, wet_signal = wavfile.read(wet_bytes_io)
                    if wet_sample_rate != config_json["sample_rate"]:
                        context.text_edit.append(
                            "Error: wet signal has unexpected sample rate"
                        )
                        return
            except:
                context.text_edit.append(
                    f"Error: Failed to load wet signal from {config_json["wet_signal"]}"
                )
                return

            # Input file is half a second of silence, a click, quarter second silence, click, half second silence
            dry_silent = dry_signal[dry_sample_rate // 8 : 3 * dry_sample_rate // 8]
            dry_clicks = dry_signal[3 * dry_sample_rate // 8 : 5 * dry_sample_rate // 4]
            dry_noise_floor = np.max(np.abs(dry_silent))
            context.text_edit.append(f"Dry noise floor: {dry_noise_floor}")

            wet_silent = wet_signal[wet_sample_rate // 8 : 3 * wet_sample_rate // 8]
            wet_clicks = wet_signal[3 * wet_sample_rate // 8 : 5 * wet_sample_rate // 4]
            wet_noise_floor = np.max(np.abs(wet_silent))
            context.text_edit.append(f"Wet noise floor: {wet_noise_floor}")

            dry_click_indices = _find_clicks(dry_clicks, wet_noise_floor)
            if len(dry_click_indices) != 2:
                context.text_edit.append(
                    f"Error: Failed to locate clicks in dry signal"
                )
                return

            wet_click_indices = _find_clicks(wet_clicks, wet_noise_floor)
            if len(wet_click_indices) != 2:
                context.text_edit.append(
                    f"Error: Failed to locate clicks in wet signal"
                )
                return

            # Click delta is the time between two clicks on a single track
            # Click delta delta is the difference between the 'Click deltas' of the wet and dry tracks
            dry_click_delta = dry_click_indices[1] - dry_click_indices[0]
            wet_click_delta = wet_click_indices[1] - wet_click_indices[0]
            click_delta_delta = abs(dry_click_delta - wet_click_delta)
            context.text_edit.append(f"Click delta delta: {click_delta_delta}")

            if click_delta_delta > 4:
                context.text_edit.append("Error: Click delta delta is greater than 4")
                return

            latency_samples_a = wet_click_indices[0] - dry_click_indices[0]
            latency_samples_b = wet_click_indices[1] - dry_click_indices[1]
            latency_samples = min(latency_samples_a, latency_samples_b)
            context.text_edit.append(f"Recording latency: {latency_samples} samples")

    except zipfile.BadZipFile:
        context.text_edit.append("Error: File is not a valid zip archive")
        return
    except:
        context.text_edit.append("Error: Unknown error occurred")
        return
