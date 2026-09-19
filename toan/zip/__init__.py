# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

SAVE_README_TEXT = [
    "This zip file was created by Toan Machine to be used for training a NAM capture.",
    "If you just want to create a capture there isn't much to see here.",
    "Contents:\n"
    + "- config.json: metadata about the device and recording\n"
    + "- dry.wav: dry signal\n"
    + "- wet.wav: wet signal",
    "https://github.com/jlwitthuhn/ToanMachine",
]

import io
import zipfile

import numpy as np
import scipy


def create_training_zip(
    sample_rate: int,
    signal_dry: np.ndarray,
    signal_wet: np.ndarray,
    dev_make: str,
    dev_model: str,
    segment_clicks: tuple[int, int],
    segment_train: tuple[int, int],
    segment_test: tuple[int, int],
    segment_sweep: tuple[int, int],
    dbu: float | None = None,
) -> io.BytesIO:
    wav_dry = io.BytesIO()
    scipy.io.wavfile.write(
        wav_dry,
        sample_rate,
        signal_dry.astype(np.float32),
    )
    wav_dry.seek(0)

    wav_wet = io.BytesIO()
    scipy.io.wavfile.write(
        wav_wet,
        sample_rate,
        signal_wet.astype(np.float32),
    )
    wav_wet.seek(0)

    metadata = {
        "version": 0,
        "device_make": dev_make,
        "device_model": dev_model,
        "sample_rate": sample_rate,
        "segments": {
            "clicks": [segment_clicks[0], segment_clicks[1]],
            "train": [segment_train[0], segment_train[1]],
            "test": [segment_test[0], segment_test[1]],
            "sweep": [segment_sweep[0], segment_sweep[1]],
        },
        "input_level_dbu": dbu,
        "dry_signal": "dry.wav",
        "wet_signal": "wet.wav",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip:
        zip.writestr("readme.txt", "\n\n".join(SAVE_README_TEXT))
        zip.writestr("config.json", json.dumps(metadata, indent=4))
        zip.writestr("dry.wav", wav_dry.getvalue())
        zip.writestr("wet.wav", wav_wet.getvalue())

    return zip_buffer
