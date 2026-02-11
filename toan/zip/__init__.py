# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json

SAVE_README_TEXT = [
    "This zip file was created by Toan Machine to be used for training a neural network.",
    "If you just want to create a capture there isn't much to see here.",
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
    test_offset: int,
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
        "test_offset": test_offset,
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
