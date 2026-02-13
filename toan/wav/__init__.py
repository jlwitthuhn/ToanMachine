# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def load_and_resample_wav(sample_rate: int, path: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
        this_sample_rate, this_signal = wavfile.read(path)
    if len(this_signal.shape) == 2:
        this_signal = this_signal[:, 0]
    if this_sample_rate != sample_rate:
        this_sample_count = len(this_signal)
        desired_sample_count = int(this_sample_count * (sample_rate / this_sample_rate))
        this_signal = resample(this_signal, desired_sample_count)
    return this_signal
