# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import soundfile as sf
from scipy.signal import resample


def load_and_resample_wav(sample_rate: int, path: str) -> np.ndarray:
    this_signal, this_sample_rate = sf.read(path, dtype="float32")
    if len(this_signal.shape) == 2:
        this_signal = this_signal[:, 0]
    if this_sample_rate != sample_rate:
        this_sample_count = len(this_signal)
        desired_sample_count = int(this_sample_count * (sample_rate / this_sample_rate))
        this_signal = resample(this_signal, desired_sample_count)
    assert type(this_signal) == np.ndarray
    return this_signal.astype(np.float32)
