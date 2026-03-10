# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def effect_vibrato(
    signal: np.ndarray, sample_rate: int, vibrato_frequency: float, depth: float
):
    t = np.arange(len(signal)) / sample_rate
    delay_samples_max = depth * sample_rate
    delay_samples = delay_samples_max * np.sin(2.0 * np.pi * vibrato_frequency * t)
    indices_float = np.arange(len(signal)) - delay_samples
    indices_float = np.clip(indices_float, 0, len(signal) - 2)
    indices_pre = np.floor(indices_float).astype(int)
    indices_post = indices_pre + 1
    indices_frac = indices_float - indices_pre
    signal[:] = (
        indices_frac * signal[indices_post] + (1 - indices_frac) * signal[indices_pre]
    )
