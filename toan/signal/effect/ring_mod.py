# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
from scipy.signal import resample


def effect_ring_mod(
    signal: np.ndarray, sample_rate: int, carrier_frequency: float
) -> np.ndarray:
    # Make it wide and resample back down to minimize aliasing
    SAMPLE_MULTIPLIER = 2
    signal_wide = resample(signal, len(signal) * SAMPLE_MULTIPLIER)
    t = np.arange(len(signal_wide)) / (sample_rate * SAMPLE_MULTIPLIER)
    carrier = np.sin(2.0 * np.pi * carrier_frequency * t)
    result_wide = signal_wide * carrier
    return resample(result_wide, len(signal))
