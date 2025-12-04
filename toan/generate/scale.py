# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from toan.generate.tone import generate_tone


def _generate_semitone_scale_frequencies(
    start_freq: float, additional_steps: int
) -> list[float]:
    assert additional_steps > 0
    result = [start_freq]
    for i in range(additional_steps):
        this_freq = start_freq * math.pow(2, (i + 1) / 12)
        result.append(this_freq)
    return result


def generate_chromatic_scale(
    sample_rate: int,
    low_freq: float,
    steps: int,
    amplitude: float,
    note_duration: float,
) -> list[np.ndarray]:
    freqs = _generate_semitone_scale_frequencies(low_freq, steps - 1)
    result = []
    for freq in freqs:
        this_tone = generate_tone(sample_rate, freq, amplitude, note_duration, True)
        result.append(this_tone)
    return result
