# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
from enum import Enum

import numpy as np

from toan.signal.pluck import generate_pluck
from toan.signal.tone import generate_tone


def _generate_semitone_scale_frequencies(
    start_freq: float, additional_steps: int
) -> list[float]:
    assert additional_steps > 0
    result = [start_freq]
    for i in range(additional_steps):
        this_freq = start_freq * math.pow(2, (i + 1) / 12)
        result.append(this_freq)
    return result


class ScaleSound(Enum):
    TONE = 1
    PLUCK = 2


def generate_chromatic_scale(
    sample_rate: int,
    low_freq: float,
    steps: int,
    note_duration: float,
    sound_type: ScaleSound,
) -> list[np.ndarray]:
    freqs = _generate_semitone_scale_frequencies(low_freq, steps - 1)
    result = []
    for freq in freqs:
        match sound_type:
            case ScaleSound.TONE:
                this_tone = generate_tone(sample_rate, freq, note_duration, True)
                result.append(this_tone)
            case ScaleSound.PLUCK:
                this_pluck = generate_pluck(sample_rate, freq, note_duration)
                this_pluck = this_pluck / np.abs(this_pluck).max()
                result.append(this_pluck)
            case _:
                assert False
    return result
