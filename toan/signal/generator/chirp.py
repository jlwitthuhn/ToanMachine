# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.signal.generator.gaussian import generate_gaussian_pulse


def generate_chirp(
    sample_rate: int,
    begin_freq: float,
    end_freq: float,
    duration: float,
    blend_ending: int = None,
) -> np.ndarray:
    if blend_ending is None:
        blend_ending = sample_rate // 1000

    sample_duration = int(sample_rate * duration)
    assert blend_ending <= sample_duration
    f = np.logspace(np.log10(begin_freq), np.log10(end_freq), sample_duration, False)
    f_rad = f * 2 * np.pi
    phase = np.cumsum(f_rad) / sample_rate
    result = np.sin(phase)
    if blend_ending > 0:
        end_mult = generate_gaussian_pulse(blend_ending * 2, 0)[-blend_ending:]
        result[-blend_ending:] = end_mult * result[-blend_ending:]
    return result
