# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.music import get_note_frequency_by_name
from toan.signal.generator.pluck import generate_pluck
from toan.signal.mix import concat_signals

# B C D E F D F
# E# C# E#
# E C E

# At A434 so no waveforms line up with training data
A4_FREQUENCY = 434


# Only the finest public domain tunes
def generate_mountain_king(
    sample_rate: int,
) -> np.ndarray:
    bpm: int = 138
    pause_samples = int(sample_rate / 32)
    quarter_note_samples: int = int(sample_rate / (bpm / 60.0)) - pause_samples
    eighth_note_samples: int = int(sample_rate / (bpm / 120.0)) - pause_samples

    freq_b = get_note_frequency_by_name("B", 2, A4_FREQUENCY)
    freq_c = get_note_frequency_by_name("C", 3, A4_FREQUENCY)
    freq_d = get_note_frequency_by_name("D", 3, A4_FREQUENCY)
    freq_e = get_note_frequency_by_name("E", 3, A4_FREQUENCY)
    freq_f = get_note_frequency_by_name("F", 3, A4_FREQUENCY)

    plucks: list[np.ndarray] = []
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_b))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_c))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_d))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_e))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_f))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_d))
    plucks.append(generate_pluck(sample_rate, eighth_note_samples, freq_f))

    return concat_signals(plucks, pause_samples)
