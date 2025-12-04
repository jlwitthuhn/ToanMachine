# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.generate.chirp import generate_chirp
from toan.generate.chord import generate_major_chord_chirp
from toan.generate.scale import generate_chromatic_scale
from toan.mix import concat_signals
from toan.music import get_note_frequency_by_name, get_note_index_by_name


def generate_capture_signal(sample_rate: int, amplitude: float) -> np.ndarray:
    sweep_up = generate_chirp(sample_rate, 20.0, 20000.0, amplitude, 10.0)

    # Lowest bass guitar note is E1
    scale_lo = get_note_frequency_by_name("E", 1, 440)
    index_lo = get_note_index_by_name("E", 1)
    # Guitar high E string played at the 24th fret is E6, add one octave to be safe
    index_hi = get_note_index_by_name("E", 7)
    scale_steps = index_hi - index_lo
    scale_list = generate_chromatic_scale(
        sample_rate, scale_lo, scale_steps, amplitude, 0.25
    )
    scale = concat_signals(scale_list)

    # Sweeping major chord
    sweep_major_chord = generate_major_chord_chirp(
        sample_rate, "E", 1, "E", 7, amplitude, 6.0
    )

    return concat_signals([sweep_up, scale, sweep_major_chord], sample_rate // 2)
