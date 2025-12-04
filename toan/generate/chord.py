# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from toan.generate.chirp import generate_chirp
from toan.music import get_note_frequency_by_name, get_note_index_by_name


def _increase_semitones(frequency: float, semitones: int) -> float:
    return frequency * math.pow(2, semitones / 12)


def generate_major_chord_chirp(
    sample_rate: int,
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    begin_index = get_note_index_by_name(begin_note, begin_octave)
    end_index = get_note_index_by_name(end_note, end_octave)
    semitones_count = end_index - begin_index - 7

    root_begin: float = get_note_frequency_by_name(begin_note, begin_octave, 440)
    root_end: float = _increase_semitones(root_begin, semitones_count)
    root_chirp = generate_chirp(
        sample_rate, root_begin, root_end, amplitude / 3, duration
    )

    third_begin: float = _increase_semitones(root_begin, 4)
    third_end: float = _increase_semitones(third_begin, semitones_count)
    third_chirp = generate_chirp(
        sample_rate, third_begin, third_end, amplitude / 3, duration
    )

    fifth_begin: float = _increase_semitones(root_begin, 7)
    fifth_end: float = _increase_semitones(fifth_begin, semitones_count)
    fifth_chirp = generate_chirp(
        sample_rate, fifth_begin, fifth_end, amplitude / 3, duration
    )

    return root_chirp + third_chirp + fifth_chirp
