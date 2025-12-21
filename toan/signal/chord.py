# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
from enum import Enum

import numpy as np

from toan.music import get_note_frequency_by_name, get_note_index_by_name
from toan.signal.chirp import generate_chirp


def _increase_semitones(frequency: float, semitones: int) -> float:
    return frequency * math.pow(2, semitones / 12)


class ChordType(Enum):
    Octave = 1
    Tritone = 2
    Major = 3
    Minor = 4
    Diminished = 5


def generate_generic_chord_chirp(
    sample_rate: int,
    shape: list[int],
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    begin_index = get_note_index_by_name(begin_note, begin_octave)
    end_index = get_note_index_by_name(end_note, end_octave)
    semitone_width = max(shape)
    semitone_count = end_index - begin_index - semitone_width
    assert 0 < semitone_width <= semitone_count

    note_count = len(shape) + 1

    root_begin: float = get_note_frequency_by_name(begin_note, begin_octave, 440.0)
    root_end: float = _increase_semitones(root_begin, semitone_count)
    result = generate_chirp(
        sample_rate, root_begin, root_end, amplitude / note_count, duration
    )

    for this_note_offset in shape:
        note_begin: float = _increase_semitones(root_begin, this_note_offset)
        note_end: float = _increase_semitones(note_begin, semitone_count)
        result += generate_chirp(
            sample_rate, note_begin, note_end, amplitude / note_count, duration
        )

    return result


def generate_named_chord_chirp(
    type: ChordType,
    sample_rate: int,
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    shape = None
    match type:
        case ChordType.Octave:
            shape = [12]
        case ChordType.Tritone:
            shape = [6]
        case ChordType.Major:
            shape = [4, 7]
        case ChordType.Minor:
            shape = [3, 7]
        case ChordType.Diminished:
            shape = [3, 6]
        case _:
            assert False
    return generate_generic_chord_chirp(
        sample_rate,
        shape,
        begin_note,
        begin_octave,
        end_note,
        end_octave,
        amplitude,
        duration,
    )
