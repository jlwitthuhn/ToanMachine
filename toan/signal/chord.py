# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
from enum import Enum

import numpy as np

from toan.mix import concat_signals
from toan.music import get_note_frequency_by_name, get_note_index_by_name
from toan.signal.pluck import generate_pluck


def _increase_semitones(frequency: float, semitones: int) -> float:
    return frequency * math.pow(2, semitones / 12)


class ChordType(Enum):
    # 2 notes
    Octave = 1
    Tritone = 2
    # 3 notes
    Major = 3
    Minor = 4
    Diminished = 5
    # 5 notes
    MinorNinth = 6
    # 6 notes
    GuitarStrings = 7


def generate_generic_chord_pluck_scale(
    sample_rate: int,
    shape: list[int],
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    single_duration: float,
) -> np.ndarray:
    begin_index = get_note_index_by_name(begin_note, begin_octave)
    end_index = get_note_index_by_name(end_note, end_octave)
    semitone_width = max(shape)
    semitone_count = end_index - begin_index - semitone_width
    assert 0 < semitone_width <= semitone_count

    chord_list: list[np.ndarray] = []
    for i in range(semitone_count):
        root_frequency: float = get_note_frequency_by_name(
            begin_note, begin_octave, 440.0
        )
        root_frequency = _increase_semitones(root_frequency, i)
        frequencies: list[float] = [root_frequency]
        for extra_semitones in shape:
            frequencies.append(_increase_semitones(root_frequency, extra_semitones))

        pluck_list: list[np.ndarray] = []
        for frequency in frequencies:
            pluck = generate_pluck(sample_rate, frequency, single_duration)
            pluck_list.append(pluck)

        plucks = np.add.reduce(pluck_list)
        plucks = plucks / np.abs(plucks).max()
        chord_list.append(plucks)

    return concat_signals(chord_list, sample_rate // 10)


def generate_named_chord_pluck_scale(
    type: ChordType,
    sample_rate: int,
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    single_duration: float,
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
        case ChordType.MinorNinth:
            shape = [3, 7, 10, 14]
        case ChordType.GuitarStrings:
            shape = [5, 10, 15, 19, 24]
        case _:
            assert False
    return generate_generic_chord_pluck_scale(
        sample_rate,
        shape,
        begin_note,
        begin_octave,
        end_note,
        end_octave,
        single_duration,
    )
