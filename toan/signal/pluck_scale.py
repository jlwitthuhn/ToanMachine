# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from toan.mix import concat_signals
from toan.music import get_note_frequency_by_name, get_note_index_by_name
from toan.music.chord import ChordType
from toan.signal.pluck import generate_generic_chord_pluck


def _increase_semitones(frequency: float, semitones: int) -> float:
    return frequency * math.pow(2, semitones / 12)


def generate_generic_chord_pluck_scale(
    sample_rate: int,
    shape: list[int],
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    single_duration: float,
    offset_duration: float = 1.8e-3,
    decay: float = 0.99,
) -> np.ndarray:
    begin_index = get_note_index_by_name(begin_note, begin_octave)
    end_index = get_note_index_by_name(end_note, end_octave)
    semitone_width = max([0] + shape)
    semitone_count = end_index - begin_index - semitone_width
    assert 0 <= semitone_width <= semitone_count

    chord_list: list[np.ndarray] = []
    for i in range(semitone_count):
        root_frequency: float = get_note_frequency_by_name(
            begin_note, begin_octave, 440.0
        )
        root_frequency = _increase_semitones(root_frequency, i)
        this_chord = generate_generic_chord_pluck(
            sample_rate, shape, root_frequency, single_duration, offset_duration, decay
        )
        chord_list.append(this_chord)

    return concat_signals(chord_list, sample_rate // 16)


def generate_named_chord_pluck_scale(
    type: ChordType,
    sample_rate: int,
    begin_note: str,
    begin_octave: int,
    end_note: str,
    end_octave: int,
    single_duration: float,
    offset_duration: float = 1.8e-3,
    decay: float = 0.99,
) -> np.ndarray:
    shape = type.get_shape()
    return generate_generic_chord_pluck_scale(
        sample_rate,
        shape,
        begin_note,
        begin_octave,
        end_note,
        end_octave,
        single_duration,
        offset_duration,
        decay,
    )
