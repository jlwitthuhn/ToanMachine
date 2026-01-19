# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import math

import numpy as np

from toan.music.chord import ChordType
from toan.music.frequency import increase_frequency_by_semitones
from toan.signal.trig import generate_sine_wave

WARBLE_CYCLES_PER_SECOND = 2
MAXIMUM_PHASE_OFFSET = math.pi / 4


def _generate_signal_from_frequencies(
    sample_rate: int, frequencies: np.ndarray, phase_offset: float
) -> np.ndarray:
    print(frequencies[0], phase_offset)
    f_rad = 2 * np.pi * frequencies
    phase = np.cumsum(f_rad) / sample_rate
    return np.sin(phase + phase_offset)


def _generate_tone_warble(
    sample_rate: int, duration: float, frequency: float, phase_offset: float
) -> np.ndarray:
    sample_count = int(sample_rate * duration)
    frequencies = np.zeros(sample_count)
    frequencies.fill(frequency)
    frequency_multiplier = generate_sine_wave(
        sample_count, sample_rate // WARBLE_CYCLES_PER_SECOND, 0.99, 1.01
    )
    frequencies *= frequency_multiplier
    return _generate_signal_from_frequencies(sample_rate, frequencies, phase_offset)


def generate_warble_chord(
    sample_rate: int, duration: float, root_frequency: float, chord: ChordType
) -> np.ndarray:
    notes: list[int] = [0]
    for this_offset in chord.get_shape():
        notes.append(this_offset)
    note_signals = []
    for index, note in enumerate(notes):
        if len(notes) > 1:
            phase: float = index / (len(notes) - 1) * MAXIMUM_PHASE_OFFSET
        else:
            phase = 0.0
        frequency = increase_frequency_by_semitones(root_frequency, note)
        note_signal = _generate_tone_warble(sample_rate, duration, frequency, phase)
        note_signals.append(note_signal)
    result = np.add.reduce(note_signals)
    result = result / np.abs(result).max()
    return result
