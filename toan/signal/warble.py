# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.music.chord import ChordType
from toan.music.frequency import increase_frequency_by_semitones
from toan.signal.trig import generate_sine_wave

WARBLE_CYCLES_PER_SECOND = 2


def _generate_signal_from_frequencies(
    sample_rate: int, frequencies: np.ndarray
) -> np.ndarray:
    f_rad = 2 * np.pi * frequencies
    phase = np.cumsum(f_rad) / sample_rate
    return np.sin(phase)


def _generate_tone_warble(
    sample_rate: int, duration: float, frequency: float
) -> np.ndarray:
    sample_count = int(sample_rate * duration)
    frequencies = np.zeros(sample_count)
    frequencies.fill(frequency)
    frequency_multiplier = generate_sine_wave(
        sample_count, sample_rate // WARBLE_CYCLES_PER_SECOND, 0.99, 1.01
    )
    frequencies *= frequency_multiplier
    return _generate_signal_from_frequencies(sample_rate, frequencies)


def generate_warble_chord(
    sample_rate: int, duration: float, root_frequency: float, chord: ChordType
) -> np.ndarray:
    notes: list[int] = [0]
    for this_offset in chord.get_shape():
        notes.append(this_offset)
    note_signals = []
    for note in notes:
        frequency = increase_frequency_by_semitones(root_frequency, note)
        note_signal = _generate_tone_warble(sample_rate, duration, frequency)
        note_signals.append(note_signal)
    result = np.add.reduce(note_signals)
    result = result / np.abs(result).max()
    return result
