# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only
import math

import numpy as np

from toan.music.chord import ChordType
from toan.music.frequency import increase_frequency_by_semitones
from toan.signal.trig import generate_sine_wave

BIG_CYCLES_PER_SECOND = 3.0

PHASE_CYCLES_PER_SECOND = 4.0
MAXIMUM_PHASE_OFFSET = math.pi / 2.0

WARBLE_PITCH_BIG_MIN = 0.75
WARBLE_PITCH_BIG_MAX = 1.333

WARBLE_PITCH_SMALL_MIN = 0.99
WARBLE_PITCH_SMALL_MAX = 1.01


def _generate_signal_from_frequencies(
    sample_rate: int, frequencies: np.ndarray, phase_offset: float
) -> np.ndarray:
    f_rad = 2 * np.pi * frequencies
    phase = np.cumsum(f_rad) / sample_rate
    return np.sin(phase + phase_offset)


def _generate_tone_warble(
    sample_rate: int, modulation: np.ndarray, frequency: float, phase_offset: float
) -> np.ndarray:
    sample_count = len(modulation)
    frequencies = np.zeros(sample_count)
    frequencies.fill(frequency)
    frequencies *= modulation
    frequency_multiplier = generate_sine_wave(
        sample_count,
        int(sample_rate / PHASE_CYCLES_PER_SECOND),
        WARBLE_PITCH_SMALL_MIN,
        WARBLE_PITCH_SMALL_MAX,
    )
    frequencies *= frequency_multiplier
    return _generate_signal_from_frequencies(sample_rate, frequencies, phase_offset)


def _generate_top_level_modulation(sample_rate: int, duration: float) -> np.ndarray:
    sample_count = int(sample_rate * duration)
    return generate_sine_wave(
        sample_count,
        int(sample_rate / BIG_CYCLES_PER_SECOND),
        WARBLE_PITCH_BIG_MIN,
        WARBLE_PITCH_BIG_MAX,
    )


def generate_warble_chord(
    sample_rate: int,
    duration: float,
    root_frequency: float,
    chord: ChordType,
    octave_count: int,
    octave_scale: float,
) -> np.ndarray:
    modulation = _generate_top_level_modulation(sample_rate, duration)
    notes: list[int] = [0]
    for this_offset in chord.get_shape():
        notes.append(this_offset)
    octaves = []
    for octave_index in range(octave_count + 1):
        note_signals = []
        for index, note in enumerate(notes):
            if len(notes) > 1:
                phase: float = index / (len(notes) - 1) * MAXIMUM_PHASE_OFFSET
            else:
                phase = 0.0
            frequency = increase_frequency_by_semitones(
                root_frequency, note + 12 * octave_index
            )
            if frequency * WARBLE_PITCH_BIG_MAX >= sample_rate // 2:
                # Something something nyquist frequency
                break
            note_signal = _generate_tone_warble(
                sample_rate, modulation, frequency, phase
            )
            note_signals.append(note_signal)
        this_octave = np.add.reduce(note_signals)
        this_octave = this_octave / np.abs(this_octave).max()
        this_octave_scale = octave_scale**octave_index
        octaves.append(this_octave * this_octave_scale)
    result = np.add.reduce(octaves)
    result = result / np.abs(result).max()
    return result
