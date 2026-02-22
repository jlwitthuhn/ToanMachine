# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np

from toan.music.frequency import increase_frequency_by_semitones

# Control how aggressive the low-pass filter is
# Numbers closer to 0.5 will filter the most
SPLIT_A = 0.60


# Generate a pluck using a Karplus-Strong filter over random noise
def generate_pluck(
    sample_rate: int,
    frequency: float,
    duration: float,
    decay: float,
    pre_smooth: int = 0,
) -> np.ndarray:
    out_sample_count = int(duration * sample_rate)
    result = np.zeros(out_sample_count)

    buffer_width = int(sample_rate / frequency)

    buffer = np.random.uniform(-1.0, 1.0, buffer_width)
    for _ in range(pre_smooth):
        for i in range(buffer_width):
            buffer[i] = buffer[i] * SPLIT_A + buffer[i - 1] * (1.0 - SPLIT_A)
    buffer = buffer / np.max(np.abs(buffer))

    previous: float = 0.0

    for i in range(out_sample_count):
        buffer_index = i % buffer_width
        value = decay * (buffer[buffer_index] * SPLIT_A + previous * (1.0 - SPLIT_A))
        buffer[buffer_index] = value
        result[i] = value
        previous = value

    return result


def generate_generic_chord_pluck(
    sample_rate: int,
    shape: list[int],
    root_frequency: float,
    duration: float,
    offset_duration: float = 1.8e-3,
    decay: float = 0.99,
    pre_smooth: int = 0,
) -> np.ndarray:
    frequencies: list[float] = [root_frequency]
    for extra_semitones in shape:
        frequencies.append(
            increase_frequency_by_semitones(root_frequency, extra_semitones)
        )

    pluck_list: list[np.ndarray] = []
    for idx, frequency in enumerate(frequencies):
        offset = int(idx * offset_duration * sample_rate)
        pluck_raw = generate_pluck(sample_rate, frequency, duration, decay, pre_smooth)
        if offset == 0:
            pluck_list.append(pluck_raw)
        else:
            offset_buffer = np.zeros(offset)
            pluck_list.append(np.concatenate((offset_buffer, pluck_raw[:-offset])))

    chord = np.add.reduce(pluck_list)
    chord = chord / np.abs(chord).max()
    return chord
