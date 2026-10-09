# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from toan.music.frequency import increase_frequency_by_semitones

# Partials above this are left out, a pickup barely passes them anyway
MAX_PARTIAL_FREQUENCY = 20000.0

# Muting the string at the end of a note fades it by this much
RELEASE_DEPTH_DB = 80.0


@dataclass
class PluckConfig:
    # Seconds for the lowest partials to decay by 60 dB
    sustain: float = 7.5
    # Partials at this frequency decay twice as fast as the lowest ones
    damping_frequency: float = 800.0
    # Where the string is picked and where the pickup sits,
    # as fractions of the string length measured from the bridge
    pick_position: float = 0.18
    pickup_position: float = 0.10
    # How much the pick position varies between notes, relative to pick_position
    pick_position_jitter: float = 0.10
    # String stiffness, which stretches upper partials sharp
    inharmonicity: float = 5e-5
    # The pickup coil and cable capacitance form a resonant low-pass filter
    pickup_resonance: float = 3500.0
    pickup_q: float = 2.0
    # Seconds at the end of each note spent muting the string
    release: float = 0.08


# Sum amplitude * exp(-decay_rate * t) * sin(2 * pi * frequency * t) over all partials
def _sum_partials(
    sample_count: int,
    sample_rate: int,
    frequencies: np.ndarray,
    decay_rates: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    # Writing each sample index as block_index * block_size + offset splits every
    # exponential into two factors, which turns the whole sum into one complex
    # matrix product instead of one exponential per partial per sample
    block_size = max(1, math.isqrt(sample_count))
    block_count = -(-sample_count // block_size)
    poles = (2j * math.pi * frequencies - decay_rates) / sample_rate
    block_starts = np.exp(np.outer(np.arange(block_count) * block_size, poles))
    offsets = np.exp(np.outer(poles, np.arange(block_size)))
    blocks = (block_starts * amplitudes) @ offsets
    return blocks.imag.reshape(-1)[:sample_count]


# Second-order resonant low-pass from the RBJ audio EQ cookbook
def _apply_pickup_filter(
    signal: np.ndarray, sample_rate: int, resonance: float, q: float
) -> np.ndarray:
    w0 = 2.0 * math.pi * resonance / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b = [(1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0]
    a = [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha]
    return lfilter(b, a, signal)


# Generate a pluck with modal synthesis, as a sum of exponentially decaying partials
# shaped like the output of a magnetic pickup under a picked string
def generate_pluck(
    sample_rate: int,
    sample_count: int,
    frequency: float,
    config: PluckConfig = PluckConfig(),
) -> np.ndarray:
    # One period of noise is drawn per note to keep the global random stream, and with
    # it every capture signal block generated after the plucks, the same as when this
    # was a Karplus-Strong generator. Its first value varies the pick position.
    noise = np.random.uniform(-1.0, 1.0, int(sample_rate / frequency))
    pick_position = config.pick_position * (
        1.0 + config.pick_position_jitter * noise[0]
    )

    # Stiffness moves partial n to n * sqrt(1 + B * n^2),
    # normalized so the fundamental stays at the requested frequency
    max_frequency = min(MAX_PARTIAL_FREQUENCY, 0.45 * sample_rate)
    n = np.arange(1, int(max_frequency / frequency) + 1)
    b = config.inharmonicity
    frequencies = frequency * n * np.sqrt((1.0 + b * n * n) / (1.0 + b))
    n = n[frequencies < max_frequency]
    frequencies = frequencies[frequencies < max_frequency]

    # A string released from rest by a pick has mode amplitudes of
    # sin(n * pi * pick) / n^2. The pickup senses string velocity, which scales
    # mode n by n, at its own spot along the string, which scales it by
    # sin(n * pi * pickup).
    amplitudes = (
        np.sin(n * math.pi * pick_position)
        * np.sin(n * math.pi * config.pickup_position)
        / n
    )

    decay_rates = (
        math.log(1000.0)
        / config.sustain
        * (1.0 + (frequencies / config.damping_frequency) ** 2)
    )

    result = _sum_partials(
        sample_count, sample_rate, frequencies, decay_rates, amplitudes
    )

    # Mute the string at the end so the note fades out instead of being cut off.
    # Damping builds up gradually, an abrupt change would add a click of its own.
    release_count = min(sample_count, int(config.release * sample_rate))
    if release_count > 0:
        release_steps = np.arange(1, release_count + 1) / release_count
        result[-release_count:] *= 10.0 ** (-RELEASE_DEPTH_DB / 20.0 * release_steps**2)

    result = _apply_pickup_filter(
        result, sample_rate, config.pickup_resonance, config.pickup_q
    )

    peak = np.max(np.abs(result), initial=0.0)
    if peak > 0.0:
        result = result / peak
    return result


def generate_generic_chord_pluck(
    sample_rate: int,
    shape: list[int],
    root_frequency: float,
    duration: float,
    offset_duration: float = 2.0e-3,
    config: PluckConfig = PluckConfig(),
) -> np.ndarray:
    frequencies: list[float] = [root_frequency]
    for extra_semitones in shape:
        frequencies.append(
            increase_frequency_by_semitones(root_frequency, extra_semitones)
        )

    pluck_list: list[np.ndarray] = []
    for idx, frequency in enumerate(frequencies):
        offset = int(idx * offset_duration * sample_rate)
        sample_count = math.floor(sample_rate * duration) - offset
        pluck_raw = generate_pluck(sample_rate, sample_count, frequency, config)
        if offset == 0:
            pluck_list.append(pluck_raw)
        else:
            pluck_list.append(np.concatenate((np.zeros(offset), pluck_raw)))

    chord = np.add.reduce(pluck_list)
    chord = chord / np.abs(chord).max()
    return chord
