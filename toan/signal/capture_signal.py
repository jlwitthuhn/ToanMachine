# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.mix import concat_signals
from toan.music.chord import ChordType
from toan.signal.chirp import generate_chirp
from toan.signal.gaussian import generate_gaussian_pulse
from toan.signal.noise import generate_white_noise
from toan.signal.pluck_scale import generate_named_chord_pluck_scale
from toan.signal.trig import generate_cosine_wave, generate_sine_wave
from toan.signal.warble import generate_warble_chord

SWEEP_DURATION = 12.0
NOISE_SHORT_DURATION = 1.5
NOISE_LONG_DURATION = 3.5
NOTE_DURATION = 0.70
PLUCK_DECAY = 0.985


def _generate_calibration_block(sample_rate: int) -> np.ndarray:
    silence_quarter = np.zeros(sample_rate // 4)
    impulse = np.ones(1)

    return concat_signals(
        [
            silence_quarter,
            impulse,
            impulse,
            silence_quarter,
        ],
        sample_rate // 4,
    )


def _generate_impulse_block(sample_rate: int) -> np.ndarray:
    impulse0 = np.ones(1)
    impulse1 = generate_gaussian_pulse(sample_rate // 4000)
    impulse2 = generate_gaussian_pulse(sample_rate // 3000)
    impulse3 = generate_gaussian_pulse(sample_rate // 2500)
    impulse4 = generate_gaussian_pulse(sample_rate // 2100)
    impulse5 = generate_gaussian_pulse(sample_rate // 1800)
    impulse6 = generate_gaussian_pulse(sample_rate // 1500)
    impulse7 = generate_gaussian_pulse(sample_rate // 1350)
    return concat_signals(
        [
            impulse0,
            impulse1,
            impulse2,
            impulse3,
            impulse4,
            impulse5,
            impulse6,
            impulse7,
        ],
        sample_rate // 4,
    )


def _generate_sweep_block(sample_rate: int, duration: float) -> np.ndarray:
    sweep_up = generate_chirp(sample_rate, 18.0, 22000.0, duration)
    sweep_down = generate_chirp(sample_rate, 22000.0, 18.0, duration / 2)

    cosine_multiplier = generate_cosine_wave(
        len(sweep_down), sample_rate // 5, 0.08, 1.0
    )
    sine_multiplier = generate_sine_wave(len(sweep_down), sample_rate // 5, 0.08, 1.0)

    sweep_down_cos = sweep_down * cosine_multiplier
    sweep_down_sin = sweep_down * sine_multiplier

    return concat_signals(
        [
            sweep_up,
            sweep_down_cos,
            sweep_down_sin,
        ],
        sample_rate // 4,
    )


def _generate_warble_block(sample_rate: int, chords: list[ChordType]) -> np.ndarray:
    def generate_warble_signal(shape: ChordType):
        return generate_warble_chord(
            sample_rate, SWEEP_DURATION / 2, 55.0, shape, 10, 0.68
        )

    buffer_size = int(sample_rate * SWEEP_DURATION / 2)
    modulation = generate_gaussian_pulse(buffer_size, buffer_size // 3)
    chord_buffers = []
    for chord in chords:
        this_chord_buffer = generate_warble_signal(chord)
        assert len(modulation) == len(this_chord_buffer)
        chord_buffers.append(this_chord_buffer * modulation)
    return concat_signals(chord_buffers, sample_rate // 4)


def _generate_plucked_block(sample_rate: int, chords: list[ChordType]) -> np.ndarray:
    def generate_plucked_scale(shape: ChordType, offset_duration: float):
        return generate_named_chord_pluck_scale(
            shape,
            sample_rate,
            "E",
            1,
            "G",
            6,
            NOTE_DURATION,
            offset_duration,
            PLUCK_DECAY,
        )

    buffers = []
    for i, chord in enumerate(chords):
        offset = i * 0.6e-3
        buffers.append(generate_plucked_scale(chord, offset))
    return concat_signals(buffers, sample_rate // 4)


def _generate_white_noise_block(sample_rate: int) -> np.ndarray:
    noise_samples_short = int(sample_rate * NOISE_SHORT_DURATION)
    white_noise_full = generate_white_noise(noise_samples_short)
    white_noise_half = white_noise_full * 0.5
    white_noise_quarter = white_noise_half * 0.5
    gaussian_samples = int(sample_rate * NOISE_LONG_DURATION)
    white_noise_gaussian = generate_white_noise(
        gaussian_samples
    ) * generate_gaussian_pulse(gaussian_samples)
    return concat_signals(
        [white_noise_full, white_noise_half, white_noise_quarter, white_noise_gaussian],
        sample_rate // 4,
    )


def generate_capture_signal(sample_rate: int) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(0x35)

    warble_chords: list[ChordType] = [
        ChordType.PerfectFifth,
        ChordType.Diminished,
        ChordType.Major,
        ChordType.MinorSeventh,
        ChordType.GuitarStrings,
    ]

    plucked_chords: list[ChordType] = [
        ChordType.RootOnly,
        ChordType.Tritone,
        ChordType.Major,
        ChordType.Minor,
        ChordType.MajorSeventh,
        ChordType.MinorNinth,
        ChordType.GuitarStrings,
    ]

    block_impulse = _generate_impulse_block(sample_rate)
    block_sweep = _generate_sweep_block(sample_rate, SWEEP_DURATION)
    block_warble = _generate_warble_block(sample_rate, warble_chords)
    block_plucked = _generate_plucked_block(sample_rate, plucked_chords)
    block_white_noise = _generate_white_noise_block(sample_rate)

    signal_train = concat_signals(
        [
            block_impulse,
            block_sweep,
            block_warble,
            block_plucked,
            block_white_noise,
        ],
        sample_rate // 4,
    )

    block_calibration = _generate_calibration_block(sample_rate)

    np.random.set_state(rng_state)

    silence_half_second = np.zeros(sample_rate // 2)

    return concat_signals(
        [
            block_calibration,
            silence_half_second,
            signal_train,
            silence_half_second,
            silence_half_second,
        ],
        0,
    )
