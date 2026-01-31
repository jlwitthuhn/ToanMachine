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


def generate_capture_signal(sample_rate: int) -> np.ndarray:
    np.random.seed(12345)
    quarter_second_samples = sample_rate // 4

    # Quarter second of silence
    silence_quarter = np.zeros(quarter_second_samples)

    # Impulse to measure latency
    impulse_latency = np.ones(1)

    # Impulses just for fun
    impulse0 = np.ones(1)
    impulse1 = generate_gaussian_pulse(sample_rate // 4000)
    impulse2 = generate_gaussian_pulse(sample_rate // 3000)
    impulse3 = generate_gaussian_pulse(sample_rate // 2500)
    impulse4 = generate_gaussian_pulse(sample_rate // 2100)
    impulse5 = generate_gaussian_pulse(sample_rate // 1800)
    impulse6 = generate_gaussian_pulse(sample_rate // 1500)
    impulse7 = generate_gaussian_pulse(sample_rate // 1350)

    # Sweep of audible frequencies
    sweep_up = generate_chirp(sample_rate, 18.0, 21000.0, SWEEP_DURATION)
    sweep_down = generate_chirp(sample_rate, 21000.0, 18.0, SWEEP_DURATION / 2)

    cosine_multiplier = generate_cosine_wave(
        len(sweep_down), sample_rate // 5, 0.08, 1.0
    )
    sine_multiplier = generate_sine_wave(len(sweep_down), sample_rate // 5, 0.08, 1.0)

    sweep_down_cos = sweep_down * cosine_multiplier
    sweep_down_sin = sweep_down * sine_multiplier

    def generate_warble_signal(shape: ChordType):
        return generate_warble_chord(
            sample_rate, SWEEP_DURATION / 2, 55.0, shape, 10, 0.68
        )

    warble_octave = generate_warble_signal(ChordType.Octave)
    warble_modulation = generate_gaussian_pulse(
        len(warble_octave), len(warble_octave) // 3
    )
    warble_octave *= warble_modulation

    warble_perfect_fifth = generate_warble_signal(ChordType.PerfectFifth)
    warble_perfect_fifth *= warble_modulation

    warble_diminished = generate_warble_signal(ChordType.Diminished)
    warble_diminished *= warble_modulation

    warble_major = generate_warble_signal(ChordType.Major)
    warble_major *= warble_modulation

    warble_minor_seventh = generate_warble_signal(ChordType.MinorSeventh)
    warble_minor_seventh *= warble_modulation

    warble_guitar = generate_warble_signal(ChordType.GuitarStrings)
    warble_guitar *= warble_modulation

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

    scale_root = generate_plucked_scale(ChordType.RootOnly, 0.0)
    scale_tritone_chord = generate_plucked_scale(ChordType.Tritone, 1.8e-3)
    scale_major_chord = generate_plucked_scale(ChordType.Major, 2.2e-3)
    scale_minor_chord = generate_plucked_scale(ChordType.Minor, 2.6e-3)
    scale_major_seventh_chord = generate_plucked_scale(ChordType.MajorSeventh, 3.0e-3)
    scale_minor_ninth_chord = generate_plucked_scale(ChordType.MinorNinth, 3.4e-3)
    scale_guitar_chord = generate_plucked_scale(ChordType.GuitarStrings, 3.8e-3)

    noise_samples_short = int(sample_rate * NOISE_SHORT_DURATION)
    white_noise_full = generate_white_noise(noise_samples_short)
    white_noise_half = white_noise_full * 0.5
    white_noise_quarter = white_noise_half * 0.5
    gaussian_samples = int(sample_rate * NOISE_LONG_DURATION)
    white_noise_gaussian = generate_white_noise(
        gaussian_samples
    ) * generate_gaussian_pulse(gaussian_samples)

    signal_calibrate_latency = concat_signals(
        [
            silence_quarter,
            impulse_latency,
            impulse_latency,
            silence_quarter,
        ],
        quarter_second_samples,
    )

    signal_train = concat_signals(
        [
            impulse0,
            impulse1,
            impulse2,
            impulse3,
            impulse4,
            impulse5,
            impulse6,
            impulse7,
            sweep_up,
            sweep_down_cos,
            sweep_down_sin,
            warble_octave,
            warble_perfect_fifth,
            warble_diminished,
            warble_major,
            warble_minor_seventh,
            warble_guitar,
            scale_root,
            scale_tritone_chord,
            scale_major_chord,
            scale_minor_chord,
            scale_major_seventh_chord,
            scale_minor_ninth_chord,
            scale_guitar_chord,
            white_noise_full,
            white_noise_half,
            white_noise_quarter,
            white_noise_gaussian,
        ],
        quarter_second_samples,
    )

    return concat_signals(
        [
            signal_calibrate_latency,
            silence_quarter,
            silence_quarter,
            signal_train,
            silence_quarter,
            silence_quarter,
            silence_quarter,
            silence_quarter,
        ],
        0,
    )
