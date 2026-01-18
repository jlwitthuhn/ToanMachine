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
NOISE_SHORT_DURATION = 1.25
NOISE_LONG_DURATION = 3.0
NOTE_DURATION = 0.73
PLUCK_DECAY = 0.988


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

    cosine_multiplier = generate_cosine_wave(len(sweep_up), sample_rate // 4, 0.08, 1.0)
    sine_multiplier = generate_sine_wave(len(sweep_up), sample_rate // 4, 0.08, 1.0)

    sweep_up_cos = sweep_up * cosine_multiplier
    sweep_up_sin = sweep_up * sine_multiplier

    warble_test = generate_warble_chord(sample_rate, 5.0, 220.0, ChordType.RootOnly)

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
            sweep_up_cos,
            sweep_up_sin,
            warble_test,
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
