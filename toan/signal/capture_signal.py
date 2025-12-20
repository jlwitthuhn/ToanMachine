# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.mix import concat_signals
from toan.music import get_note_frequency_by_name, get_note_index_by_name
from toan.signal.chirp import generate_chirp
from toan.signal.chord import generate_major_chord_chirp, generate_tritone_chirp
from toan.signal.gaussian import generate_gaussian_pulse
from toan.signal.noise import generate_white_noise
from toan.signal.scale import generate_chromatic_scale
from toan.signal.trig import generate_cosine_wave

SEGMENT_DURATION = 6.0


def generate_capture_signal(sample_rate: int, amplitude: float) -> np.ndarray:
    np.random.seed(12345)
    quarter_second_samples = sample_rate // 4

    # Quarter second of silence
    silence_quarter = np.zeros(quarter_second_samples)

    # Impulse to measure latency
    impulse_latency = np.ones(1) * amplitude

    # Impulses just for fun
    impulse0 = np.ones(1) * amplitude
    impulse1 = generate_gaussian_pulse(sample_rate // 4000) * amplitude
    impulse2 = generate_gaussian_pulse(sample_rate // 3000) * amplitude
    impulse3 = generate_gaussian_pulse(sample_rate // 2000) * amplitude
    impulse4 = generate_gaussian_pulse(sample_rate // 1500) * amplitude

    # Sweep of audible frequencies
    sweep_up = generate_chirp(sample_rate, 20.0, 20000.0, amplitude, SEGMENT_DURATION)

    # Lowest bass guitar note is E1
    scale_lo = get_note_frequency_by_name("E", 1, 440)
    index_lo = get_note_index_by_name("E", 1)
    # Guitar high E string played at the 24th fret is E6, add one octave to be safe
    index_hi = get_note_index_by_name("E", 7)
    scale_steps = index_hi - index_lo
    scale_list = generate_chromatic_scale(
        sample_rate, scale_lo, scale_steps, amplitude, 0.25
    )
    scale_step_samples = len(scale_list[0])
    scale_step_gaussian = generate_gaussian_pulse(scale_step_samples)
    for scale_step in scale_list:
        scale_step *= scale_step_gaussian
        pass
    scale = concat_signals(scale_list)

    sweep_major_chord = generate_major_chord_chirp(
        sample_rate, "E", 1, "E", 7, amplitude, SEGMENT_DURATION
    )
    sweep_tritone = generate_tritone_chirp(
        sample_rate, "E", 1, "E", 7, amplitude, SEGMENT_DURATION
    )
    assert len(sweep_major_chord) == len(sweep_tritone)
    cosine_multiplier = generate_cosine_wave(
        len(sweep_major_chord), sample_rate // 4, 0.1, 1.0
    )
    sweep_major_chord_cosine = sweep_major_chord * cosine_multiplier
    sweep_tritone_cosine = sweep_tritone * cosine_multiplier

    white_noise_full = generate_white_noise(sample_rate) * amplitude
    white_noise_half = white_noise_full * 0.5
    white_noise_quarter = white_noise_half * 0.5
    gaussian_samples = sample_rate * 2
    white_noise_gaussian = (
        generate_white_noise(gaussian_samples)
        * generate_gaussian_pulse(gaussian_samples)
        * amplitude
    )

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
            white_noise_full,
            white_noise_half,
            white_noise_quarter,
            white_noise_gaussian,
            sweep_up,
            scale,
            sweep_major_chord,
            sweep_major_chord_cosine,
            sweep_tritone,
            sweep_tritone_cosine,
        ],
        quarter_second_samples,
    )

    return concat_signals(
        [
            signal_calibrate_latency,
            silence_quarter,
            signal_train,
            silence_quarter,
            silence_quarter,
            silence_quarter,
            silence_quarter,
        ],
        0,
    )
