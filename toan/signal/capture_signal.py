# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

import numpy as np

from toan.music.chord import ChordType
from toan.persistence.builtin_wav import BuiltinWav, get_builtin_wav_signal
from toan.signal.effect import EffectType, apply_effect
from toan.signal.generator.chirp import generate_chirp
from toan.signal.generator.gaussian import generate_gaussian_pulse
from toan.signal.generator.noise import generate_white_noise
from toan.signal.generator.pluck_scale import generate_named_chord_pluck_scale
from toan.signal.generator.trig import generate_cosine_wave, generate_sine_wave
from toan.signal.generator.warble import generate_warble_chord
from toan.signal.mix import concat_signals


@dataclass
class ChordWithEffects:
    chord: ChordType
    effect: EffectType


@dataclass
class CaptureSignalConfig:
    sweep_duration: float = 10.0
    warble_duration: float = 6.5
    noise_duration: float = 8.0
    pluck_note_duration: float = 0.65
    pluck_decay: float = 0.982
    pluck_pre_smooth: int = 1
    warble_chords: list[ChordWithEffects] = field(
        default_factory=lambda: [
            ChordWithEffects(ChordType.Tritone, EffectType.Flanger4Hz),
            ChordWithEffects(ChordType.MajorSeventh, EffectType.Nothing),
            ChordWithEffects(ChordType.Octave, EffectType.Nothing),
            ChordWithEffects(ChordType.AugmentedTriad, EffectType.Nothing),
            ChordWithEffects(ChordType.MinorTriad, EffectType.Nothing),
            ChordWithEffects(ChordType.AugmentedMajorSeventhTetrad, EffectType.Nothing),
            ChordWithEffects(ChordType.DiminishedSeventhTetrad, EffectType.Nothing),
            ChordWithEffects(ChordType.DiminishedSeventhTetrad, EffectType.Nothing),
            ChordWithEffects(ChordType.MinorMajorSeventhTetrad, EffectType.Nothing),
        ]
    )
    plucked_chords: list[ChordWithEffects] = field(
        default_factory=lambda: [
            ChordWithEffects(ChordType.MajorSixth, EffectType.Nothing),
            ChordWithEffects(ChordType.MinorThird, EffectType.Nothing),
            ChordWithEffects(ChordType.MajorTriad, EffectType.Nothing),
            ChordWithEffects(ChordType.DominantSeventhTetrad, EffectType.Delay0400),
        ]
    )
    builtin_wavs: list[BuiltinWav] = field(
        default_factory=lambda: [
            BuiltinWav.T3K_BASS_ROLLIN,
            BuiltinWav.T3K_GUITAR_CREAM,
        ]
    )
    rand_seed: int = 0x35


@dataclass
class CaptureSignalWithDetails:
    signal: np.ndarray
    sample_rate: int
    segment_clicks: tuple[int, int]
    segment_train: tuple[int, int]
    segment_sweep: tuple[int, int]


def _generate_calibration_block(sample_rate: int) -> np.ndarray:
    silence_quarter = np.zeros(sample_rate // 4)
    impulse = np.ones(1) * 0.5

    return concat_signals(
        [
            silence_quarter,
            impulse,
            impulse,
            silence_quarter,
        ],
        sample_rate // 4,
    )


def _generate_sweep_block(sample_rate: int, duration: float) -> tuple[np.ndarray, int]:
    sweep_max = min(24000, sample_rate // 2)
    sweep_up = generate_chirp(sample_rate, 18.0, sweep_max, duration)
    sweep_down = generate_chirp(sample_rate, sweep_max, 18.0, duration / 2)

    sweep_up_end = len(sweep_up)

    cosine_multiplier = generate_cosine_wave(
        len(sweep_down), sample_rate // 5, 0.08, 1.0
    )
    sine_multiplier = generate_sine_wave(len(sweep_down), sample_rate // 5, 0.08, 1.0)

    sweep_down_cos = sweep_down * cosine_multiplier
    sweep_down_sin = sweep_down * sine_multiplier

    sweep_pairs = [
        (500, sweep_max),
        (750, sweep_max),
        (1000, sweep_max),
        (1500, sweep_max),
        (2000, sweep_max),
        (3000, sweep_max),
        (4000, sweep_max),
        (6000, sweep_max),
        (8000, sweep_max),
        (12000, sweep_max),
    ]
    extra_sweeps = []
    for f_start, f_end in sweep_pairs:
        extra_sweeps.append(generate_chirp(sample_rate, f_start, f_end, 0.36))
        extra_sweeps.append(generate_chirp(sample_rate, f_end, f_start, 0.36))
    extra_block = concat_signals(extra_sweeps, sample_rate // 24)

    return (
        concat_signals(
            [
                sweep_up,
                sweep_down_cos,
                sweep_down_sin,
                extra_block,
            ],
            sample_rate // 4,
        ),
        sweep_up_end,
    )


def _generate_warble_block(
    sample_rate: int, chords: list[ChordWithEffects], duration: float
) -> np.ndarray:
    if len(chords) == 0:
        return np.zeros(1)

    def generate_warble_signal(shape: ChordType):
        return generate_warble_chord(sample_rate, duration, 55.0, shape, 10, 0.68)

    buffer_size = int(sample_rate * duration)
    modulation = generate_gaussian_pulse(buffer_size, buffer_size // 3)
    chord_buffers = []
    for chord in chords:
        this_chord_buffer = generate_warble_signal(chord.chord)
        this_chord_buffer = apply_effect(this_chord_buffer, sample_rate, chord.effect)
        assert len(modulation) == len(this_chord_buffer)
        chord_buffers.append(this_chord_buffer * modulation)
    return concat_signals(chord_buffers, sample_rate // 4)


def _generate_plucked_block(
    sample_rate: int,
    chords: list[ChordWithEffects],
    note_duration: float,
    pluck_decay: float,
    pre_smooth: int = 0,
) -> np.ndarray:
    if len(chords) == 0:
        return np.zeros(1)

    def generate_plucked_scale(shape: ChordType, offset_duration: float):
        return generate_named_chord_pluck_scale(
            shape,
            sample_rate,
            "E",
            1,
            "G",
            6,
            note_duration,
            offset_duration,
            pluck_decay,
            pre_smooth,
        )

    buffers = []
    for i, chord in enumerate(chords):
        offset = i * 0.6e-3
        this_chord_buffer = generate_plucked_scale(chord.chord, offset)
        this_chord_buffer = apply_effect(this_chord_buffer, sample_rate, chord.effect)
        buffers.append(this_chord_buffer)
    return concat_signals(buffers, sample_rate // 4)


def _generate_white_noise_block(sample_rate: int, duration: float) -> np.ndarray:
    samples = int(sample_rate * duration)
    plateau_width = samples // 5
    white_noise = generate_white_noise(samples)
    pulse = generate_gaussian_pulse(samples, plateau_width)
    return white_noise * pulse


def _generate_builtin_wav_block(sample_rate: int, wavs: list[BuiltinWav]) -> np.ndarray:
    if len(wavs) == 0:
        return np.zeros(1)
    buffers = []
    for wav in wavs:
        this_signal = get_builtin_wav_signal(sample_rate, wav)
        buffers.append(this_signal)
    return concat_signals(buffers, sample_rate // 4)


def generate_capture_signal(
    sample_rate: int, config: CaptureSignalConfig = CaptureSignalConfig()
) -> CaptureSignalWithDetails:
    rng_state = np.random.get_state()
    np.random.seed(config.rand_seed)

    main_sweep_begin = 0
    block_sweep, main_sweep_end = _generate_sweep_block(
        sample_rate, config.sweep_duration
    )
    block_warble = _generate_warble_block(
        sample_rate, config.warble_chords, config.warble_duration
    )
    block_plucked = _generate_plucked_block(
        sample_rate,
        config.plucked_chords,
        config.pluck_note_duration,
        config.pluck_decay,
        config.pluck_pre_smooth,
    )
    block_white_noise = _generate_white_noise_block(sample_rate, config.noise_duration)
    block_builtin_wavs = _generate_builtin_wav_block(sample_rate, config.builtin_wavs)

    main_sweep_begin += 0
    main_sweep_end += 0
    signal_train = concat_signals(
        [
            block_sweep,
            block_warble,
            block_plucked,
            block_white_noise,
            block_builtin_wavs,
        ],
        sample_rate // 4,
    )

    block_calibration = _generate_calibration_block(sample_rate)

    np.random.set_state(rng_state)

    silence_half_second = np.zeros(sample_rate // 2)

    main_sweep_begin += len(block_calibration) + len(silence_half_second)
    main_sweep_end += len(block_calibration) + len(silence_half_second)
    raw_signal = concat_signals(
        [
            block_calibration,
            silence_half_second,
            signal_train,
            silence_half_second,
            silence_half_second,
        ],
        0,
    )
    segment_clicks = (0, len(block_calibration) + len(silence_half_second))
    segment_train = (len(block_calibration), len(raw_signal))
    segment_sweep = (main_sweep_begin, main_sweep_end)
    return CaptureSignalWithDetails(
        raw_signal,
        sample_rate,
        segment_clicks,
        segment_train,
        segment_sweep,
    )
