# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum

import numpy as np

from toan.signal.effect.delay import effect_delay
from toan.signal.effect.vibrato import effect_vibrato


class EffectType(Enum):
    Nothing = enum.auto()
    Delay0100 = enum.auto()
    Delay0200 = enum.auto()
    Delay0400 = enum.auto()
    FeedbackDelay0100 = enum.auto()
    FeedbackDelay0200 = enum.auto()
    FeedbackDelay0400 = enum.auto()
    Vibrato4Hz = enum.auto()
    Vibrato7Hz = enum.auto()
    Flanger4Hz = enum.auto()
    Flanger7Hz = enum.auto()


def apply_effect(
    signal: np.ndarray, sample_rate: int, effect: EffectType, normalize: bool = True
) -> None:
    match effect:
        case EffectType.Nothing:
            pass
        case EffectType.Delay0100:
            effect_delay(signal, int(sample_rate * 0.1), 0.5, False)
        case EffectType.Delay0200:
            effect_delay(signal, int(sample_rate * 0.2), 0.5, False)
        case EffectType.Delay0400:
            effect_delay(signal, int(sample_rate * 0.4), 0.5, False)
        case EffectType.FeedbackDelay0100:
            effect_delay(signal, int(sample_rate * 0.1), 0.4, True)
        case EffectType.FeedbackDelay0200:
            effect_delay(signal, int(sample_rate * 0.2), 0.4, True)
        case EffectType.FeedbackDelay0400:
            effect_delay(signal, int(sample_rate * 0.4), 0.4, True)
        case EffectType.Vibrato4Hz:
            effect_vibrato(signal, sample_rate, 4.0, 0.002)
        case EffectType.Vibrato7Hz:
            effect_vibrato(signal, sample_rate, 7.0, 0.002)
        case EffectType.Flanger4Hz:
            effect_vibrato(signal, sample_rate, 4.0, 0.001, 0.5)
        case EffectType.Flanger4Hz:
            effect_vibrato(signal, sample_rate, 7.0, 0.001, 0.5)
    if normalize:
        signal /= np.max(np.abs(signal))
