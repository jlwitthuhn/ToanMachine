# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from enum import Enum

import numpy as np

from toan.signal.effect.delay import effect_delay


class EffectType(Enum):
    Nothing = enum.auto()
    Delay01 = enum.auto()
    Delay02 = enum.auto()


def apply_effect(
    signal: np.ndarray, sample_rate: int, effect: EffectType, normalize: bool = True
) -> None:
    match effect:
        case EffectType.Nothing:
            pass
        case EffectType.Delay01:
            effect_delay(signal, sample_rate, 0.5)
        case EffectType.Delay02:
            effect_delay(signal, sample_rate // 2, 0.5)
    if normalize:
        signal /= np.max(np.abs(signal))
