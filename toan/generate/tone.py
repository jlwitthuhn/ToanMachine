# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_tone(
    sample_rate: int, frequency: float, amplitude: float, duration: float, fade: bool
) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    out = amplitude * np.sin(2 * np.pi * frequency * t)
    if fade:
        # TODO: This should fade in the start of the tone over several samples to avoid hard edges
        fade_samples = sample_rate // 40
        fade_array = np.ma.core.ones_like(out)
        fade_array[:fade_samples] = np.linspace(0.0, 1.0, num=fade_samples)
        fade_array[-fade_samples:] = np.linspace(1.0, 0.0, num=fade_samples)
        assert out.shape == fade_array.shape
        out *= fade_array
    return out
