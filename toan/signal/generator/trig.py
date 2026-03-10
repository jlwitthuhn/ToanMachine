# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_cosine_wave(
    samples: int, period: int, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    t = np.arange(samples)
    raw = np.cos(2 * np.pi * t / period)
    return min_val + (raw + 1.0) * (max_val - min_val) / 2.0


def generate_sine_wave(
    samples: int, period: int, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    t = np.arange(samples)
    raw = np.sin(2 * np.pi * t / period)
    return min_val + (raw + 1.0) * (max_val - min_val) / 2.0
