# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_chirp(
    sample_rate: float,
    begin_freq: float,
    end_freq: float,
    amplitude: float,
    duration: float,
) -> np.ndarray:
    f = np.logspace(
        np.log10(begin_freq), np.log10(end_freq), int(duration * sample_rate), False
    )
    f_rad = f * 2 * np.pi
    phase = np.cumsum(f_rad) / sample_rate
    return amplitude * np.sin(phase)
