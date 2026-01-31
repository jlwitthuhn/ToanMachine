# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_gaussian_pulse(width: int, plateau_width: int = 0) -> np.ndarray:
    real_width = width - plateau_width
    sigma = real_width * 0.16
    n = np.arange(real_width)
    mu = (real_width - 1) / 2
    result = np.exp(-0.5 * ((n - mu) / sigma) ** 2)
    if plateau_width > 0:
        half_width = int(real_width / 2)
        fill_width = width - 2 * half_width
        result = np.concat(
            [result[:half_width], np.ones(fill_width), result[-half_width:]], axis=0
        )
    assert len(result) == width
    return result
