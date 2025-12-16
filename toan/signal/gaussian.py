# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_gaussian_pulse(width: int) -> np.ndarray:
    sigma = width * 0.16
    n = np.arange(width)
    mu = (width - 1) / 2
    imp = np.exp(-0.5 * ((n - mu) / sigma) ** 2)
    return imp
