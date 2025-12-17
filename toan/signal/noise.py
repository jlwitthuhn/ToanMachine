# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def generate_white_noise(samples: int) -> np.ndarray:
    return np.random.normal(0, 1, samples)
