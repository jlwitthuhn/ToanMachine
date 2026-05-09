# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def effect_delay(
    signal: np.ndarray, samples: int, gain: float, feedback: bool
) -> np.ndarray:
    result = signal.copy()

    def get_sample(index: int) -> float:
        if index < 0:
            return 0.0
        return result[index]

    for i in range(len(result)):
        this_index = i if feedback else len(result) - i - 1
        result[this_index] = result[this_index] + gain * get_sample(
            this_index - samples
        )

    return result
