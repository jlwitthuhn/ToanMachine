# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def effect_delay(signal: np.ndarray, samples: int, gain: float, feedback: bool) -> None:
    def get_sample(index: int) -> float:
        if index < 0:
            return 0.0
        return signal[index]

    for i in range(len(signal)):
        this_index = i if feedback else len(signal) - i - 1
        signal[this_index] = signal[this_index] + gain * get_sample(
            this_index - samples
        )
