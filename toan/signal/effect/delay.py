# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


def effect_delay(signal: np.ndarray, samples: int, gain: float) -> None:
    def get_sample(index: int) -> float:
        if index < 0:
            return 0.0
        return signal[index]

    # This must be applied backwards to work in-place
    # Otherwise the delayed signal will itself be repeated
    # Maybe make that a toggleable option later
    for i in range(len(signal)):
        this_index = len(signal) - 1 - i
        signal[this_index] = signal[this_index] + gain * get_sample(
            this_index - samples
        )
