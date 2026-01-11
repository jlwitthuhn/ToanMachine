# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


# Generate a pluck using a Karplus-Strong filter over random noise
def generate_pluck(
    sample_rate: int, frequency: float, duration: float, decay: float
) -> np.ndarray:
    out_sample_count = int(duration * sample_rate)
    result = np.zeros(out_sample_count)

    buffer_width = int(sample_rate / frequency)
    buffer = np.random.choice([-1.0, 1.0], size=buffer_width)

    previous: float = 0.0

    for i in range(out_sample_count):
        buffer_index = i % buffer_width
        value = decay * (buffer[buffer_index] + previous) / 2.0
        buffer[buffer_index] = value
        result[i] = value
        previous = value

    return result
