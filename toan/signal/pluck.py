# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

# Control how aggressive the low-pass filter is
# Numbers closer to 0.5 will filter the most
SPLIT_A = 0.57


# Generate a pluck using a Karplus-Strong filter over random noise
def generate_pluck(
    sample_rate: int, frequency: float, duration: float, decay: float
) -> np.ndarray:
    out_sample_count = int(duration * sample_rate)
    result = np.zeros(out_sample_count)

    buffer_width = int(sample_rate / frequency)

    buffer = np.random.uniform(-1.0, 1.0, buffer_width)
    buffer = buffer / np.max(np.abs(buffer))

    previous: float = 0.0

    for i in range(out_sample_count):
        buffer_index = i % buffer_width
        value = decay * (buffer[buffer_index] * SPLIT_A + previous * (1.0 - SPLIT_A))
        buffer[buffer_index] = value
        result[i] = value
        previous = value

    return result
