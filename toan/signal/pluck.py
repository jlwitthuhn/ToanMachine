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

    noise_width = int(sample_rate / frequency)
    noise_buffer = np.random.uniform(-1.0, 1.0, noise_width)

    for i in range(out_sample_count):
        next_sample = decay * (0.65 * noise_buffer[0] + 0.35 * noise_buffer[1])
        noise_buffer = np.append(noise_buffer[1:], next_sample)
        result[i] = next_sample

    return result
