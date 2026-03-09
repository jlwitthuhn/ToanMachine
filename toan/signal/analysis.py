# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalClickDetails:
    first_click: int
    delta: int


def find_dry_clicks(signal: np.ndarray) -> SignalClickDetails | None:
    assert signal.ndim == 1
    eps = 0.001

    clicks: list[int] = []
    for i in range(len(signal)):
        if signal[i] > eps:
            clicks.append(i)

    if len(clicks) != 2:
        return None

    return SignalClickDetails(first_click=clicks[0], delta=clicks[1] - clicks[0])
