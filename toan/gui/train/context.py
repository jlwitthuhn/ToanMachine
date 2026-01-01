# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np


class TrainingContext:
    input_path: str
    signal_dry: np.ndarray | None = None
    signal_wet: np.ndarray | None = None
