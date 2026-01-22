# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading

from toan.model.nam_wavenet import NamWaveNet
from toan.training import TrainingSummary


class TrainingProgressContext:
    lock: threading.Lock = threading.Lock()
    iters_done: int = 0
    iters_total: int = 1
    loss_train: float = 1.0
    loss_test: float = 1.0

    model: NamWaveNet | None = None
    summary: TrainingSummary | None = None

    quit: bool = False
