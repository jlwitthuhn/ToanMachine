# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading

import numpy as np

from toan.model.metadata import ModelMetadata
from toan.model.nam_wavenet import NamWaveNet
from toan.model.nam_wavenet_config import NamWaveNetConfig
from toan.training import TrainingSummary


class TrainingProgressContext:
    model_config: NamWaveNetConfig | None = None
    metadata: ModelMetadata | None = None
    sample_rate: int | None = None

    signal_dry_test: np.ndarray | None = None
    signal_wet_test: np.ndarray | None = None
    signal_dry_train: np.ndarray | None = None
    signal_wet_train: np.ndarray | None = None

    lock: threading.Lock = threading.Lock()
    iters_done: int = 0
    iters_total: int = 1
    loss_train: float = 1.0
    loss_test: float = 1.0

    model: NamWaveNet | None = None
    summary: TrainingSummary | None = None

    quit: bool = False
