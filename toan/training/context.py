# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading

import numpy as np

from toan.model.metadata import ModelMetadata
from toan.model.nam_a1_wavenet import NamA1WaveNet
from toan.model.nam_a1_wavenet_config import NamA1WaveNetConfig
from toan.model.nam_a2_wavenet_config import NamA2WaveNetConfig
from toan.training import TrainingStageSummary


class TrainingProgressContext:
    model_config: NamA1WaveNetConfig | NamA2WaveNetConfig | None = None
    metadata: ModelMetadata | None = None
    sample_rate: int | None = None

    signal_dry_test: np.ndarray | None = None
    signal_wet_test: np.ndarray | None = None
    signal_dry_train: np.ndarray | None = None
    signal_wet_train: np.ndarray | None = None

    lock: threading.Lock = threading.Lock()
    iters_done: int = 0
    iters_total: int = 1
    loss_train: float | None = None
    loss_test: float | None = None

    model: NamA1WaveNet | None = None
    summary: TrainingStageSummary | None = None

    quit: bool = False
