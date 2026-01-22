# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from toan.model.metadata import ModelMetadata
from toan.model.nam_wavenet_config import NamWaveNetConfig
from toan.training.context import TrainingProgressContext


class TrainingGuiContext:
    input_path: str
    loaded_metadata: ModelMetadata | None = None
    sample_rate: int = 0
    model_config: NamWaveNetConfig | None = None

    signal_dry: np.ndarray | None = None
    signal_wet: np.ndarray | None = None
    signal_dry_test: np.ndarray | None = None
    signal_wet_test: np.ndarray | None = None

    progress_context: TrainingProgressContext = TrainingProgressContext()
