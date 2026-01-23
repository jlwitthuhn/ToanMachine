# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass

from toan.training import LossFunction


@dataclass
class TrainingConfig:
    num_steps: int = 600
    test_interval: int = 25
    warmup_steps: int = 50
    batch_size: int = 64
    input_sample_width: int = 8192 + 2048
    learn_rate_hi: float = 8.0e-4
    learn_rate_lo: float = 1.5e-4
    weight_decay: float = 7.5e-3
    loss_fn: LossFunction = LossFunction.RMSE
