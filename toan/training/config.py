# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

from toan.training import LossFunction


@dataclass
class TrainingConfig:
    num_steps: int = 500
    test_interval: int = 25
    warmup_steps: int = 60
    # If batch_size is 0, batch_size_list will be used
    batch_size: int = 32
    batch_size_list: list[tuple[float, int]] = field(default_factory=list)
    input_sample_width: int = 8192 + 4096
    learn_rate_hi: float = 5.0e-3
    learn_rate_lo: float = 3.0e-3
    weight_decay: float = 7.5e-3
    loss_fn: LossFunction = LossFunction.RMSE
