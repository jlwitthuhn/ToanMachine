# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

from toan.training import LossFunction


@dataclass
class TrainingStageConfig:
    steps_warmup: int = 100
    steps_main: int = 1100
    test_interval: int = 50
    # If batch_size is 0, batch_size_list will be used
    batch_size: int = 0
    batch_size_list: list[tuple[float, int]] = field(
        default_factory=lambda: [(0.0, 24), (0.4, 40), (0.8, 56)]
    )
    input_sample_width: int = 8192 + 4096
    learn_rate_hi: float = 3.0e-3
    learn_rate_lo: float = 5.0e-3
    weight_decay: float = 1.1e-2
    loss_fn: LossFunction = LossFunction.MSE
    adam_betas: list[float] = field(default_factory=lambda: [0.89, 0.98])

    def steps_total(self) -> int:
        return self.steps_warmup + self.steps_main


@dataclass
class TrainingConfig:
    stages: list[TrainingStageConfig] = field(
        default_factory=lambda: [TrainingStageConfig()]
    )
    rng_seed: int = 0x35

    def steps_total(self) -> int:
        total = 0
        for stage in self.stages:
            total += stage.steps_total()
        return total
