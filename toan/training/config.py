# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

from toan.model.presets import ModelConfigPreset
from toan.training.loss import LossFunction


@dataclass
class TrainingStageConfig:
    steps_warmup: int = 100
    steps_main: int = 1750
    test_interval: int = 100
    # If batch_size is 0, batch_size_list will be used
    batch_size: int = 0
    batch_size_list: list[tuple[float, int]] = field(
        default_factory=lambda: [(0.0, 24), (0.50, 40), (0.70, 56)]
    )
    input_sample_width: int = 1024 * 16
    learn_rate_hi: float = 6.0e-3
    learn_rate_lo: float = 1.5e-3
    weight_decay: float = 1.0e-2
    loss_fn: LossFunction = LossFunction.NamOriginal
    adam_betas: list[float] = field(default_factory=lambda: [0.89, 0.98])

    def steps_total(self) -> int:
        return self.steps_warmup + self.steps_main


@dataclass
class TrainingConfig:
    stages: list[TrainingStageConfig] = field(
        default_factory=lambda: [TrainingStageConfig()]
    )
    rng_seed: int = 0x35
    compile_model: bool = False
    final_output_steps: int = 120
    final_output_num: int = 4

    def steps_total(self) -> int:
        total = 0
        for stage in self.stages:
            total += stage.steps_total()
        return total


def _get_a2_training_config() -> TrainingConfig:
    return TrainingConfig()


def get_training_config_from_preset(selected_preset: ModelConfigPreset):
    match selected_preset:
        case ModelConfigPreset.A2_NAM:
            return _get_a2_training_config()
        case _:
            raise NotImplementedError
