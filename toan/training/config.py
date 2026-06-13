# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

from toan.model.presets import ModelConfigPreset
from toan.training.loss import LossFunction


@dataclass
class TrainingStageConfig:
    steps_warmup: int = 100
    steps_main: int = 1300
    test_interval: int = 50
    # If batch_size is 0, batch_size_list will be used
    batch_size: int = 0
    batch_size_list: list[tuple[float, int]] = field(
        default_factory=lambda: [(0.0, 32), (0.5, 48), (0.9, 64)]
    )
    input_sample_width: int = 1024 * 16
    learn_rate_hi: float = 3.5e-3
    learn_rate_lo: float = 3.0e-3
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
    final_output_steps: int = 100
    final_output_num: int = 3

    def steps_total(self) -> int:
        total = 0
        for stage in self.stages:
            total += stage.steps_total()
        return total


def _get_a1_training_config() -> TrainingConfig:
    return TrainingConfig()


def _get_a1_xstd_training_config() -> TrainingConfig:
    config = TrainingConfig()
    the_stage = config.stages[0]
    the_stage.input_sample_width = 1024 * 24
    the_stage.steps_main = 1000
    the_stage.learn_rate_hi = 5.0e-3
    the_stage.learn_rate_lo = 2.0e-3
    return config


def _get_a1_rev_ystd_training_config() -> TrainingConfig:
    config = _get_a1_xstd_training_config()
    the_stage = config.stages[0]
    the_stage.input_sample_width = 1024 * 36
    the_stage.steps_main = 700
    the_stage.learn_rate_hi = 6.0e-3
    the_stage.learn_rate_lo = 1.0e-3
    return config


def _get_a2_training_config() -> TrainingConfig:
    config = TrainingConfig()
    the_stage = config.stages[0]
    the_stage.batch_size_list = [(0.0, 24), (0.50, 40), (0.70, 56)]
    the_stage.steps_warmup = 100
    the_stage.steps_main = 1750
    the_stage.test_interval = 100
    the_stage.learn_rate_hi = 6.0e-3
    the_stage.learn_rate_lo = 1.5e-3
    return config


def get_training_config_from_preset(selected_preset: ModelConfigPreset):
    match selected_preset:
        case (
            ModelConfigPreset.A1_NAM_STANDARD
            | ModelConfigPreset.A1_NAM_LITE
            | ModelConfigPreset.A1_NAM_FEATHER
        ):
            return _get_a1_training_config()
        case ModelConfigPreset.A1_CUSTOM_XSTD:
            return _get_a1_xstd_training_config()
        case ModelConfigPreset.A1_CUSTOM_REVYSTD:
            return _get_a1_rev_ystd_training_config()
        case ModelConfigPreset.A2_NAM:
            return _get_a2_training_config()
        case _:
            raise NotImplementedError
