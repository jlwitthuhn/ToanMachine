# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from typing import Any

import numpy as np
from mlx import core as mx
from mlx import nn as nn
from mlx import optimizers as optimizers

from toan.model.nam_a1_wavenet_config import NamA1WaveNetConfig
from toan.model.nam_a1_wavenet_mlx import NamA1WaveNetMlx
from toan.model.nam_a1_wavenet_mlx_functional import NamA1WavenetMlxFunctional
from toan.model.nam_a2_wavenet_config import NamA2WaveNetConfig
from toan.model.nam_a2_wavenet_mlx import NamA2WaveNetMlx
from toan.training import TrainingStageSummary
from toan.training.config import TrainingConfig, TrainingStageConfig
from toan.training.context import TrainingProgressContext
from toan.training.data_loader_mlx import TrainingDataLoaderMlx
from toan.training.loss_mlx import LossFunction, calculate_loss_mlx


def run_training_loop_mlx(context: TrainingProgressContext, config: TrainingConfig):
    assert len(config.stages) > 0
    if isinstance(context.model_config, NamA1WaveNetConfig):
        model = NamA1WaveNetMlx(
            context.model_config,
            context.metadata,
            context.sample_rate,
            rng_seed=config.rng_seed,
        )
    elif isinstance(context.model_config, NamA2WaveNetConfig):
        model = NamA2WaveNetMlx(
            context.model_config,
            context.metadata,
            context.sample_rate,
            rng_seed=config.rng_seed,
        )
    else:
        assert False
    print(model.parameter_count)
    mx.eval(model.parameters())

    np_rng_state = np.random.get_state()
    np.random.seed(config.rng_seed)

    def get_batch_size(stage_cfg: TrainingStageConfig, iter: int) -> int:
        if stage_cfg.batch_size > 0:
            return stage_cfg.batch_size
        assert len(stage_cfg.batch_size_list) > 0
        progress = iter / stage_cfg.steps_total()
        result = 1
        for threshold, count in stage_cfg.batch_size_list:
            if progress >= threshold:
                result = count
            else:
                break
        return result

    def get_learn_rate(stage_cfg: TrainingStageConfig) -> Any:
        decay_lr = optimizers.cosine_decay(
            stage_cfg.learn_rate_hi,
            stage_cfg.steps_main,
            stage_cfg.learn_rate_lo,
        )
        if stage_config.steps_warmup > 0:
            warmup_lr = optimizers.linear_schedule(
                stage_cfg.learn_rate_hi / 100.0,
                stage_cfg.learn_rate_hi,
                stage_cfg.steps_warmup,
            )
            return optimizers.join_schedules(
                [warmup_lr, decay_lr], [stage_config.steps_warmup]
            )
        else:
            return decay_lr

    def get_test_data() -> tuple[mx.array, mx.array]:
        input = mx.array(context.signal_dry_test).reshape((1, -1))
        output = mx.array(context.signal_wet_test)[model.receptive_field - 1 :].reshape(
            (1, -1)
        )
        return input, output

    def measure_test_loss(func: LossFunction) -> float:
        model.train(False)
        test_in, test_out = get_test_data()
        model_out = model(test_in)
        return calculate_loss_mlx(func, model_out, test_out).item()

    with context.lock:
        context.iters_done = 0
        context.iters_total = config.steps_total()

    for stage_config in config.stages:
        summary = TrainingStageSummary(
            test_interval=stage_config.test_interval,
            warmup_length=stage_config.steps_warmup,
        )
        context.summary = summary

        learn_rate = get_learn_rate(stage_config)

        data_loader = TrainingDataLoaderMlx(
            context.signal_dry_train,
            context.signal_wet_train,
            stage_config.input_sample_width,
            model.receptive_field,
        )

        # Make a loss function in the shape that MLX expects
        def loss_fn(model_in, inputs: mx.array, targets: mx.array):
            if isinstance(model_in, nn.Module):
                outputs = model_in(inputs)
            elif isinstance(model_in, dict):
                if isinstance(context.model_config, NamA1WaveNetConfig):
                    outputs = NamA1WavenetMlxFunctional.forward_model(
                        context.model_config, model_in, inputs
                    )
                else:
                    raise NotImplementedError
            else:
                raise ValueError("`model_in` must be a `nn.Module` or `dict`")
            return calculate_loss_mlx(stage_config.loss_fn, outputs, targets)

        optimizer = optimizers.AdamW(
            learning_rate=learn_rate,
            betas=stage_config.adam_betas,
            weight_decay=stage_config.weight_decay,
        )
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        def do_step(batch_in_step: mx.array, batch_out_step: mx.array) -> mx.array:
            loss, grads = loss_and_grad_fn(model, batch_in_step, batch_out_step)
            optimizer.update(model, grads)
            return loss

        if config.compile_model:
            state = [model.state, optimizer.state]
            do_step = mx.compile(do_step, inputs=state, outputs=state)

        train_loss_buffer = mx.ones(12)
        train_loss_buffer_sz = len(train_loss_buffer)

        for i in range(stage_config.steps_total()):
            if context.quit:
                return
            model.train(True)
            this_batch_size = get_batch_size(stage_config, i)
            batch_in, batch_out = data_loader.make_batch(this_batch_size)

            loss = do_step(batch_in, batch_out)

            train_loss_buffer[i % train_loss_buffer_sz] = loss
            mx.eval(model.parameters())

            summary.losses_train.append(loss.item())

            with context.lock:
                context.iters_done = i
                context.loss_train = train_loss_buffer.mean().item()

                if (
                    context.signal_dry_test is not None
                    and stage_config.test_interval > 0
                ):
                    if i % stage_config.test_interval == stage_config.test_interval - 1:
                        loss = measure_test_loss(stage_config.loss_fn)
                        summary.losses_test.append(loss)
                        context.loss_test = loss

    if context.signal_dry_test is not None:
        for this_loss in LossFunction:
            context.metadata.loss_test[this_loss.name] = measure_test_loss(this_loss)
        # Overall loss will use the last stage loss fn
        loss_fn = config.stages[-1].loss_fn
        context.loss_test = context.metadata.loss_test[loss_fn.name]

    context.model = model

    np.random.set_state(np_rng_state)
