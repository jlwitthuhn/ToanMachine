# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np
import torch
from torch import optim

from toan.model.nam_a1_wavenet_config import NamA1WaveNetConfig
from toan.model.nam_a1_wavenet_torch import NamA1WaveNetTorch
from toan.training import TrainingStageSummary
from toan.training.config import TrainingConfig, TrainingStageConfig
from toan.training.context import TrainingProgressContext
from toan.training.data_loader import TrainingDataLoaderMlx
from toan.training.loss import LossFunction
from toan.training.loss_torch import calculate_loss_torch


def run_training_loop_torch(context: TrainingProgressContext, config: TrainingConfig):
    assert len(config.stages) > 0
    assert context.metadata is not None
    assert context.sample_rate is not None
    if isinstance(context.model_config, NamA1WaveNetConfig):
        model = NamA1WaveNetTorch(
            context.model_config,
            context.metadata,
            context.sample_rate,
            rng_seed=config.rng_seed,
        )
    else:
        raise NotImplementedError("Only NAM A1 is supported.")
    device = torch.device("mps")
    model.to(device)

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

    def make_lr_lambda(stage_cfg: TrainingStageConfig):
        hi = stage_cfg.learn_rate_hi
        lo = stage_cfg.learn_rate_lo
        warmup = stage_cfg.steps_warmup
        main = stage_cfg.steps_main

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                start = hi / 100.0
                frac = step / warmup
                lr = start + (hi - start) * frac
            else:
                decay_step = step - warmup
                if decay_step >= main:
                    lr = lo
                else:
                    cos = 0.5 * (1.0 + math.cos(math.pi * decay_step / main))
                    lr = lo + (hi - lo) * cos
            return lr / hi

        return lr_lambda

    def get_test_data() -> tuple[torch.Tensor, torch.Tensor]:
        input = (
            torch.from_numpy(context.signal_dry_test.copy())
            .float()
            .reshape((1, -1))
            .to(device)
        )
        output = (
            torch.from_numpy(context.signal_wet_test.copy())
            .float()[model.receptive_field - 1 :]
            .reshape((1, -1))
            .to(device)
        )
        return input, output

    def measure_test_loss(func: LossFunction) -> float:
        model.train(False)
        test_in, test_out = get_test_data()
        with torch.no_grad():
            model_out = model(test_in)
            return calculate_loss_torch(func, model_out, test_out).item()

    with context.lock:
        context.iters_done = 0
        context.iters_total = config.steps_total()

    for stage_config in config.stages:
        summary = TrainingStageSummary(
            test_interval=stage_config.test_interval,
            warmup_length=stage_config.steps_warmup,
        )
        context.summary = summary

        data_loader = TrainingDataLoaderMlx(
            context.signal_dry_train,
            context.signal_wet_train,
            stage_config.input_sample_width,
            model.receptive_field,
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=stage_config.learn_rate_hi,
            betas=tuple(stage_config.adam_betas),
            weight_decay=stage_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_lr_lambda(stage_config)
        )

        def do_step(
            batch_in_step: torch.Tensor, batch_out_step: torch.Tensor
        ) -> torch.Tensor:
            optimizer.zero_grad()
            outputs = model(batch_in_step)
            loss = calculate_loss_torch(stage_config.loss_fn, outputs, batch_out_step)
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss

        if config.compile_model:
            raise NotImplementedError("Compilation is not supported for torch models")

        train_loss_buffer = torch.ones(12)
        train_loss_buffer_sz = train_loss_buffer.numel()

        for i in range(stage_config.steps_total()):
            if context.quit:
                return
            model.train(True)
            this_batch_size = get_batch_size(stage_config, i)
            batch_in_np, batch_out_np = data_loader.make_batch(this_batch_size)
            batch_in = torch.from_numpy(batch_in_np).float().to(device)
            batch_out = torch.from_numpy(batch_out_np).float().to(device)

            loss = do_step(batch_in, batch_out)

            train_loss_buffer[i % train_loss_buffer_sz] = loss.detach()

            summary.losses_train.append(loss.item())

            with context.lock:
                context.iters_done = i
                context.loss_train = train_loss_buffer.mean().item()

                if (
                    context.signal_dry_test is not None
                    and stage_config.test_interval > 0
                ):
                    if i % stage_config.test_interval == stage_config.test_interval - 1:
                        loss_test = measure_test_loss(stage_config.loss_fn)
                        summary.losses_test.append(loss_test)
                        context.loss_test = loss_test

    if context.signal_dry_test is not None:
        for this_loss in LossFunction:
            context.metadata.loss_test[this_loss.name] = measure_test_loss(this_loss)
        # Overall loss will use the last stage loss fn
        loss_fn = config.stages[-1].loss_fn
        context.loss_test = context.metadata.loss_test[loss_fn.name]

    context.model = model

    np.random.set_state(np_rng_state)
