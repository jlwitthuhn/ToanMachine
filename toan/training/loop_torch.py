# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

import numpy as np
import torch
from torch import optim

from toan.model.metadata import ModelA1Metadata, ModelA2Metadata
from toan.model.nam_a1_wavenet_config import NamA1WaveNetConfig
from toan.model.nam_a1_wavenet_torch import NamA1WaveNetTorch
from toan.model.nam_a2_wavenet_config import NamA2WaveNetContainerConfig
from toan.model.nam_a2_wavenet_torch import NamA2WaveNetTorch
from toan.training import TrainingStageSummary
from toan.training.config import TrainingConfig, TrainingStageConfig
from toan.training.context import TrainingProgressContext
from toan.training.data_loader import TrainingDataLoaderMlx
from toan.training.loss import LossFunction
from toan.training.loss_torch import calculate_loss_torch


def _calculate_submodel_losses(
    loss_fn: LossFunction,
    model_output: torch.Tensor,
    target: torch.Tensor,
) -> list[torch.Tensor]:
    # A2 models stack one prediction per submodel as (num_submodels, batch, length),
    # so report each submodel's own loss. A1 models produce a single output.
    if model_output.ndim == 3:
        return [
            calculate_loss_torch(loss_fn, model_output[i], target)
            for i in range(model_output.shape[0])
        ]
    return [calculate_loss_torch(loss_fn, model_output, target)]


def _calculate_model_loss(
    loss_fn: LossFunction,
    model_output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # Train every submodel jointly by summing each submodel's own loss
    submodel_losses = _calculate_submodel_losses(loss_fn, model_output, target)
    total = submodel_losses[0]
    for submodel_loss in submodel_losses[1:]:
        total = total + submodel_loss
    return total


def run_training_loop_torch(context: TrainingProgressContext, config: TrainingConfig):
    assert len(config.stages) > 0
    assert context.metadata is not None
    assert context.sample_rate is not None
    if isinstance(context.model_config, NamA1WaveNetConfig):
        model = NamA1WaveNetTorch(
            context.model_config,
            ModelA1Metadata.from_generic(context.metadata),
            context.sample_rate,
            rng_seed=config.rng_seed,
        )
    elif isinstance(context.model_config, NamA2WaveNetContainerConfig):
        model = NamA2WaveNetTorch(
            context.model_config,
            ModelA2Metadata.from_generic(context.metadata),
            context.sample_rate,
            rng_seed=config.rng_seed,
        )
    else:
        raise NotImplementedError("Only NAM A1 and A2 are supported.")
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

    def measure_test_loss_per_submodel(
        func: LossFunction,
    ) -> tuple[float, list[float]]:
        model.train(False)
        test_in, test_out = get_test_data()
        with torch.no_grad():
            model_out = model(test_in)
            per_submodel = [
                loss.item()
                for loss in _calculate_submodel_losses(func, model_out, test_out)
            ]
        return sum(per_submodel), per_submodel

    def measure_test_loss(func: LossFunction) -> float:
        return measure_test_loss_per_submodel(func)[0]

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
            loss = _calculate_model_loss(stage_config.loss_fn, outputs, batch_out_step)
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
        num_submodels = (
            len(model.submodels) if isinstance(model, NamA2WaveNetTorch) else 1
        )
        submodel_loss_tests: list[dict[str, float]] = [{} for _ in range(num_submodels)]
        for this_loss in LossFunction:
            full_loss, per_submodel = measure_test_loss_per_submodel(this_loss)
            context.metadata.loss_test[this_loss.name] = full_loss
            for submodel_dict, submodel_loss in zip(submodel_loss_tests, per_submodel):
                submodel_dict[this_loss.name] = submodel_loss

        model.metadata.loss_test = dict(context.metadata.loss_test)
        if isinstance(model, NamA2WaveNetTorch):
            for submodel_metadata, submodel_loss_test in zip(
                model.submodel_metadata, submodel_loss_tests
            ):
                submodel_metadata.loss_test = submodel_loss_test

        # Overall loss will use the last stage loss fn
        loss_fn = config.stages[-1].loss_fn
        context.loss_test = context.metadata.loss_test[loss_fn.name]

    if isinstance(model, NamA2WaveNetTorch):
        model.populate_loudness_and_gain_metadata()

    context.model = model

    np.random.set_state(np_rng_state)
