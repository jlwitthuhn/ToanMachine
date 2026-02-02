# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from mlx import core as mx
from mlx import nn as nn
from mlx import optimizers as optimizers

from toan.model.nam_wavenet import NamWaveNet
from toan.training import LossFunction, TrainingSummary
from toan.training.config import TrainingConfig
from toan.training.context import TrainingProgressContext
from toan.training.data_loader import TrainingDataLoader


def run_training_loop(context: TrainingProgressContext, config: TrainingConfig):
    mx.random.seed(0o35)
    model = NamWaveNet(context.model_config, context.metadata, context.sample_rate)
    assert model is not None
    mx.eval(model.parameters())

    summary = TrainingSummary(test_interval=config.test_interval)
    context.summary = summary

    decay_lr = optimizers.cosine_decay(
        config.learn_rate_hi, config.steps_main, config.learn_rate_lo
    )
    if config.steps_warmup > 0:
        warmup_lr = optimizers.linear_schedule(
            config.learn_rate_hi / 100.0, config.learn_rate_hi, config.steps_warmup
        )
        learn_rate = optimizers.join_schedules(
            [warmup_lr, decay_lr], [config.steps_warmup]
        )
    else:
        learn_rate = decay_lr

    data_loader = TrainingDataLoader(
        context.signal_dry_train,
        context.signal_wet_train,
        config.input_sample_width,
        model.receptive_field,
    )

    # Make a loss function in the shape that MLX expects
    def loss_fn(model_in, inputs: mx.array, outputs: mx.array):
        return model_in.loss(inputs, outputs, config.loss_fn)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optimizers.AdamW(
        learning_rate=learn_rate, weight_decay=config.weight_decay
    )

    with context.lock:
        context.iters_done = 0
        context.iters_total = config.steps_total()
        context.loss_train = 1.0
        context.loss_test = 1.0

    train_loss_buffer = mx.ones(16)
    train_loss_buffer_sz = len(train_loss_buffer)

    def get_test_data() -> tuple[mx.array, mx.array]:
        input = mx.array(context.signal_dry_test).reshape((1, -1))
        output = mx.array(context.signal_wet_test)[model.receptive_field - 1 :].reshape(
            (1, -1)
        )
        return input, output

    def measure_test_loss(func: LossFunction) -> float:
        model.train(False)
        test_in, test_out = get_test_data()
        return model.loss(test_in, test_out, func).item()

    def get_batch_size(iter: int) -> int:
        if config.batch_size > 0:
            return config.batch_size
        assert len(config.batch_size_list) > 0
        progress = iter / config.steps_total()
        result = 1
        for threshold, count in config.batch_size_list:
            if progress >= threshold:
                result = count
            else:
                break
        return result

    for i in range(config.steps_total()):
        if context.quit:
            return
        model.train(True)
        this_batch_size = get_batch_size(i)
        batch_in, batch_out = data_loader.make_batch(this_batch_size)
        loss, grads = loss_and_grad_fn(model, batch_in, batch_out)
        optimizer.update(model, grads)
        train_loss_buffer[i % train_loss_buffer_sz] = loss
        mx.eval(model.parameters())

        summary.losses_train.append(loss.item())

        with context.lock:
            context.iters_done = i
            context.loss_train = train_loss_buffer.mean().item()

            if context.signal_dry_test is not None:
                if i % config.test_interval == config.test_interval - 1:
                    loss = measure_test_loss(config.loss_fn)
                    summary.losses_test.append(loss)
                    context.loss_test = loss

    context.metadata.loss_test_rmse = measure_test_loss(LossFunction.RMSE)
    context.metadata.loss_test_esr = measure_test_loss(LossFunction.ESR)
    context.model = model
