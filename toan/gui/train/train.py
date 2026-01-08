# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from PySide6 import QtWidgets

from toan.gui.train import TrainingContext
from toan.model.nam_wavenet import NamWaveNet
from toan.model.nam_wavenet_config import default_wavenet_config

TRAIN_TEXT = [
    "Your model is now training. After training has finished you will be asked to choose a location for the NAM file."
]


class TrainTrainPage(QtWidgets.QWizardPage):
    context: TrainingContext

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Train")
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("\n\n".join(TRAIN_TEXT), self)
        label.setWordWrap(True)
        layout.addWidget(label)

        hline = QtWidgets.QFrame(self)
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(hline)

        label_progress = QtWidgets.QLabel("Progress:", self)
        layout.addWidget(label_progress)

        self.bar_progress = QtWidgets.QProgressBar(self)
        layout.addWidget(self.bar_progress)

    def initializePage(self):
        _run_training(self.context, _TrainingConfig())


@dataclass
class _TrainingConfig:
    num_steps: int = 500
    warmup_steps: int = 100
    batch_size: int = 48
    learn_rate_hi: float = 2.0e-4


def _generate_batch(
    context: TrainingContext, receptive_field: int, batch_count: int, input_samples: int
) -> tuple[mx.array, mx.array]:
    input_list: list[mx.array] = []
    output_list: list[mx.array] = []
    dry_sample_begin: int = 0
    dry_sample_end: int = len(context.signal_dry) - input_samples - 1
    for i in range(batch_count):
        input_begin = mx.random.randint(dry_sample_begin, dry_sample_end).item()
        input_end = input_begin + input_samples
        wet_begin = input_begin + receptive_field - 1
        assert wet_begin < input_end
        input_list.append(mx.array(context.signal_dry[input_begin:input_end]))
        output_list.append(mx.array(context.signal_wet[wet_begin:input_end]))
    return mx.array(input_list), mx.array(output_list)


def _run_training(context: TrainingContext, config: _TrainingConfig):
    model_config = default_wavenet_config()
    model = NamWaveNet(model_config)
    assert model is not None
    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, NamWaveNet.loss_fn)
    optimizer = optimizers.AdamW(learning_rate=config.learn_rate_hi)

    print("Beginning training...")
    print(f"input width: {model.receptive_field}")
    print(f"params: {model.parameter_count}")

    context.progress_iters_done = 0
    context.progress_iters_total = config.num_steps
    context.progress_loss = 1.0

    loss_buffer = mx.ones(18)
    loss_buffer_sz = len(loss_buffer)

    for i in range(config.num_steps):
        model.train(True)
        batch_in, batch_out = _generate_batch(
            context, model.receptive_field, config.batch_size, 8192
        )
        loss, grads = loss_and_grad_fn(model, batch_in, batch_out)
        optimizer.update(model, grads)
        loss_buffer[i % loss_buffer_sz] = loss
        if i > 0 and i % 25 == 0:
            print(f"{i} loss: {loss_buffer.mean()}")
