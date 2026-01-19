# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from PySide6 import QtCore, QtWidgets

from toan.gui.train import TrainingContext
from toan.model.nam_wavenet import NamWaveNet
from toan.training import TrainingSummary
from toan.training.loader import TrainingDataLoader

TRAIN_TEXT = [
    "Your model is now training. After training has finished you will be asked to choose a location for the NAM file."
]


class TrainTrainPage(QtWidgets.QWizardPage):
    context: TrainingContext
    refresh_timer: QtCore.QTimer

    progress_bar: QtWidgets.QProgressBar
    progress_desc: QtWidgets.QLabel

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context
        self.context.progress_lock = threading.Lock()
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.setInterval(100)
        self.refresh_timer.setSingleShot(False)
        self.refresh_timer.timeout.connect(self.refresh_page)
        self.refresh_timer.start()

        self.setCommitPage(True)
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

        self.progress_bar = QtWidgets.QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.progress_desc = QtWidgets.QLabel("Loss estimate:", self)
        layout.addWidget(self.progress_desc)

    def initializePage(self):
        def thread_func():
            _run_training(self.context, _TrainingConfig())

        threading.Thread(target=thread_func).start()

    def isComplete(self) -> bool:
        return self.context.model is not None

    def validatePage(self) -> bool:
        return self.context.model is not None

    def refresh_page(self):
        with self.context.progress_lock:
            self.progress_bar.setMaximum(self.context.progress_iters_total)
            self.progress_bar.setValue(self.context.progress_iters_done)
            self.progress_bar.repaint()
            self.progress_desc.setText(
                f"Loss estimate: {self.context.progress_loss:.4f}"
            )
            if self.context.model is not None:
                self.refresh_timer.stop()
                self.completeChanged.emit()


@dataclass
class _TrainingConfig:
    num_steps: int = 600
    warmup_steps: int = 50
    batch_size: int = 64
    learn_rate_hi: float = 8.0e-4
    learn_rate_lo: float = 1.5e-4
    weight_decay: float = 7.5e-3


def _run_training(context: TrainingContext, config: _TrainingConfig):
    mx.random.seed(0o35)
    model = NamWaveNet(
        context.model_config, context.loaded_metadata, context.sample_rate
    )
    assert model is not None
    mx.eval(model.parameters())

    summary = TrainingSummary()

    normal_steps = config.num_steps - config.warmup_steps
    decay_lr = optimizers.cosine_decay(
        config.learn_rate_hi, normal_steps, config.learn_rate_lo
    )
    if config.warmup_steps > 0:
        warmup_lr = optimizers.linear_schedule(
            config.learn_rate_hi / 100.0, config.learn_rate_hi, config.warmup_steps
        )
        learn_rate = optimizers.join_schedules(
            [warmup_lr, decay_lr], [config.warmup_steps]
        )
    else:
        learn_rate = decay_lr

    data_loader = TrainingDataLoader(
        context.signal_dry, context.signal_wet, 8192 + 2048, model.receptive_field
    )

    loss_and_grad_fn = nn.value_and_grad(model, NamWaveNet.loss_fn)
    optimizer = optimizers.AdamW(
        learning_rate=learn_rate, weight_decay=config.weight_decay
    )

    with context.progress_lock:
        context.progress_iters_done = 0
        context.progress_iters_total = config.num_steps
        context.progress_loss = 1.0

    loss_buffer = mx.ones(16)
    loss_buffer_sz = len(loss_buffer)

    for i in range(config.num_steps):
        model.train(True)
        batch_in, batch_out = data_loader.make_batch(config.batch_size)
        loss, grads = loss_and_grad_fn(model, batch_in, batch_out)
        optimizer.update(model, grads)
        loss_buffer[i % loss_buffer_sz] = loss
        mx.eval(model.parameters())

        summary.losses.append(loss.item())

        with context.progress_lock:
            context.progress_iters_done = i
            if i > 0 and i % 5 == 0:
                context.progress_loss = loss_buffer.mean().item()

    context.model = model
    context.training_summary = summary
