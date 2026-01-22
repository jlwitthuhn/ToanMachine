# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import datetime
import threading
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from PySide6 import QtCore, QtWidgets

from toan.formatting import format_seconds_as_mmss
from toan.gui.train import TrainingContext
from toan.model.nam_wavenet import NamWaveNet
from toan.training import LossFunction, TrainingSummary
from toan.training.loader import TrainingDataLoader

TRAIN_TEXT = [
    "Your model is now training. After training has finished you will be asked to choose a location for the NAM file."
]


class TrainTrainPage(QtWidgets.QWizardPage):
    context: TrainingContext
    refresh_timer: QtCore.QTimer

    progress_bar: QtWidgets.QProgressBar
    progress_desc: QtWidgets.QLabel

    timestamp_begin: datetime.datetime | None = None
    timer_label: QtWidgets.QLabel

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context
        self.context.progress_lock = threading.Lock()
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.setInterval(150)
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

        self.progress_desc = QtWidgets.QLabel("Training loss:", self)
        layout.addWidget(self.progress_desc)

        self.timer_label = QtWidgets.QLabel(
            f"Time spent: {format_seconds_as_mmss(0)}", self
        )
        layout.addWidget(self.timer_label)

    def initializePage(self):
        def thread_func():
            _run_training(self.context, _TrainingConfig())

        self.context.quit_training = False
        self.timestamp_begin = datetime.datetime.now()
        threading.Thread(target=thread_func).start()

    def cleanupPage(self):
        self.refresh_timer.stop()
        self.context.quit_training = True

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
                f"Training loss: {self.context.progress_train_loss:.4f}"
            )
            if self.context.model is not None:
                self.refresh_timer.stop()
                self.completeChanged.emit()
        if self.timestamp_begin is not None:
            current_time = datetime.datetime.now()
            time_elapsed = current_time - self.timestamp_begin
            formatted_time = format_seconds_as_mmss(time_elapsed.total_seconds())
            self.timer_label.setText(f"Time spent: {formatted_time}")


@dataclass
class _TrainingConfig:
    num_steps: int = 600
    warmup_steps: int = 50
    batch_size: int = 64
    learn_rate_hi: float = 8.0e-4
    learn_rate_lo: float = 1.5e-4
    weight_decay: float = 7.5e-3
    loss_fn: LossFunction = LossFunction.RMSE


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

    def loss_fn(model_in, inputs: mx.array, outputs: mx.array):
        if config.loss_fn == LossFunction.RMSE:
            return NamWaveNet.loss_rmse(model_in, inputs, outputs)
        elif config.loss_fn == LossFunction.ESR:
            return NamWaveNet.loss_esr(model_in, inputs, outputs)
        assert False

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optimizers.AdamW(
        learning_rate=learn_rate, weight_decay=config.weight_decay
    )

    with context.progress_lock:
        context.progress_iters_done = 0
        context.progress_iters_total = config.num_steps
        context.progress_train_loss = 1.0

    train_loss_buffer = mx.ones(16)
    train_loss_buffer_sz = len(train_loss_buffer)

    def get_test_data() -> tuple[mx.array, mx.array]:
        input = mx.array(context.signal_dry_test).reshape((1, -1))
        output = mx.array(context.signal_wet_test)[model.receptive_field - 1 :].reshape(
            (1, -1)
        )
        return input, output

    for i in range(config.num_steps):
        if context.quit_training:
            return
        model.train(True)
        batch_in, batch_out = data_loader.make_batch(config.batch_size)
        loss, grads = loss_and_grad_fn(model, batch_in, batch_out)
        optimizer.update(model, grads)
        train_loss_buffer[i % train_loss_buffer_sz] = loss
        mx.eval(model.parameters())

        summary.losses_train.append(loss.item())

        with context.progress_lock:
            context.progress_iters_done = i
            context.progress_train_loss = train_loss_buffer.mean().item()

        TEST_INTERVAL = 25

        if context.signal_dry_test is not None:
            if i % TEST_INTERVAL == TEST_INTERVAL - 1:
                model.train(False)
                test_in, test_out = get_test_data()
                loss = model.loss_rmse(test_in, test_out).item()
                summary.losses_test.append(loss)

    context.model = model
    context.training_summary = summary
