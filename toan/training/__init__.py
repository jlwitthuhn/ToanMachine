# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import enum
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class LossFunction(Enum):
    RMSE = enum.auto()
    ESR = enum.auto()


@dataclass
class TrainingSummary:
    losses_train: list[float] = field(default_factory=list)
    losses_test: list[float] = field(default_factory=list)
    test_interval: int = 100

    def generate_loss_graph(self, smooth_factor: int) -> Figure:
        fig, ax = plt.subplots()
        ax.set_title("Training Loss")

        np_losses_train = np.array(self.losses_train)
        smooth_losses_train = np.convolve(
            np_losses_train, np.ones((smooth_factor,)) / smooth_factor, mode="valid"
        ).tolist()
        eval_points_train = np.arange(len(smooth_losses_train)) + smooth_factor // 2
        ax.plot(eval_points_train, smooth_losses_train, label="train")

        if len(self.losses_test) > 0:
            eval_points_test = (
                np.arange(len(self.losses_test)) + 1
            ) * self.test_interval
            ax.plot(eval_points_test, self.losses_test, label="test")

        ax.grid(True)
        ax.legend()

        return fig
