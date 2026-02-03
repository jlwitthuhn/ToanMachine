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
    MSE = enum.auto()
    RMSE = enum.auto()
    ESR = enum.auto()


@dataclass
class TrainingStageSummary:
    losses_train: list[float] = field(default_factory=list)
    losses_test: list[float] = field(default_factory=list)
    test_interval: int = 100
    warmup_length: int = 0

    def generate_loss_graph(self, smooth_factor: int) -> Figure:
        fig, ax = plt.subplots()
        ax.set_title("Training Loss")

        def clip_warmup(
            losses: np.ndarray, points: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            assert len(losses.shape) == 1
            assert len(points.shape) == 1
            assert len(losses) == len(points)
            points_clipped = 0
            for point in points:
                if point < self.warmup_length:
                    points_clipped += 1
            return losses[points_clipped:], points[points_clipped:]

        np_losses_train = np.array(self.losses_train)
        smooth_losses_train = np.convolve(
            np_losses_train, np.ones((smooth_factor,)) / smooth_factor, mode="valid"
        )
        eval_points_train = np.arange(len(smooth_losses_train)) + smooth_factor // 2
        plot_losses_train, plot_points_train = clip_warmup(
            smooth_losses_train, eval_points_train
        )
        ax.plot(plot_points_train, plot_losses_train, label="train")

        clip_warmup(smooth_losses_train, eval_points_train)

        if len(self.losses_test) > 0:
            eval_points_test = (
                np.arange(len(self.losses_test)) + 1
            ) * self.test_interval
            plot_losses_test, plot_points_test = clip_warmup(
                np.array(self.losses_test), eval_points_test
            )
            ax.plot(plot_points_test, plot_losses_test, label="test")

        ax.grid(True)
        ax.legend()

        return fig
