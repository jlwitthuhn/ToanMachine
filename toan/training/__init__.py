# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


@dataclass
class TrainingSummary:
    losses: list[float] = field(default_factory=list)

    def generate_loss_graph(self, smooth_factor: int) -> Figure:
        fig, ax = plt.subplots()
        ax.set_title("Training Loss")

        np_losses = np.array(self.losses)
        smooth_losses = np.convolve(
            np_losses, np.ones((smooth_factor,)) / smooth_factor, mode="valid"
        ).tolist()
        eval_points = np.arange(len(smooth_losses)) + smooth_factor // 2

        ax.plot(eval_points, smooth_losses, label="loss")
        ax.grid(True)

        return fig
