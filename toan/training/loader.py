# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import mlx.core as mx
import numpy as np


class TrainingDataLoader:
    signal_dry: np.ndarray
    signal_wet: np.ndarray

    dry_width: int
    wet_width: int
    receptive_field: int

    dry_begin_points: list[int]

    def __init__(
        self,
        signal_dry: np.ndarray,
        signal_wet: np.ndarray,
        dry_width: int,
        receptive_field: int,
    ):
        assert len(signal_dry) == len(signal_wet)
        self.signal_dry = signal_dry
        self.signal_wet = signal_wet
        self.dry_width = dry_width
        self.wet_width = dry_width - receptive_field + 1
        self.receptive_field = receptive_field
        self.dry_begin_points = []

        def dry_begin_from_wet_begin(wet_begin: int) -> int:
            delta = self.dry_width - self.wet_width
            result = wet_begin - delta
            assert result >= 0
            return result

        # Scan forward from the first valid wet sample
        wet_begin_first = self.dry_width - self.wet_width
        this_wet_begin = wet_begin_first
        while this_wet_begin < len(signal_wet):
            maybe = this_wet_begin + self.wet_width
            if maybe < len(signal_wet):
                dry_begin = dry_begin_from_wet_begin(this_wet_begin)
                self.dry_begin_points.append(dry_begin)
            this_wet_begin = maybe

        # TODO: Same thing but scan backwards from the end, this will probably be a different offset

    def make_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        input_list: list[mx.array] = []
        output_list: list[mx.array] = []
        for i in range(batch_size):
            index_begin = np.random.randint(len(self.dry_begin_points))
            sample_begin = self.dry_begin_points[index_begin]
            sample_end = sample_begin + self.dry_width
            this_input = mx.array(self.signal_dry[sample_begin:sample_end])
            this_output = mx.array(
                self.signal_wet[sample_end - self.wet_width : sample_end]
            )
            input_list.append(this_input)
            output_list.append(this_output)
            assert len(this_input) == 10240
            assert len(this_output) == 6148
        return mx.array(input_list), mx.array(output_list)
