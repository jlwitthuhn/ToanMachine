# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json
import math
import os
import threading
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from toan.model.nam_a2_wavenet_presets import get_a2_wavenet_config
from toan.model.presets import ModelConfigPreset
from toan.training.config import TrainingConfig, get_training_config_from_preset
from toan.training.context import TrainingProgressContext
from toan.training.loop_torch import run_training_loop_torch
from toan.training.zip_loader import ZipLoaderContext, run_zip_loader

THE_PRESET: ModelConfigPreset = ModelConfigPreset.A2_NAM


def _get_model_config(preset: ModelConfigPreset):
    config = get_a2_wavenet_config(preset)
    if config is None:
        raise NotImplementedError(f"No model config for preset {preset}")
    return config


@dataclass
class _LossStats:
    min: float = math.inf
    max: float = math.inf
    std: float = math.inf
    med: float = math.inf
    mean: float = math.inf

    def __init__(self, losses: list[float]):
        self.min = float(np.min(losses))
        self.max = float(np.max(losses))
        self.mean = float(np.mean(losses))
        self.std = float(np.std(losses)) if len(losses) >= 3 else math.inf
        self.med = float(np.median(losses)) if len(losses) >= 3 else math.inf

    def as_formatted_str(self) -> str:
        vars = [f" min: {self.min}", f" max: {self.max}"]
        if self.std < math.inf:
            vars.append(f" std: {self.std}")
        if self.med < math.inf:
            vars.append(f" med: {self.med}")
        vars.append(f"mean: {self.mean}")
        return "\n".join(vars)


def main():
    arg_parser = ArgumentParser(
        description="Script to train a NAM model with no gui. Does not support recording.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument("zip_path", type=str, help="Path to recording zip file")
    arg_parser.add_argument(
        "--output",
        type=str,
        help=(
            "Directory for per-configuration loss graphs and training.jsonl, which "
            "appends training loss statistics and stage timing"
        ),
    )

    args = arg_parser.parse_args()

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    print("Loading recording zip file...")
    zip_context = ZipLoaderContext()
    run_zip_loader(zip_context, args.zip_path)

    def do_iteration(
        name: str, train_config: TrainingConfig, index: int = 1
    ) -> tuple[float, list[float]]:
        print(f"Iteration {index}")

        train_context = TrainingProgressContext()

        model_config = _get_model_config(THE_PRESET)

        train_context.model_config = model_config
        train_context.metadata = zip_context.metadata
        train_context.sample_rate = zip_context.sample_rate

        train_context.signal_dry_test = zip_context.signal_dry_test[:]
        train_context.signal_wet_test = zip_context.signal_wet_test[:]
        train_context.signal_dry_train = zip_context.signal_dry[:]
        train_context.signal_wet_train = zip_context.signal_wet[:]

        def thread_func():
            run_training_loop_torch(train_context, train_config)

        print("Data loaded, beginning training...")
        threading.Thread(target=thread_func).start()

        with tqdm(total=train_config.steps_total()) as progress_bar:
            last_loss: float = train_context.loss_test
            while True:
                with train_context.lock:
                    if train_context.loss_test != last_loss:
                        last_loss = train_context.loss_test
                        progress_bar.set_description(
                            f"Test: {train_context.loss_test:0.8f}"
                        )
                    if train_context.model is not None:
                        break
                    progress_bar.update(train_context.iters_done - progress_bar.n)
                time.sleep(1.0)

        if args.output is not None:
            fig: Figure = train_context.summaries[-1].generate_loss_graph(3)
            try:
                fig.savefig(os.path.join(args.output, f"loss_{name}.png"))
            finally:
                plt.close(fig)

        stage_timing = [summary.duration_seconds for summary in train_context.summaries]
        if train_context.loss_test is not None:
            return train_context.loss_test, stage_timing
        else:
            return train_context.loss_train, stage_timing

    loss_dict: dict[str, _LossStats] = {}

    def multi_train_with_config(
        label: str,
        train_config: TrainingConfig,
        count: int = 1,
    ):
        print(f"Beginning training for {label}")
        losses: list[float] = []
        stage_timings: list[list[float]] = []
        original_seed = train_config.rng_seed
        for i in range(count):
            train_config.rng_seed = original_seed + i
            loss, stage_timing = do_iteration(label, train_config, i)
            losses.append(loss)
            stage_timings.append(stage_timing)
        loss_stats = _LossStats(losses)
        if args.output is not None:
            with open(
                os.path.join(args.output, "training.jsonl"), "a", encoding="utf-8"
            ) as output_file:
                output_file.write(
                    json.dumps(
                        {
                            "name": label,
                            "loss": asdict(loss_stats),
                            "stage_timing": np.median(stage_timings, axis=0).tolist(),
                        }
                    )
                    + "\n"
                )
        print(f"{label} summary:")
        print(loss_stats.as_formatted_str())
        loss_dict[label] = loss_stats

    iter_count = 7
    train_config = get_training_config_from_preset(THE_PRESET)
    train_config.stages[0].test_interval = 0

    # Copy paste the below bit to do multiple training runs with different configs
    multi_train_with_config("default", train_config, iter_count)

    print()
    print("++ Summary ++")
    print()
    for label, stats in loss_dict.items():
        print(f"{label}:")
        print(stats.as_formatted_str())
        print()


if __name__ == "__main__":
    main()
