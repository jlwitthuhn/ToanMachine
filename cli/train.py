# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math
import os
import threading
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

from toan.model.nam_a1_wavenet_presets import get_a1_wavenet_config
from toan.model.presets import ModelConfigPreset
from toan.training.config import TrainingConfig
from toan.training.context import TrainingProgressContext
from toan.training.loop import run_training_loop
from toan.training.zip_loader import ZipLoaderContext, run_zip_loader


@dataclass
class _LossStats:
    min: float = math.inf
    max: float = math.inf
    std: float = math.inf
    med: float = math.inf
    mean: float = math.inf

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

    args = arg_parser.parse_args()

    print("Loading recording zip file...")
    zip_context = ZipLoaderContext()
    run_zip_loader(zip_context, args.zip_path)

    def do_iteration(
        name: str, train_config: TrainingConfig, save_model: bool = True, index: int = 1
    ) -> float:
        print(f"Beginning training for {name}")

        train_context = TrainingProgressContext()

        model_preset = ModelConfigPreset.NAM_A1_STANDARD
        model_config = get_a1_wavenet_config(model_preset)
        train_context.model_config = model_config
        train_context.metadata = zip_context.metadata
        train_context.sample_rate = zip_context.sample_rate

        train_context.signal_dry_test = zip_context.signal_dry_test[:]
        train_context.signal_wet_test = zip_context.signal_wet_test[:]
        train_context.signal_dry_train = zip_context.signal_dry[:]
        train_context.signal_wet_train = zip_context.signal_wet[:]

        def thread_func():
            run_training_loop(train_context, train_config)

        print("Data loaded, beginning training...")
        threading.Thread(target=thread_func).start()

        with tqdm(total=train_config.steps_total()) as progress_bar:
            last_loss: float = train_context.loss_test
            while True:
                with train_context.lock:
                    if train_context.loss_test != last_loss:
                        last_loss = train_context.loss_test
                        progress_bar.set_description(
                            f"Test: {train_context.loss_test:0.5f}"
                        )
                    if train_context.model is not None:
                        break
                    progress_bar.update(train_context.iters_done - progress_bar.n)
                time.sleep(1.0)

        if save_model:
            print("Training complete, saving model...")
            model_root_path = f"./output/{name}"
            graph_path = f"{model_root_path}/graph.png"
            os.makedirs(model_root_path, exist_ok=True)
            fig: Figure = train_context.summary.generate_loss_graph(3)
            fig.savefig(graph_path)
            model_path = f"{model_root_path}/model.nam"
            with open(model_path, "w") as file:
                file.write(train_context.model.export_nam_json_str())

        return train_context.loss_test

    loss_dict: dict[str, _LossStats] = {}

    def do_iteration_and_log(
        label: str,
        train_config: TrainingConfig,
        save_model: bool = True,
        count: int = 1,
    ):
        losses: list[float] = []
        original_seed = train_config.rng_seed
        for i in range(count):
            train_config.rng_seed = original_seed + i
            loss = do_iteration(label, train_config, save_model, i)
            losses.append(loss)
        loss_min: float = np.min(losses)
        loss_max: float = np.max(losses)
        loss_mean: float = float(np.mean(losses))
        loss_stats = _LossStats(min=loss_min, max=loss_max, mean=loss_mean)
        if len(losses) >= 3:
            loss_stats.std = float(np.std(losses))
            loss_stats.med = float(np.median(losses))
        print(f"{label} summary:")
        print(loss_stats.as_formatted_str())
        loss_dict[label] = loss_stats

    # Copy paste the below bit to do multiple training runs with different configs

    train_config = TrainingConfig()
    do_iteration_and_log("default", train_config, False, 2)


if __name__ == "__main__":
    main()
