# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import threading
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from toan.model.nam_wavenet_presets import get_wavenet_config
from toan.model.presets import ModelConfigPreset
from toan.training.config import TrainingConfig
from toan.training.context import TrainingProgressContext
from toan.training.loop import run_training_loop
from toan.training.zip_loader import ZipLoaderContext, run_zip_loader


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

    def train_model(name: str, train_config: TrainingConfig):
        print(f"Beginning training for {name}")

        train_context = TrainingProgressContext()

        model_preset = ModelConfigPreset.NAM_STANDARD
        model_config = get_wavenet_config(model_preset)
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
        last_loss: float = train_context.loss_test
        threading.Thread(target=thread_func).start()

        while True:
            with train_context.lock:
                if train_context.loss_test != last_loss:
                    last_loss = train_context.loss_test
                    print(f"Test loss: {last_loss}")
                if train_context.model is not None:
                    break
            time.sleep(1.0)

        print("Training complete")

    # Copy paste the below bit to do multiple training runs with different configs

    train_config = TrainingConfig()
    train_model("default", train_config)


if __name__ == "__main__":
    main()
