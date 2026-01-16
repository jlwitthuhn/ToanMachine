# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import os

import platformdirs


def create_user_wav_dir() -> None:
    root_dir = platformdirs.user_data_dir("toan", "toan")
    wav_dir = os.path.join(root_dir, "extra")
    os.makedirs(wav_dir, exist_ok=True)
