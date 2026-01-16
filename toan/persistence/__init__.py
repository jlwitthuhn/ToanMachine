# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs


@dataclass
class UserWavDesc:
    path: str
    filename: str


def create_user_wav_dir() -> None:
    os.makedirs(get_user_wav_dir(), exist_ok=True)


def get_user_wav_dir() -> str:
    root_dir = platformdirs.user_data_dir("toan", "toan")
    return os.path.join(root_dir, "extra")


def get_user_wav_list() -> list[UserWavDesc]:
    result = []
    wav_dir = Path(get_user_wav_dir())
    for file in wav_dir.glob("*.wav"):
        path = wav_dir / file.name
        result.append(UserWavDesc(str(path), file.name))
    return result
