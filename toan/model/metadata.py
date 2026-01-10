# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass


@dataclass
class ModelMetadata:
    name: str
    gear_make: str
    gear_model: str
