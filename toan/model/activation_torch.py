# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from torch import nn


def get_activation_module_torch(activation: str) -> nn.Module:
    match activation:
        case "Tanh":
            return nn.Tanh()
    raise NotImplementedError
