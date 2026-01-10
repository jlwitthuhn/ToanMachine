# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import dataclasses
import datetime
from dataclasses import dataclass


@dataclass
class ModelMetadata:
    name: str
    gear_make: str
    gear_model: str

    def export_dict(self) -> dict:
        result = dataclasses.asdict(self)
        current_date = datetime.datetime.now()
        result["date"] = {
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "hour": current_date.hour,
            "minute": current_date.minute,
            "second": current_date.second,
        }
        return result
