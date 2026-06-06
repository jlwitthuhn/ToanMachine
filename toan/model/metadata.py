# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import dataclasses
import datetime
from dataclasses import dataclass


def _export_metadata_dict(metadata) -> dict:
    result = dataclasses.asdict(metadata)
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


@dataclass
class ModelGenericMetadata:
    name: str
    gear_make: str
    gear_model: str
    comment: str | None = None
    loss_test: dict[str, float] = dataclasses.field(default_factory=dict)

    def export_dict(self) -> dict:
        return _export_metadata_dict(self)


@dataclass
class ModelA1Metadata:
    name: str
    gear_make: str
    gear_model: str
    comment: str | None = None
    loss_test: dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_generic(cls, generic: ModelGenericMetadata) -> "ModelA1Metadata":
        return cls(
            name=generic.name,
            gear_make=generic.gear_make,
            gear_model=generic.gear_model,
            comment=generic.comment,
            loss_test=dict(generic.loss_test),
        )

    def export_dict(self) -> dict:
        return _export_metadata_dict(self)


@dataclass
class ModelA2Metadata:
    name: str
    gear_make: str
    gear_model: str
    comment: str | None = None
    loss_test: dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_generic(cls, generic: ModelGenericMetadata) -> "ModelA2Metadata":
        return cls(
            name=generic.name,
            gear_make=generic.gear_make,
            gear_model=generic.gear_model,
            comment=generic.comment,
            loss_test=dict(generic.loss_test),
        )

    def export_dict(self) -> dict:
        return _export_metadata_dict(self)


@dataclass
class SubmodelA2Metadata:
    name: str
    gear_make: str
    gear_model: str
    comment: str | None = None
    loss_test: dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_generic(cls, generic: ModelGenericMetadata) -> "SubmodelA2Metadata":
        return cls(
            name=generic.name,
            gear_make=generic.gear_make,
            gear_model=generic.gear_model,
            comment=generic.comment,
            loss_test=dict(generic.loss_test),
        )

    @classmethod
    def from_a2(cls, a2: ModelA2Metadata) -> "SubmodelA2Metadata":
        return cls(
            name=a2.name,
            gear_make=a2.gear_make,
            gear_model=a2.gear_model,
            comment=a2.comment,
            loss_test=dict(a2.loss_test),
        )

    def export_dict(self) -> dict:
        return _export_metadata_dict(self)
