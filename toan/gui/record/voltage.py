# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import math

# Frequency of the sine played while measuring voltage with a multimeter
TONE_FREQUENCY = 300

# 0 dBu is 0.775 V RMS
DBU_REFERENCE_VOLTS = math.sqrt(0.6)

UNIT_MILLIVOLTS_RMS = "Millivolts RMS"
UNIT_VOLTS_RMS = "Volts RMS"
UNIT_DBU = "dBu"

VOLTAGE_UNITS = [UNIT_MILLIVOLTS_RMS, UNIT_VOLTS_RMS, UNIT_DBU]


def to_dbu(value: float, unit: str) -> float | None:
    if unit == UNIT_DBU:
        return value
    if unit == UNIT_MILLIVOLTS_RMS:
        volts = value / 1000.0
    elif unit == UNIT_VOLTS_RMS:
        volts = value
    else:
        return None
    if volts <= 0:
        return None
    return 20.0 * math.log10(volts / DBU_REFERENCE_VOLTS)


def dbu_to_millivolts_rms(dbu: float) -> float:
    return DBU_REFERENCE_VOLTS * (10.0 ** (dbu / 20.0)) * 1000.0
