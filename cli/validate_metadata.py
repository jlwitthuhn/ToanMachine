# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

from toan.model.metadata import ModelA2Metadata
from toan.model.nam_a2_wavenet_config import json_a2_wavenet_container_config
from toan.model.nam_a2_wavenet_torch import NamA2WaveNetTorch

# Loudness is in dB and gain is a 0..1 ratio; these are written to json as
# 32-bit floats, so allow a small tolerance when comparing.
LOUDNESS_TOLERANCE_DB = 0.05
GAIN_TOLERANCE = 1.0e-4


def _format_comparison(
    label: str, stored: float | None, measured: float, tol: float
) -> tuple[str, bool]:
    if stored is None:
        return (
            f"  {label}: stored=<missing> measured={measured:.6f}  FAIL (no stored value)",
            False,
        )
    diff = abs(stored - measured)
    ok = diff <= tol
    status = "OK" if ok else "FAIL"
    return (
        f"  {label}: stored={stored:.6f} measured={measured:.6f} "
        f"diff={diff:.6g} (tol={tol:g})  {status}",
        ok,
    )


def main() -> int:
    arg_parser = ArgumentParser(
        description="Validate the loudness/gain metadata in an A2 .nam file.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument("nam_path", type=str, help="Path to an A2 .nam file")
    args = arg_parser.parse_args()

    print("Loading nam file...")
    with open(args.nam_path) as file:
        root = json.load(file)

    architecture = root.get("architecture")
    if architecture != "SlimmableContainer":
        print(
            f"Error: expected an A2 (SlimmableContainer) profile, got "
            f"architecture={architecture!r}."
        )
        return 2

    config = json_a2_wavenet_container_config(root["config"])
    sample_rate = float(root["sample_rate"])

    # Metadata content does not affect the computation; build a placeholder so we
    # can construct the model, then read the stored values straight from json.
    metadata = ModelA2Metadata(name="", gear_make="", gear_model="")
    model = NamA2WaveNetTorch(config, metadata, sample_rate)

    submodel_entries = root["config"]["submodels"]
    submodel_weights = [entry["model"]["weights"] for entry in submodel_entries]
    model.import_nam_linear_weights(submodel_weights)

    device = torch.device("mps")
    model.to(device)

    print("Recomputing loudness and gain...")
    model.populate_loudness_and_gain_metadata()

    all_ok = True

    print("\nPer-submodel:")
    for index, entry in enumerate(submodel_entries):
        stored = entry["model"].get("metadata", {})
        measured = model.submodel_metadata[index]
        print(f" Submodel {index} (max_value={entry['max_value']}):")
        for label, stored_val, measured_val, tol in (
            (
                "loudness",
                stored.get("loudness"),
                measured.loudness,
                LOUDNESS_TOLERANCE_DB,
            ),
            ("gain", stored.get("gain"), measured.gain, GAIN_TOLERANCE),
        ):
            line, ok = _format_comparison(label, stored_val, measured_val, tol)
            print(line)
            all_ok = all_ok and ok

    print("\nTop-level (best submodel):")
    top_stored = root.get("metadata", {})
    for label, stored_val, measured_val, tol in (
        (
            "loudness",
            top_stored.get("loudness"),
            model.metadata.loudness,
            LOUDNESS_TOLERANCE_DB,
        ),
        ("gain", top_stored.get("gain"), model.metadata.gain, GAIN_TOLERANCE),
    ):
        line, ok = _format_comparison(label, stored_val, measured_val, tol)
        print(line)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("PASS: stored metadata matches recomputed values.")
        return 0
    print("FAIL: stored metadata does not match recomputed values.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
