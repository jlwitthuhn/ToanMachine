# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def main():
    arg_parser = ArgumentParser(
        description="Script to train a NAM model with no gui. Does not support recording.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument("zip_path", type=str, help="Path to recording zip file")

    arg_parser.parse_args()

    print("Conglaturations")


if __name__ == "__main__":
    main()
