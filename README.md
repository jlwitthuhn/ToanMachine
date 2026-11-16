# Toan Machine

Toan Machine is intended to be the one program you need to train a high-quality [Neural Amp Modeler](https://www.neuralampmodeler.com/) A2 model. All code is licensed under the terms of the [GPLv3](./LICENSE).

The main design philosophy behind this project is to create an interface that directly guides an inexperienced user through to process of creating a high quality calibrated capture. This includes:
* Recording wet/dry signal directly within Toan Machine, no DAW required.
* Calibrating input and output voltage with a multimeter to ensure accuracy.
* Recorded wet/dry pair and metadata are bundled together as a single zip file. It is impossible to mix up your tracks or load incorrect metadata.
* Users can inject their own guitar DI into the training or test data.

Toan Machine is currently only compatible with Apple Silicon Macs. Support for nVidia cards on Windows is planned for the future.

# Installation

There are two main ways to install Toan Machine, depending on your level of familiarity with Python environment management.

## Pre-built .App Bundle

The easiest way to install Toan Machine is to download a pre-built App from the releases section [here](https://github.com/jlwitthuhn/ToanMachine/releases). Download the Mac DMG file and follow the instructions contained inside.

## Run From Source

If you would like to run from this source instead, you will need to:
* Clone this git repo
* Create a python virtual environment for Toan Machine
* Inside your virtual environment, install the packages listed in [requirements.txt](./requirements.txt)
* From the root of this repo, run `python3 gui.py`

Once the application is running the wizards will guide you through the recording and training process.

# Usage

For information about how to connect your pedal for capture see the [Quick Start Guide](./docs/quick_start.pdf).

# Acknowledgements

Toan Machine is based on [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler/) which was created by Steven Atkinson. See [third_party/nam](./third_party/nam) for more information.
