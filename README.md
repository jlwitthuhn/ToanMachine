# Toan Machine

Toan Machine is a tool to assist with training a model for use with [Neural Amp Modeler](https://www.neuralampmodeler.com/). All code is licensed under the terms of the [GPLv3](./LICENSE).

This utility allows a user to:
* Record a test signal passing through some piece of equipment (pedal, amp, etc.)
* From a recorded signal, train a NAM neural net to replicate that equipment.
* Load an existing NAM file to hear how it sounds.

Toan Machine is currently built on MLX and will only work on Apple-Silicon-based Macs.

# Installation

To install and run Toan Machine you will need to:
* Clone this git repo
* Create a python virtual environment for Toan Machine
* Inside your virtual environment, install the packages listed in [requirements.txt](./requirements.txt)
* From the root of this repo, run `python3 gui.py`

Once the application is running the wizards will guide you through the recording and training process.

# Usage

For information about how to connect your pedal for capture see the [Quick Start Guide](./docs/quick_start.pdf).

# Acknowledgements

Toan Machine is based on [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler/) which was created by Steven Atkinson. See [third_party/nam](./third_party/nam) for more information.
