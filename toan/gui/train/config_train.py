# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.train.context import TrainingGuiContext
from toan.training.config import TrainingStageConfig, get_training_config_from_preset


class TrainTrainConfigPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    radio_default: QtWidgets.QRadioButton

    edit_warmup_steps: QtWidgets.QLineEdit
    edit_main_steps: QtWidgets.QLineEdit
    edit_input_width: QtWidgets.QLineEdit
    edit_lr_hi: QtWidgets.QLineEdit
    edit_lr_lo: QtWidgets.QLineEdit

    def __init__(self, parent, context: TrainingGuiContext):
        super().__init__(parent)
        self.context = context

        self.setCommitPage(True)
        self.setTitle("Training Configuration")

        layout = QtWidgets.QVBoxLayout(self)

        self.radio_default = QtWidgets.QRadioButton("Default training configuration")
        self.radio_default.toggled.connect(self.on_default_toggled)
        layout.addWidget(self.radio_default)

        layout.addSpacing(8)

        radio_custom = QtWidgets.QRadioButton("Custom training configuration")
        radio_custom.toggled.connect(self.on_custom_toggled)
        layout.addWidget(radio_custom)

        form_widget = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(form_widget)
        form_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        form_layout.setFormAlignment(QtCore.Qt.AlignLeft)
        form_layout.setContentsMargins(0, 0, 0, 0)

        self.edit_warmup_steps = QtWidgets.QLineEdit(form_widget)
        form_layout.addRow("Warmup steps:", self.edit_warmup_steps)

        self.edit_main_steps = QtWidgets.QLineEdit(form_widget)
        form_layout.addRow("Main steps:", self.edit_main_steps)

        self.edit_input_width = QtWidgets.QLineEdit(form_widget)
        form_layout.addRow("Input width:", self.edit_input_width)

        self.edit_lr_hi = QtWidgets.QLineEdit(form_widget)
        form_layout.addRow("Learn rate begin:", self.edit_lr_hi)

        self.edit_lr_lo = QtWidgets.QLineEdit(form_widget)
        form_layout.addRow("Learn rate end:", self.edit_lr_lo)

        layout.addWidget(form_widget)

    def initializePage(self):
        # Start on the default configuration: fill the boxes with the preset
        # defaults and lock them for editing.
        self.radio_default.setChecked(True)
        self.populate_defaults()
        self.edit_set_read_only(True)

    def validatePage(self):
        # The boxes always show the values that will be used, so commit
        # whatever they currently hold regardless of which radio is selected.
        try:
            new_warmup_steps = int(self.edit_warmup_steps.text())
            new_main_steps = int(self.edit_main_steps.text())
            new_input_width = int(self.edit_input_width.text())
            new_lr_hi = float(self.edit_lr_hi.text())
            new_lr_lo = float(self.edit_lr_lo.text())
        except ValueError:
            return False

        the_stage = self.context.train_config.stages[0]
        the_stage.steps_warmup = new_warmup_steps
        the_stage.steps_main = new_main_steps
        the_stage.input_sample_width = new_input_width
        the_stage.learn_rate_hi = new_lr_hi
        the_stage.learn_rate_lo = new_lr_lo
        return True

    def on_default_toggled(self, checked: bool):
        # Selecting default re-populates the default values and disables input.
        if checked:
            self.populate_defaults()
            self.edit_set_read_only(True)

    def on_custom_toggled(self, checked: bool):
        # Selecting custom opens the boxes for editing without changing values.
        if checked:
            self.edit_set_read_only(False)

    def default_stage(self) -> TrainingStageConfig:
        default_config = get_training_config_from_preset(self.context.model_preset)
        return default_config.stages[0]

    def populate_defaults(self):
        the_stage = self.default_stage()
        self.edit_warmup_steps.setText(str(the_stage.steps_warmup))
        self.edit_main_steps.setText(str(the_stage.steps_main))
        self.edit_input_width.setText(str(the_stage.input_sample_width))
        self.edit_lr_hi.setText(str(the_stage.learn_rate_hi))
        self.edit_lr_lo.setText(str(the_stage.learn_rate_lo))

    def edit_set_read_only(self, read_only: bool):
        self.edit_warmup_steps.setReadOnly(read_only)
        self.edit_main_steps.setReadOnly(read_only)
        self.edit_input_width.setReadOnly(read_only)
        self.edit_lr_hi.setReadOnly(read_only)
        self.edit_lr_lo.setReadOnly(read_only)
