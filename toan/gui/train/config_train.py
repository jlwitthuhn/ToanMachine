# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.train.context import TrainingGuiContext


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
        layout.addWidget(self.radio_default)

        layout.addSpacing(8)

        radio_custom = QtWidgets.QRadioButton("Custom training configuration")
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
        the_stage = self.context.train_config.stages[0]
        self.radio_default.setChecked(True)

        self.edit_warmup_steps.setReadOnly(True)
        self.edit_warmup_steps.setText(str(the_stage.steps_warmup))

        self.edit_main_steps.setReadOnly(True)
        self.edit_main_steps.setText(str(the_stage.steps_main))

        self.edit_input_width.setReadOnly(True)
        self.edit_input_width.setText(str(the_stage.input_sample_width))

        self.edit_lr_hi.setReadOnly(True)
        self.edit_lr_hi.setText(str(the_stage.learn_rate_hi))

        self.edit_lr_lo.setReadOnly(True)
        self.edit_lr_lo.setText(str(the_stage.learn_rate_lo))
