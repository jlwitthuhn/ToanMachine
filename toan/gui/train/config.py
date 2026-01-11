# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.train.context import TrainingContext
from toan.model.size_presets import ModelSizePreset


class TrainConfigPage(QtWidgets.QWizardPage):
    context: TrainingContext

    edit_model_name: QtWidgets.QLineEdit
    edit_device_make: QtWidgets.QLineEdit
    edit_device_model: QtWidgets.QLineEdit
    edit_sample_rate: QtWidgets.QLineEdit
    combo_size: QtWidgets.QComboBox

    def __init__(self, parent, context: TrainingContext):
        super().__init__(parent)
        self.context = context

        self.setCommitPage(True)
        self.setTitle("Configuration")
        layout = QtWidgets.QFormLayout(self)

        self.edit_model_name = QtWidgets.QLineEdit(self)
        self.edit_model_name.setMinimumWidth(250)
        layout.addRow("NAM model name:", self.edit_model_name)

        self.edit_device_make = QtWidgets.QLineEdit(self)
        self.edit_device_make.setMinimumWidth(250)
        layout.addRow("Device manufacturer:", self.edit_device_make)

        self.edit_device_model = QtWidgets.QLineEdit(self)
        self.edit_device_model.setMinimumWidth(250)
        layout.addRow("Device model:", self.edit_device_model)

        self.edit_sample_rate = QtWidgets.QLineEdit(self)
        self.edit_sample_rate.setMinimumWidth(250)
        self.edit_sample_rate.setReadOnly(True)
        layout.addRow("Sample rate:", self.edit_sample_rate)

        self.combo_size = QtWidgets.QComboBox(self)
        for allowed_model in [ModelSizePreset.NAM_STANDARD]:
            label = allowed_model.get_label()
            self.combo_size.addItem(label, allowed_model.value)
        layout.addRow("Model size:", self.combo_size)

    def initializePage(self):
        self.edit_model_name.setText(self.context.loaded_metadata.name)
        self.edit_device_make.setText(self.context.loaded_metadata.gear_make)
        self.edit_device_model.setText(self.context.loaded_metadata.gear_model)
        self.edit_sample_rate.setText(str(self.context.sample_rate))

    def validatePage(self) -> bool:
        self.context.loaded_metadata.name = self.edit_model_name.text()
        self.context.loaded_metadata.gear_make = self.edit_device_make.text()
        self.context.loaded_metadata.gear_model = self.edit_device_model.text()
        return True
