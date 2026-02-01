# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.train.context import TrainingGuiContext
from toan.model.nam_wavenet_presets import get_wavenet_config
from toan.model.presets import ModelConfigPreset


class TrainConfigPage(QtWidgets.QWizardPage):
    context: TrainingGuiContext

    edit_model_name: QtWidgets.QLineEdit
    edit_device_make: QtWidgets.QLineEdit
    edit_device_model: QtWidgets.QLineEdit
    edit_sample_rate: QtWidgets.QLineEdit
    combo_size: QtWidgets.QComboBox
    edit_comment: QtWidgets.QLineEdit

    def __init__(self, parent, context: TrainingGuiContext):
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
        for allowed_model in [
            ModelConfigPreset.NAM_STANDARD,
            ModelConfigPreset.NAM_LITE,
            ModelConfigPreset.NAM_FEATHER,
            ModelConfigPreset.TOAN_STANDARD_PLUS,
        ]:
            label = allowed_model.get_label()
            self.combo_size.addItem(label, allowed_model.value)
        layout.addRow("Model size:", self.combo_size)

        self.edit_comment = QtWidgets.QLineEdit(self)
        self.edit_comment.setMinimumWidth(250)
        layout.addRow("Comment:", self.edit_comment)

    def initializePage(self):
        self.edit_model_name.setText(self.context.loaded_metadata.name)
        self.edit_device_make.setText(self.context.loaded_metadata.gear_make)
        self.edit_device_model.setText(self.context.loaded_metadata.gear_model)
        self.edit_sample_rate.setText(str(self.context.sample_rate))
        self.edit_comment.setText(self.context.loaded_metadata.comment)

    def validatePage(self) -> bool:
        self.context.loaded_metadata.name = self.edit_model_name.text()
        self.context.loaded_metadata.gear_make = self.edit_device_make.text()
        self.context.loaded_metadata.gear_model = self.edit_device_model.text()
        self.context.model_config = get_wavenet_config(
            ModelConfigPreset(self.combo_size.currentData())
        )
        self.context.loaded_metadata.comment = self.edit_comment.text()
        return True
