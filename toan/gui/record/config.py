# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtWidgets

from toan.gui.record.context import RecordingContext


class RecordConfigPage(QtWidgets.QWizardPage):
    context: RecordingContext
    device_make_edit: QtWidgets.QLineEdit
    device_model_edit: QtWidgets.QLineEdit
    radio_option_441: QtWidgets.QRadioButton
    radio_option_480: QtWidgets.QRadioButton

    def __init__(self, parent: QtWidgets.QWidget, context: RecordingContext):
        super().__init__(parent)
        self.context = context

        self.setTitle("Settings")
        layout = QtWidgets.QVBoxLayout(self)

        def create_form_panel() -> QtWidgets.QWidget:
            form_panel = QtWidgets.QWidget(self)
            form_layout = QtWidgets.QFormLayout(form_panel)

            self.device_make_edit = QtWidgets.QLineEdit(form_panel)
            self.device_make_edit.setMinimumWidth(250)

            self.device_model_edit = QtWidgets.QLineEdit(form_panel)
            self.device_model_edit.setMinimumWidth(250)

            radio_widget = QtWidgets.QWidget(form_panel)
            radio_layout = QtWidgets.QVBoxLayout(radio_widget)
            radio_layout.setContentsMargins(0, 0, 0, 0)

            self.radio_option_441 = QtWidgets.QRadioButton("44.1 kHz", radio_widget)
            radio_layout.addWidget(self.radio_option_441)
            self.radio_option_480 = QtWidgets.QRadioButton("48 kHz", radio_widget)
            radio_layout.addWidget(self.radio_option_480)

            form_layout.addRow("Device Manufacturer:", self.device_make_edit)
            form_layout.addRow("Device Model:", self.device_model_edit)
            form_layout.addRow("Sample Rate:", radio_widget)
            return form_panel

        layout.addStretch(1)

        form_panel = create_form_panel()
        layout.addWidget(form_panel)

        layout.addStretch(1)

    def initializePage(self):
        self.device_make_edit.setText(self.context.device_make)
        self.device_model_edit.setText(self.context.device_model)
        if self.context.sample_rate == 44100:
            self.radio_option_441.setChecked(True)
        elif self.context.sample_rate == 48000:
            self.radio_option_480.setChecked(True)

    def validatePage(self):
        self.context.device_make = self.device_make_edit.text()
        self.context.device_model = self.device_model_edit.text()
        if self.radio_option_441.isChecked():
            self.context.sample_rate = 44100
        elif self.radio_option_480.isChecked():
            self.context.sample_rate = 48000
        return True
