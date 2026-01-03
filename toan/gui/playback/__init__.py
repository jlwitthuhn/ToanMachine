# This file is part of Toan Machine and is licensed under the GPLv3
# https://www.gnu.org/licenses/gpl-3.0.en.html
# SPDX-License-Identifier: GPL-3.0-only

from PySide6 import QtCore, QtWidgets

from toan.gui.playback.context import PlaybackContext
from toan.gui.playback.input import PlaybackInputFilePage
from toan.gui.playback.intro import PlaybackIntroPage
from toan.gui.playback.listen import PlaybackListenPage
from toan.gui.playback.validate import PlaybackValidatePage


class PlaybackWizard(QtWidgets.QWizard):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        context: PlaybackContext = PlaybackContext()

        self.addPage(PlaybackIntroPage(self))
        self.addPage(PlaybackInputFilePage(self, context))
        self.addPage(PlaybackValidatePage(self, context))
        self.addPage(PlaybackListenPage(self, context))

        self.setWindowTitle("Playback Wizard")
        self.setModal(True)
