"""Consistent keyboard behavior for two-choice confirmation dialogs."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


class YesNoMessageBox(QMessageBox):
    """A Yes/No message box with explicit, platform-independent shortcuts.

    ``Y`` and either Enter key choose Yes; ``N`` and Escape choose No.  Yes is
    always the default and initially focused button so macOS must not promote
    No merely because of native button ordering or role styling.
    """

    def __init__(self, parent: QWidget | None, title: str, text: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setIcon(QMessageBox.Question)
        self.setText(text)
        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        self.yes_button = self.button(QMessageBox.Yes)
        self.no_button = self.button(QMessageBox.No)
        self.setDefaultButton(self.yes_button)
        self.setEscapeButton(self.no_button)
        self.yes_button.setAutoDefault(True)
        self.yes_button.setDefault(True)
        self.no_button.setAutoDefault(False)
        self.no_button.setDefault(False)

        self._application_filter_installed = False

    def _handle_confirmation_key(self, event) -> bool:
        if (
            event.type() != QEvent.KeyPress
            or event.isAutoRepeat()
            or event.modifiers() not in (Qt.NoModifier, Qt.KeypadModifier)
        ):
            return False
        if event.key() in (Qt.Key_Y, Qt.Key_Return, Qt.Key_Enter):
            button = self.yes_button
        elif event.key() in (Qt.Key_N, Qt.Key_Escape):
            button = self.no_button
        else:
            return False
        event.accept()
        button.click()
        return True

    def eventFilter(self, watched, event) -> bool:
        app = QApplication.instance()
        is_active_confirmation = app is not None and (
            app.activeModalWidget() is self or app.activeWindow() is self
        )
        if is_active_confirmation and self._handle_confirmation_key(event):
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:
        if not self._handle_confirmation_key(event):
            super().keyPressEvent(event)

    def _install_application_filter(self) -> None:
        app = QApplication.instance()
        if app is not None and not self._application_filter_installed:
            app.installEventFilter(self)
            self._application_filter_installed = True

    def _remove_application_filter(self) -> None:
        app = QApplication.instance()
        if app is not None and self._application_filter_installed:
            app.removeEventFilter(self)
        self._application_filter_installed = False

    def _focus_yes_button(self) -> None:
        if not self.isVisible():
            return
        self.yes_button.setDefault(True)
        self.no_button.setDefault(False)
        self.yes_button.setFocus(Qt.OtherFocusReason)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._install_application_filter()
        self._focus_yes_button()
        # Apply once more after the platform style completes button ordering.
        QTimer.singleShot(0, self._focus_yes_button)

    def hideEvent(self, event) -> None:
        self._remove_application_filter()
        super().hideEvent(event)


def ask_yes_no(parent: QWidget | None, title: str, text: str) -> bool:
    """Show the standard two-choice confirmation and return whether Yes won."""

    return YesNoMessageBox(parent, title, text).exec() == QMessageBox.Yes
