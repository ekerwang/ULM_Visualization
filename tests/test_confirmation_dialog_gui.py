import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from ulm_track_correction_gui.gui.confirmation_dialog import YesNoMessageBox


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_yes_is_default_and_focused_while_no_is_escape(app):
    box = YesNoMessageBox(None, "Confirm", "Proceed?")
    try:
        box.show()
        app.processEvents()

        assert box.defaultButton() is box.yes_button
        assert box.escapeButton() is box.no_button
        assert box.yes_button.isDefault()
        assert not box.no_button.isDefault()
        assert QApplication.focusWidget() is box.yes_button
    finally:
        box.close()


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        (Qt.Key_Y, QMessageBox.Yes),
        (Qt.Key_N, QMessageBox.No),
        (Qt.Key_Return, QMessageBox.Yes),
        (Qt.Key_Enter, QMessageBox.Yes),
        (Qt.Key_Escape, QMessageBox.No),
    ),
)
def test_confirmation_shortcuts_choose_expected_button(app, key, expected):
    box = YesNoMessageBox(None, "Confirm", "Proceed?")
    box.show()
    app.processEvents()

    QTest.keyClick(QApplication.focusWidget(), key)
    app.processEvents()

    assert box.result() == expected
    assert not box.isVisible()
