import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QSizePolicy

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.gui import main_window_editing
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui.main_window_ui import (
    SCOPE_FULL_STACK,
    SCOPE_TRACK_PADDED,
    SCOPE_TRACK_RANGE,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session() -> CorrectionSession:
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [10.0, 2.0, 3.0, 1.0],
                    [11.0, 4.0, 5.0, 1.0],
                ]
            ),
            1: np.asarray([[12.0, 6.0, 7.0, 2.0]]),
        },
        tracks=[
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, np.nan]),
        ],
        image_stack=np.ones((8, 9, 2), dtype=np.float32),
    )


@pytest.fixture
def window(app):
    win = MainWindow()
    win.session = make_session()
    win.undo_manager = UndoManager(win.session)
    win.session.edit_listener = win._on_session_edit
    win.viewer.set_session(win.session)
    win.populate_track_list()
    win.configure_frame_controls()
    win._sync_enabled_state()
    yield win
    win.close()


def test_tools_panel_visually_separates_operation_levels(window, app):
    tools_layout = window.track_level_tools_box.parentWidget().layout()
    assert tools_layout.indexOf(window.track_level_tools_box) < tools_layout.indexOf(
        window.track_internal_tools_box
    )
    assert window.view_tools_box.layout().indexOf(window.mode_buttons["inspect"]) >= 0
    assert window.view_tools_box.layout().indexOf(window.undo_button) >= 0
    assert window.view_tools_box.layout().indexOf(window.redo_button) >= 0
    assert window.track_internal_tools_box.layout().indexOf(
        window.mode_buttons["assign"]
    ) >= 0
    assert window.track_internal_tools_box.layout().indexOf(
        window.mode_buttons["remove"]
    ) >= 0
    assert window.track_internal_tools_box.layout().indexOf(window.add_point_button) >= 0
    assert window.track_internal_tools_box.layout().indexOf(window.add_point_panel) >= 0
    assert window.track_level_tools_box.layout().indexOf(
        window.initialize_track_button
    ) >= 0
    assert window.track_level_tools_box.layout().indexOf(window.merge_button) >= 0
    assert window.track_level_tools_box.layout().indexOf(window.split_button) >= 0
    assert window.track_level_tools_box.layout().indexOf(window.delete_track_button) >= 0

    internal_layout = window.track_internal_tools_box.layout()
    assert internal_layout.getItemPosition(
        internal_layout.indexOf(window.mode_buttons["assign"])
    ) == (0, 0, 1, 1)
    assert internal_layout.getItemPosition(
        internal_layout.indexOf(window.mode_buttons["remove"])
    ) == (1, 0, 1, 1)
    assert internal_layout.getItemPosition(
        internal_layout.indexOf(window.add_point_button)
    ) == (2, 0, 1, 1)
    assert internal_layout.getItemPosition(
        internal_layout.indexOf(window.add_point_panel)
    ) == (3, 0, 1, 1)
    assert window.add_point_panel.header_button.isChecked()
    assert not window.add_point_panel.content.isHidden()

    track_level_layout = window.track_level_tools_box.layout()
    assert track_level_layout.getItemPosition(
        track_level_layout.indexOf(window.initialize_track_button)
    ) == (0, 0, 1, 1)
    assert track_level_layout.getItemPosition(
        track_level_layout.indexOf(window.delete_track_button)
    ) == (0, 1, 1, 1)
    assert track_level_layout.getItemPosition(
        track_level_layout.indexOf(window.merge_button)
    ) == (1, 0, 1, 1)
    assert track_level_layout.getItemPosition(
        track_level_layout.indexOf(window.split_button)
    ) == (1, 1, 1, 1)

    window.show()
    app.processEvents()
    assert window.main_splitter.sizes()[3] > 300


def test_visible_commands_show_shortcuts_and_toggle_edit_modes(
    window,
    app,
    monkeypatch,
):
    assert window.mode_buttons["inspect"].text() == "Inspect (Esc)"
    assert window.mode_buttons["assign"].text() == "Assign (A)"
    assert window.mode_buttons["remove"].text() == "Remove (R)"
    assert window.add_point_button.text() == "Add point (M)"
    assert window.initialize_track_button.text() == "New Track (N)"
    assert window.merge_button.text() == "Merge (J)"
    assert window.split_button.text() == "Split (S)"
    assert window.delete_track_button.text() == "Delete Track (D)"
    assert window.undo_button.text().startswith("Undo (")
    assert window.redo_button.text().startswith("Redo (")
    assert window.play_button.text() == "Play (Space)"
    assert window.display_panel.interpolate_button.text() == "Interpolate Dataset (I)"

    window.show()
    window.activateWindow()
    app.processEvents()
    baseline_heights = {
        mode: button.height() for mode, button in window.mode_buttons.items()
    }
    baseline_hint_heights = {
        mode: button.sizeHint().height()
        for mode, button in window.mode_buttons.items()
    }
    assert len(set(baseline_heights.values())) == 1
    assert len(set(baseline_hint_heights.values())) == 1
    assert all(
        button.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
        for button in window.mode_buttons.values()
    )
    assert not window.display_panel.interpolate_button.isChecked()
    QTest.keyClick(window, Qt.Key_I)
    app.processEvents()
    assert window.display_panel.interpolate_button.isChecked()
    assert window.display_panel.interpolate_button.text() == "Interpolated (I)"

    monkeypatch.setattr(
        main_window_editing,
        "localization_wrapper",
        lambda _frame, _params: np.empty((0, 4)),
    )
    for key, mode in (
        (Qt.Key_A, "assign"),
        (Qt.Key_R, "remove"),
        (Qt.Key_M, "add_point"),
    ):
        QTest.keyClick(window, key)
        app.processEvents()
        assert window.mode_buttons[mode].isChecked()
        assert {
            name: button.height()
            for name, button in window.mode_buttons.items()
        } == baseline_heights
        assert {
            name: button.sizeHint().height()
            for name, button in window.mode_buttons.items()
        } == baseline_hint_heights
        QTest.keyClick(window, key)
        app.processEvents()
        assert window.mode_buttons["inspect"].isChecked()
        assert {
            name: button.height()
            for name, button in window.mode_buttons.items()
        } == baseline_heights

        window.mode_buttons[mode].click()
        app.processEvents()
        assert window.mode_buttons[mode].isChecked()
        window.mode_buttons[mode].click()
        app.processEvents()
        assert window.mode_buttons["inspect"].isChecked()

    inspect_button = window.mode_buttons["inspect"]
    assert inspect_button.contentsRect().width() >= inspect_button.fontMetrics().horizontalAdvance(
        inspect_button.text()
    )
    assert inspect_button.contentsRect().height() >= inspect_button.fontMetrics().height()

    window.mode_buttons["assign"].setChecked(True)
    QTest.keyClick(window, Qt.Key_Escape)
    assert window.mode_buttons["inspect"].isChecked()


def test_assign_and_remove_stay_active_after_valid_clicks(window):
    window.track_list.setCurrentRow(0)

    window.mode_buttons["assign"].setChecked(True)
    window.on_detection_clicked(0, 0)
    assert window.correction_mode == "assign"
    assert window.mode_buttons["assign"].isChecked()

    window.mode_buttons["remove"].setChecked(True)
    window.on_detection_clicked(0, 0)
    assert window.session.local_idx_for_track_frame(0, 0) is None
    assert window.correction_mode == "remove"
    assert window.mode_buttons["remove"].isChecked()


def test_track_count_follows_current_filter(window):
    assert window.track_list.count() == 2
    assert window.track_count_label.text() == "2 shown"

    window.filter_combo.setCurrentText("verified")
    assert window.track_list.count() == 0
    assert window.track_count_label.text() == "0 shown"

    window.filter_combo.setCurrentText("Unverified")
    assert window.track_list.count() == 2
    assert window.track_count_label.text() == "2 shown"

    window.track_list.setCurrentRow(0)
    window.mark_selected_verified()
    assert window.track_list.count() == 1
    assert window.track_count_label.text() == "1 shown"
    assert "'Unverified' filter" in window.track_count_label.toolTip()


def test_number_keys_control_fixed_slots_after_layer_drag(window, app):
    window.show()
    window.activateWindow()
    window.viewer.setFocus()
    app.processEvents()

    panel = window.display_panel
    fixed_slot = panel.slot(4)
    assert fixed_slot.shortcut_badge.text() == "(4)"
    assert panel.slot_layer_key(4) == "points"
    points = panel._layer_groups["points"]
    points_before = points.isChecked()
    QTest.keyClick(window.viewer, Qt.Key_4)
    app.processEvents()
    assert points.isChecked() is not points_before

    panel.move_layer("track_path", 3)
    assert panel.slot(4) is fixed_slot
    assert panel.slot_layer_key(4) == "track_path"
    track_path = panel._layer_groups["track_path"]
    track_path_before = track_path.isChecked()
    QTest.keyClick(window.viewer, Qt.Key_4)
    app.processEvents()
    assert track_path.isChecked() is not track_path_before
    assert points.isChecked() is not points_before

    # Digits remain available for numeric parameter entry.
    track_path_after = track_path.isChecked()
    panel.interp_factor_spin.setFocus()
    panel.interp_factor_spin.selectAll()
    QTest.keyClick(panel.interp_factor_spin, Qt.Key_1)
    app.processEvents()
    assert track_path.isChecked() is track_path_after


def test_initialize_track_updates_list_selection_and_undo_redo(window):
    window.initialize_new_track()

    assert window.session.n_tracks == 3
    assert window.selected_track_id == 2
    assert window.track_list.count() == 3
    assert window.session.track_summary(2)["n_points"] == 0

    window.undo()
    assert window.session.n_tracks == 2
    assert window.selected_track_id is None
    assert window.track_list.count() == 2

    window.redo()
    assert window.session.n_tracks == 3
    assert window.selected_track_id == 2
    assert window.track_list.count() == 3


def test_split_creates_two_flagged_ids_and_structural_undo_redo(
    window,
    monkeypatch,
):
    class AcceptedSplitDialog:
        Accepted = QDialog.Accepted

        def __init__(self, *args, **kwargs):
            self.split_after_frame = 0

        def exec(self):
            return self.Accepted

    monkeypatch.setattr(main_window_editing, "SplitTrackDialog", AcceptedSplitDialog)
    monkeypatch.setattr(window, "_confirm", lambda *args: True)
    window.track_list.setCurrentRow(0)
    window.open_split_dialog()

    assert not window.session.is_track_active(0)
    assert window.session.track_status[2] == "flagged"
    assert window.session.track_status[3] == "flagged"
    assert window.selected_track_id == 2
    assert window.track_list.count() == 3

    window.undo()
    assert window.session.n_tracks == 2
    assert window.session.is_track_active(0)
    assert window.selected_track_id == 0
    assert window.track_list.count() == 2

    window.redo()
    assert window.session.n_tracks == 4
    assert not window.session.is_track_active(0)
    assert window.selected_track_id in (2, 3)
    assert window.track_list.count() == 3


@pytest.mark.parametrize("scope", [SCOPE_TRACK_RANGE, SCOPE_TRACK_PADDED])
def test_delete_track_selects_first_remaining_and_preserves_scope(
    window,
    monkeypatch,
    scope,
):
    monkeypatch.setattr(window, "_confirm", lambda *args: True)
    remaining_id = int(window.track_list.item(0).data(Qt.UserRole))
    window.track_list.setCurrentRow(1)
    deleted_id = window.selected_track_id
    window.scope_buttons[scope].click()
    assert window.playback_scope == scope

    window.delete_selected_track()

    assert not window.session.is_track_active(deleted_id)
    assert window.selected_track_id == remaining_id
    assert int(window.track_list.currentItem().data(Qt.UserRole)) == remaining_id
    assert window.track_list.count() == 1
    assert window.playback_scope == scope

    window.undo()
    assert window.session.is_track_active(deleted_id)
    assert window.selected_track_id == deleted_id
    assert window.track_list.count() == 2
    assert window.playback_scope == scope

    window.redo()
    assert not window.session.is_track_active(deleted_id)
    assert window.selected_track_id == remaining_id
    assert window.track_list.count() == 1
    assert window.playback_scope == scope


def test_delete_last_filtered_track_falls_back_to_full_stack(window, monkeypatch):
    monkeypatch.setattr(window, "_confirm", lambda *args: True)
    window.session.set_track_status(0, "verified")
    window.filter_combo.setCurrentText("unreviewed")
    assert window.track_list.count() == 1
    window.track_list.setCurrentRow(0)
    deleted_id = window.selected_track_id
    window.scope_buttons[SCOPE_TRACK_PADDED].click()

    window.delete_selected_track()

    assert not window.session.is_track_active(deleted_id)
    assert window.track_list.count() == 0
    assert window.selected_track_id is None
    assert window.playback_scope == SCOPE_FULL_STACK
