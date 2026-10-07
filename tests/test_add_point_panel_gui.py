import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.gui import main_window_editing
from ulm_track_correction_gui.gui.add_point_panel import (
    MANUAL_NUDGE_STEP_PX,
    MANUAL_PLACEMENT_METHOD,
    AddPointPanel,
)
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui.viewer import CANDIDATE_Z
from ulm_track_correction_gui.reference_processing.wrappers import (
    FS_GRADIENT_NCC_METHOD,
    RADIAL_NCC_METHOD,
    RADIAL_NON_NCC_METHOD,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session() -> CorrectionSession:
    stack = np.zeros((32, 36, 2), dtype=np.float32)
    stack[19, 24, 0] = 8.0
    stack[21, 26, 1] = 9.0
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray([[2.0, 3.0, 4.0, 1.0]]),
            1: np.asarray([[3.0, 6.0, 7.0, 2.0]]),
        },
        tracks=[np.asarray([0.0, 0.0])],
        image_stack=stack,
    )


def make_window(app) -> MainWindow:
    window = MainWindow()
    window.session = make_session()
    window.undo_manager = UndoManager(window.session)
    window.session.edit_listener = window._on_session_edit
    window.viewer.set_session(window.session)
    window.populate_track_list()
    window.configure_frame_controls()
    window.track_list.setCurrentRow(0)
    window._sync_enabled_state()
    window.show()
    window.activateWindow()
    app.processEvents()
    return window


def test_add_point_panel_defaults_match_reviewed_candidate_settings(
    app,
) -> None:
    panel = AddPointPanel()
    panel.show()
    app.processEvents()
    try:
        assert panel.header_button.isChecked()
        assert panel.content.isVisible()
        assert not panel.eraser_button.isChecked()
        assert panel.eraser_button.text() == "Erase unassigned added point"
        assert [
            (panel.method_combo.itemText(index), panel.method_combo.itemData(index))
            for index in range(panel.method_combo.count())
        ] == [
            ("Radial Symmetry (non-NCC)", RADIAL_NON_NCC_METHOD),
            ("FS_Gradient (NCC)", FS_GRADIENT_NCC_METHOD),
            ("Radial Symmetry (NCC)", RADIAL_NCC_METHOD),
            ("Manual Placement", MANUAL_PLACEMENT_METHOD),
        ]

        assert panel.current_params() == {
            "method": RADIAL_NCC_METHOD,
            "psfSizeZAxis": 3.0,
            "psfSizeXAxis": 4.5,
            "psfWindowSize": 5,
            "ampThreshold": 0.10,
            "corrThreshold": 0.40,
        }
        assert panel.candidate_style() == {
            "marker": "target",
            "color": "#00dcff",
            "radius": 1.0,
            "opacity": 0.75,
        }

        panel.method_combo.setCurrentIndex(
            panel.method_combo.findData(FS_GRADIENT_NCC_METHOD)
        )
        assert panel.filter_size_combo.isVisible()
        assert panel.current_params()["filterSize"] == 7

        panel.method_combo.setCurrentIndex(panel.method_combo.findData(RADIAL_NCC_METHOD))
        app.processEvents()
        assert panel.filter_size_combo.isHidden()

        panel.method_combo.setCurrentIndex(
            panel.method_combo.findData(MANUAL_PLACEMENT_METHOD)
        )
        app.processEvents()
        assert panel.manual_hint_label.isVisible()
        assert panel.manual_hint_label.height() >= panel.manual_hint_label.heightForWidth(
            panel.manual_hint_label.width()
        )
        assert panel.parameter_stack.isHidden()
        assert panel.current_params() == {
            "method": MANUAL_PLACEMENT_METHOD,
            "nudgeStepPx": MANUAL_NUDGE_STEP_PX,
        }
        content_layout = panel.content.layout()
        assert content_layout.indexOf(panel.eraser_button) == content_layout.count() - 1
        height_before_eraser = panel.sizeHint().height()

        panel.eraser_button.click()
        app.processEvents()
        assert panel.eraser_active()
        assert panel.manual_hint_label.isVisible()
        assert not panel.manual_hint_label.isEnabled()
        assert not panel.method_combo.isEnabled()
        assert not panel.marker_combo.isEnabled()
        assert panel.sizeHint().height() == height_before_eraser

        panel.eraser_button.click()
        app.processEvents()
        assert panel.manual_hint_label.isVisible()
        assert panel.manual_hint_label.isEnabled()
        assert panel.method_combo.isEnabled()
    finally:
        panel.close()


def test_add_point_mode_auto_localizes_restyles_and_refreshes_on_frame_change(
    app,
    monkeypatch,
) -> None:
    window = make_window(app)
    calls: list[tuple[np.ndarray, dict]] = []

    def fake_localization(frame, params):
        calls.append((np.asarray(frame), dict(params)))
        frame_number = window.frame_spin.value()
        return np.asarray([[7.5, 20.5, 25.5, float(frame_number)]])

    monkeypatch.setattr(main_window_editing, "localization_wrapper", fake_localization)
    try:
        assert window.add_point_panel.content.isVisible()
        window.mode_buttons["add_point"].setChecked(True)
        app.processEvents()

        assert window.correction_mode == "add_point"
        assert window.viewer.interaction_mode == "add_point"
        assert len(calls) == 1
        assert window.viewer.candidate_rows.shape == (1, 4)
        assert "1 candidate(s)" in window.add_point_panel.status_label.text()
        candidate_items = [
            item for item in window.viewer.scene.items() if item.data(0) == "candidate"
        ]
        assert candidate_items
        assert {item.zValue() for item in candidate_items} == {CANDIDATE_Z}
        assert CANDIDATE_Z > max(window.viewer._layer_z.values())

        window.add_point_panel.marker_combo.setCurrentText("Cross")
        app.processEvents()
        assert len(calls) == 1, "restyling must not rerun localization"
        assert window.viewer._candidate_style["marker"] == "cross"

        window.add_point_panel.corr_threshold_spin.setValue(0.45)
        assert window._add_point_refresh_timer.isActive()
        window._add_point_refresh_timer.stop()
        window.refresh_add_point_candidates()
        assert len(calls) == 2
        assert calls[-1][1]["corrThreshold"] == 0.45

        window.set_frame_1based(2)
        assert window._add_point_refresh_timer.isActive()
        assert window.viewer.candidate_rows.shape == (0, 4)
        window._add_point_refresh_timer.stop()
        window.refresh_add_point_candidates()
        assert len(calls) == 3
        assert calls[-1][0] is not calls[0][0]
        assert window._add_point_candidate_frame == 1
    finally:
        window.close()


def test_comma_and_period_cycle_add_point_methods_in_context(
    app,
    monkeypatch,
) -> None:
    window = make_window(app)
    monkeypatch.setattr(
        main_window_editing,
        "localization_wrapper",
        lambda _frame, _params: np.asarray([[7.5, 20.5, 25.5, 1.0]]),
    )
    panel = window.add_point_panel
    try:
        assert panel.current_method() == RADIAL_NCC_METHOD
        QTest.keyClick(window.viewer, Qt.Key_Comma)
        assert panel.current_method() == RADIAL_NCC_METHOD

        window.mode_buttons["add_point"].setChecked(True)
        QTest.keyClick(window.viewer, Qt.Key_Comma)
        assert panel.current_method() == FS_GRADIENT_NCC_METHOD
        window._add_point_refresh_timer.stop()

        QTest.keyClick(window.viewer, Qt.Key_Period)
        assert panel.current_method() == RADIAL_NCC_METHOD
        window._add_point_refresh_timer.stop()

        panel.method_combo.setCurrentIndex(0)
        window._add_point_refresh_timer.stop()
        QTest.keyClick(window.viewer, Qt.Key_Comma)
        assert panel.current_method() == MANUAL_PLACEMENT_METHOD
        assert window.viewer.manual_mode_badge.isVisible()
        assert "MANUAL PLACEMENT ACTIVE" in window.viewer.manual_mode_badge.text()
        QTest.keyClick(window.viewer, Qt.Key_Period)
        assert panel.current_method() == RADIAL_NON_NCC_METHOD
        assert window.viewer.manual_mode_badge.isHidden()
        window._add_point_refresh_timer.stop()

        panel.method_combo.setCurrentIndex(panel.method_combo.findData(RADIAL_NCC_METHOD))
        window._add_point_refresh_timer.stop()
        panel.psf_z_spin.setFocus()
        QTest.keyClick(panel.psf_z_spin, Qt.Key_Period)
        assert panel.current_method() == RADIAL_NCC_METHOD

        panel.eraser_button.click()
        window.viewer.setFocus()
        QTest.keyClick(window.viewer, Qt.Key_Comma)
        assert panel.current_method() == RADIAL_NCC_METHOD
    finally:
        window.close()


def test_eraser_toggle_does_not_grow_manual_settings_card(app) -> None:
    window = make_window(app)
    try:
        panel = window.add_point_panel
        panel.method_combo.setCurrentIndex(
            panel.method_combo.findData(MANUAL_PLACEMENT_METHOD)
        )
        app.processEvents()
        height_before = panel.height()

        panel.eraser_button.click()
        app.processEvents()

        assert panel.eraser_active()
        assert panel.height() <= height_before
        assert panel.manual_hint_label.height() >= (
            panel.manual_hint_label.heightForWidth(panel.manual_hint_label.width())
        )
    finally:
        window.close()


def test_clicking_candidate_keeps_mode_and_next_frame_refreshes_candidates(
    app,
    monkeypatch,
) -> None:
    window = make_window(app)

    def fake_localization(_frame, _params):
        return np.asarray([[7.5, 20.5, 25.5, 1.0]])

    monkeypatch.setattr(main_window_editing, "localization_wrapper", fake_localization)
    try:
        window.mode_buttons["add_point"].setChecked(True)
        app.processEvents()
        before = window.session.localized_by_frame[0].shape[0]
        scene_x, scene_y = window.viewer._pala_to_display_xy(20.5, 25.5)
        view_pos = window.viewer.mapFromScene(scene_x, scene_y)
        QTest.mouseClick(
            window.viewer.viewport(),
            Qt.LeftButton,
            pos=view_pos,
        )
        app.processEvents()

        assert window.session.localized_by_frame[0].shape[0] == before + 1
        new_local_idx = before
        assert window.session.is_manual_detection(0, new_local_idx)
        assert window.session.local_idx_for_track_frame(0, 0) == new_local_idx
        append_edit = next(
            edit
            for edit in reversed(window.session.edit_log)
            if edit.action == "append_manual_detection"
        )
        assert append_edit.payload["source"] == "localization_candidates"
        assert append_edit.payload["params"]["method"] == RADIAL_NCC_METHOD
        assert window.mode_buttons["add_point"].isChecked()
        assert window.correction_mode == "add_point"
        assert window.viewer.candidate_rows.shape == (1, 4)

        window.set_frame_1based(2)
        assert window._add_point_refresh_timer.isActive()
        assert window.viewer.candidate_rows.shape == (0, 4)
        window._add_point_refresh_timer.stop()
        window.refresh_add_point_candidates()
        assert window._add_point_candidate_frame == 1
        assert window.viewer.candidate_rows.shape == (1, 4)

        window.add_point_button.click()
        assert window.mode_buttons["inspect"].isChecked()
        assert window.correction_mode == "inspect"
        assert window.viewer.candidate_rows.shape == (0, 4)
    finally:
        window.close()


def test_add_point_on_verified_track_flags_atomically_and_undo_restores_it(
    app,
    monkeypatch,
) -> None:
    window = make_window(app)
    window.session.track_status[0] = "verified"
    window.populate_track_list()
    window.track_list.setCurrentRow(0)
    monkeypatch.setattr(window, "_confirm", lambda *_args: True)
    monkeypatch.setattr(
        main_window_editing,
        "localization_wrapper",
        lambda _frame, _params: np.asarray([[7.5, 20.5, 25.5, 1.0]]),
    )
    try:
        before = window.session.localized_by_frame[0].shape[0]
        window.mode_buttons["add_point"].setChecked(True)
        window.on_candidate_clicked(0, 0)

        assert window.session.track_status[0] == "flagged"
        compound = window.session.edit_log[-3:]
        assert [edit.action for edit in compound] == [
            "append_manual_detection",
            "assign_detection",
            "set_track_status",
        ]
        assert len({edit.group_id for edit in compound}) == 1

        window.undo()
        assert window.session.localized_by_frame[0].shape[0] == before
        assert window.session.local_idx_for_track_frame(0, 0) == 0
        assert window.session.track_status[0] == "verified"
    finally:
        window.close()


def test_human_defined_point_requires_keyboard_confirmation(app, monkeypatch) -> None:
    window = make_window(app)
    monkeypatch.setattr(
        main_window_editing,
        "localization_wrapper",
        lambda *_args: pytest.fail("manual placement must not run localization"),
    )
    try:
        window.add_point_panel.method_combo.setCurrentIndex(
            window.add_point_panel.method_combo.findData(MANUAL_PLACEMENT_METHOD)
        )
        window.mode_buttons["add_point"].setChecked(True)
        app.processEvents()

        assert window.correction_mode == "add_point"
        assert window.viewer._manual_candidate_placement_enabled
        assert window.viewer.candidate_rows.shape == (0, 4)
        assert window.viewer.manual_mode_badge.isVisible()
        assert "MANUAL PLACEMENT ACTIVE" in window.viewer.manual_mode_badge.text()
        assert "rgba(0, 220, 255, 225)" in window.viewer.manual_mode_badge.styleSheet()
        assert "Shift+W/A/S/D" in window.add_point_panel.status_label.text()

        before = window.session.localized_by_frame[0].shape[0]
        scene_x, scene_y = window.viewer._pala_to_display_xy(12.4, 13.6)
        QTest.mouseClick(
            window.viewer.viewport(),
            Qt.LeftButton,
            pos=window.viewer.mapFromScene(scene_x, scene_y),
        )
        app.processEvents()

        assert window.session.localized_by_frame[0].shape[0] == before
        draft = window.viewer.candidate_rows
        assert draft.shape == (1, 4)
        assert "MANUAL DRAFT PENDING" in window.viewer.manual_mode_badge.text()
        assert "rgba(245, 158, 11, 235)" in window.viewer.manual_mode_badge.styleSheet()
        initial_z, initial_x = draft[0, 1:3]

        QTest.keyClick(window.viewer, Qt.Key_D, Qt.ShiftModifier)
        QTest.keyClick(window.viewer, Qt.Key_W, Qt.ShiftModifier)
        app.processEvents()
        adjusted = window.viewer.candidate_rows[0]
        assert adjusted[1] == pytest.approx(initial_z - MANUAL_NUDGE_STEP_PX)
        assert adjusted[2] == pytest.approx(initial_x + MANUAL_NUDGE_STEP_PX)
        assert window.correction_mode == "add_point"

        QTest.keyClick(window.viewer, Qt.Key_Backspace)
        app.processEvents()
        assert window.viewer.candidate_rows.shape == (0, 4)
        assert "MANUAL PLACEMENT ACTIVE" in window.viewer.manual_mode_badge.text()
        assert window.session.localized_by_frame[0].shape[0] == before

        QTest.mouseClick(
            window.viewer.viewport(),
            Qt.LeftButton,
            pos=window.viewer.mapFromScene(scene_x, scene_y),
        )
        QTest.keyClick(window.viewer, Qt.Key_S, Qt.ShiftModifier)
        app.processEvents()
        confirmed = window.viewer.candidate_rows[0].copy()
        QTest.keyClick(window.viewer, Qt.Key_Return)
        app.processEvents()

        assert window.session.localized_by_frame[0].shape[0] == before + 1
        row = window.session.localized_by_frame[0][-1]
        assert row[0] == pytest.approx(confirmed[0])
        assert row[1] == pytest.approx(confirmed[1])
        assert row[2] == pytest.approx(confirmed[2])
        assert window.session.local_idx_for_track_frame(0, 0) == before
        append_edit = next(
            edit
            for edit in reversed(window.session.edit_log)
            if edit.action == "append_manual_detection"
        )
        assert append_edit.payload["source"] == "manual_placement"
        assert append_edit.payload["params"] == {
            "method": MANUAL_PLACEMENT_METHOD,
            "nudgeStepPx": MANUAL_NUDGE_STEP_PX,
        }
        assert window.viewer.candidate_rows.shape == (0, 4)
        assert window.correction_mode == "add_point"
        assert "MANUAL PLACEMENT ACTIVE" in window.viewer.manual_mode_badge.text()
        assert "Click the image" in window.add_point_panel.status_label.text()

        window.set_frame_1based(2)
        assert not window._add_point_refresh_timer.isActive()
        assert window.viewer.candidate_rows.shape == (0, 4)
        assert "Frame 2" in window.add_point_panel.status_label.text()
    finally:
        window.close()


def test_add_point_eraser_discards_unassigned_added_point_and_undo_restores_it(
    app,
) -> None:
    window = make_window(app)
    try:
        local_idx = window.session.append_manual_detection(
            0,
            intensity=7.0,
            z=15.0,
            x=18.0,
            track_id=0,
        )
        row_count = window.session.localized_by_frame[0].shape[0]
        window.viewer.render()
        scene_x, scene_y = window.viewer._pala_to_display_xy(15.0, 18.0)
        view_pos = window.viewer.mapFromScene(scene_x, scene_y)

        window.mode_buttons["remove"].setChecked(True)
        QTest.mouseClick(window.viewer.viewport(), Qt.LeftButton, pos=view_pos)
        app.processEvents()
        assert window.session.track_id_for_detection(0, local_idx) is None

        window.add_point_panel.eraser_button.click()
        app.processEvents()
        assert window.correction_mode == "add_point"
        assert window.add_point_panel.eraser_active()
        assert window.viewer._manual_detection_eraser_enabled
        assert window.viewer.candidate_rows.shape == (0, 4)
        assert "assigned points are protected" in (
            window.add_point_panel.status_label.text()
        )

        view_pos = window.viewer.mapFromScene(scene_x, scene_y)
        QTest.mouseClick(window.viewer.viewport(), Qt.LeftButton, pos=view_pos)
        app.processEvents()
        assert window.session.localized_by_frame[0].shape[0] == row_count
        assert window.session.is_discarded_manual_detection(0, local_idx)
        assert window.session.edit_log[-1].action == (
            "set_manual_detection_discarded"
        )
        visible_indices = {
            int(item.data(2))
            for item in window.viewer.scene.items()
            if item.data(0) == "localization"
        }
        assert local_idx not in visible_indices
        assert "omitted from saved data" in window.status_label.text()

        window.undo()
        assert not window.session.is_discarded_manual_detection(0, local_idx)
        visible_indices = {
            int(item.data(2))
            for item in window.viewer.scene.items()
            if item.data(0) == "localization"
        }
        assert local_idx in visible_indices

        window.redo()
        assert window.session.is_discarded_manual_detection(0, local_idx)
    finally:
        window.close()
