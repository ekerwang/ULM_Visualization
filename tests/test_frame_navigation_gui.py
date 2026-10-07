import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui.main_window_ui import (
    SCOPE_FULL_STACK,
    SCOPE_TRACK_PADDED,
    SCOPE_TRACK_RANGE,
    SCOPE_SHORTCUTS,
    FrameScopeSlider,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session() -> CorrectionSession:
    n_frames = 240
    track = np.full(n_frames, np.nan)
    next_track = np.full(n_frames, np.nan)
    # 1-based frames 51--150 form a 100-frame span with 98 gap frames.
    track[50] = 0.0
    track[149] = 0.0
    next_track[180] = 0.0
    next_track[199] = 0.0
    return CorrectionSession(
        localized_by_frame={
            50: np.asarray([[1.0, 1.0, 1.0, 51.0]]),
            149: np.asarray([[1.0, 2.0, 2.0, 150.0]]),
            180: np.asarray([[1.0, 3.0, 3.0, 181.0]]),
            199: np.asarray([[1.0, 4.0, 4.0, 200.0]]),
        },
        tracks=[track, next_track],
        image_stack=np.ones((3, 4, n_frames), dtype=np.float32),
    )


@pytest.fixture
def window(app):
    win = MainWindow()
    win.session = make_session()
    win.viewer.set_session(win.session)
    win.populate_track_list()
    win.configure_frame_controls()
    win._sync_enabled_state()
    yield win
    win.close()


def test_scope_switch_is_exclusive_and_track_ranges_use_full_timeline(window, app):
    assert window.playback_scope == SCOPE_FULL_STACK
    assert window.fps_spin.value() == 10
    assert window._playback_range() == (1, 240)
    assert window.scope_stack.count() == 3
    assert window.scope_stack.currentWidget() is window.scope_sliders[SCOPE_FULL_STACK]
    assert window.scope_sliders[SCOPE_FULL_STACK].isEnabled()
    assert not window.scope_buttons[SCOPE_TRACK_RANGE].isEnabled()
    assert not window.scope_buttons[SCOPE_TRACK_PADDED].isEnabled()
    assert not window.scope_actions[SCOPE_TRACK_RANGE].isEnabled()
    assert not window.scope_actions[SCOPE_TRACK_PADDED].isEnabled()
    assert window.padding_controls.isHidden()

    window.show()
    app.processEvents()
    button_widths = [button.width() for button in window.scope_buttons.values()]
    assert max(button_widths) - min(button_widths) <= 1

    window.track_list.setCurrentRow(0)
    app.processEvents()

    assert window.frame_spin.value() == 51
    assert window.scope_buttons[SCOPE_TRACK_RANGE].isEnabled()
    assert window.scope_buttons[SCOPE_TRACK_PADDED].isEnabled()
    assert window.scope_actions[SCOPE_TRACK_RANGE].isEnabled()
    assert window.scope_actions[SCOPE_TRACK_PADDED].isEnabled()
    assert not window.scope_sliders[SCOPE_TRACK_RANGE].isEnabled()
    assert window.scope_sliders[SCOPE_TRACK_RANGE].minimum() == 51
    assert window.scope_sliders[SCOPE_TRACK_RANGE].maximum() == 150

    window.scope_buttons[SCOPE_TRACK_RANGE].click()
    app.processEvents()

    assert window.playback_scope == SCOPE_TRACK_RANGE
    assert window.frame_slider is window.scope_sliders[SCOPE_TRACK_RANGE]
    assert window.scope_stack.currentWidget() is window.frame_slider
    assert window.scope_sliders[SCOPE_TRACK_RANGE].isEnabled()
    assert not window.scope_sliders[SCOPE_FULL_STACK].isEnabled()
    assert not window.scope_sliders[SCOPE_TRACK_PADDED].isEnabled()
    assert (window.frame_spin.minimum(), window.frame_spin.maximum()) == (51, 150)

    window.scope_sliders[SCOPE_TRACK_RANGE].setValue(70)
    app.processEvents()
    assert window.frame_spin.value() == 70
    assert window.viewer.frame_idx == 69
    assert window.scope_context_label.text() == "20 / 100 in track range"


def test_padded_scope_is_proportional_clipped_and_loops(window, app):
    window.track_list.setCurrentRow(0)
    window.scope_buttons[SCOPE_TRACK_PADDED].click()
    app.processEvents()

    slider = window.scope_sliders[SCOPE_TRACK_PADDED]
    assert window._playback_range() == (41, 160)
    assert slider.track_span == (51, 150)
    assert slider.segment_frame_counts() == (10, 100, 10)
    assert slider.isEnabled()
    assert window.scope_stack.currentWidget() is slider
    assert window.pad_spin.isEnabled()
    assert not window.padding_controls.isHidden()

    window.pad_spin.setValue(20)
    app.processEvents()
    assert window._playback_range() == (31, 170)
    assert slider.segment_frame_counts() == (20, 100, 20)

    slider.setValue(40)
    assert window.scope_context_label.text() == (
        "Track frames 51–150  ·  before track"
    )
    slider.setValue(slider.maximum())
    window._on_play_tick()
    assert window.frame_spin.value() == slider.minimum()

    # Switching to the finer track-only range clamps to its nearest boundary.
    window.scope_buttons[SCOPE_TRACK_RANGE].click()
    assert window.frame_spin.value() == 51
    assert window.padding_controls.isHidden()
    window.scope_buttons[SCOPE_FULL_STACK].click()
    assert window.frame_spin.value() == 51
    assert window.scope_context_label.text() == "of 240"


def test_qwe_shortcuts_switch_scope_without_toggling_layer_slots(window, app):
    window.track_list.setCurrentRow(0)
    window.show()
    window.activateWindow()
    window.viewer.setFocus()
    app.processEvents()

    assert window.scope_buttons[SCOPE_FULL_STACK].text().endswith(
        f"({_native_text(SCOPE_SHORTCUTS[SCOPE_FULL_STACK])})"
    )
    assert window.scope_buttons[SCOPE_TRACK_RANGE].text().endswith(
        f"({_native_text(SCOPE_SHORTCUTS[SCOPE_TRACK_RANGE])})"
    )
    assert window.scope_buttons[SCOPE_TRACK_PADDED].text().endswith(
        f"({_native_text(SCOPE_SHORTCUTS[SCOPE_TRACK_PADDED])})"
    )

    slot_two = window.display_panel.slot(2)
    assert slot_two.visibility_button.isEnabled()
    slot_two_visible = slot_two.group.isChecked()
    QTest.keyClick(window.viewer, Qt.Key_W)
    app.processEvents()
    assert window.playback_scope == SCOPE_TRACK_RANGE
    assert slot_two.group.isChecked() is slot_two_visible

    QTest.keyClick(window.viewer, Qt.Key_E)
    app.processEvents()
    assert window.playback_scope == SCOPE_TRACK_PADDED

    QTest.keyClick(window.viewer, Qt.Key_Q)
    app.processEvents()
    assert window.playback_scope == SCOPE_FULL_STACK

    QTest.keyClick(window.viewer, Qt.Key_2)
    app.processEvents()
    assert slot_two.group.isChecked() is not slot_two_visible


def test_clearing_track_selection_falls_back_to_full_stack(window, app):
    window.track_list.setCurrentRow(0)
    window.scope_buttons[SCOPE_TRACK_RANGE].click()
    assert window.playback_scope == SCOPE_TRACK_RANGE

    window.clear_track_selection()
    app.processEvents()

    assert window.playback_scope == SCOPE_FULL_STACK
    assert window.scope_buttons[SCOPE_FULL_STACK].isChecked()
    assert window._playback_range() == (1, 240)
    assert not window.scope_buttons[SCOPE_TRACK_RANGE].isEnabled()
    assert not window.scope_buttons[SCOPE_TRACK_PADDED].isEnabled()
    assert not window.scope_actions[SCOPE_TRACK_RANGE].isEnabled()
    assert not window.scope_actions[SCOPE_TRACK_PADDED].isEnabled()
    assert window.scope_stack.currentWidget() is window.scope_sliders[SCOPE_FULL_STACK]


@pytest.mark.parametrize(
    ("scope", "expected_range"),
    [
        (SCOPE_TRACK_RANGE, (181, 200)),
        (SCOPE_TRACK_PADDED, (171, 210)),
    ],
)
@pytest.mark.parametrize(
    ("filter_name", "initial_status", "status_action"),
    [
        ("Unverified", None, "mark_selected_verified"),
        ("verified", "verified", "flag_selected"),
    ],
)
def test_filtered_status_change_advances_track_without_leaving_scope(
    window,
    app,
    scope,
    expected_range,
    filter_name,
    initial_status,
    status_action,
):
    if initial_status is not None:
        for track_id in window.session.active_track_ids:
            window.session.set_track_status(track_id, initial_status)
        window.populate_track_list()
    window.filter_combo.setCurrentText(filter_name)
    window.track_list.setCurrentRow(0)
    completed_track_id = window.selected_track_id
    next_track_id = int(window.track_list.item(1).data(Qt.UserRole))
    window.scope_buttons[scope].click()
    assert window.playback_scope == scope

    getattr(window, status_action)()
    app.processEvents()

    assert completed_track_id not in window._row_by_track_id
    assert window.selected_track_id == next_track_id
    assert int(window.track_list.currentItem().data(Qt.UserRole)) == next_track_id
    assert window.playback_scope == scope
    assert window.scope_buttons[scope].isChecked()
    assert window._playback_range() == expected_range
    assert f"Track {completed_track_id}" in window.status_label.text()

    getattr(window, status_action)()
    app.processEvents()

    assert window.track_list.count() == 0
    assert window.selected_track_id is None
    assert window.playback_scope == SCOPE_FULL_STACK


def test_padding_counts_reflect_stack_boundary_clipping(app):
    slider = FrameScopeSlider()
    slider.setRange(1, 110)
    slider.set_track_span(1, 100)
    assert slider.segment_frame_counts() == (0, 100, 10)

    slider.setRange(91, 200)
    slider.set_track_span(101, 200)
    assert slider.segment_frame_counts() == (10, 100, 0)


def _native_text(shortcut: str) -> str:
    return QKeySequence(shortcut).toString(QKeySequence.NativeText)
