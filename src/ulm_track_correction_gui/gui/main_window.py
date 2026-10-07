"""Main window for the ULM track-correction application.

The window is assembled from responsibility-scoped mixins so each concern
lives in its own module:

- ``main_window_ui.py``        widgets, layout, menus
- ``main_window_datasetlist.py`` persistent dataset queue and progress workflow
- ``main_window_session.py``   MAT open/save, autosave edit-log wiring
- ``main_window_editing.py``   click tools, dialogs, undo/redo, status marks
- ``main_window_tracklist.py`` track list population/selection/filtering
- ``main_window_playback.py``  frame navigation and playback

This module keeps only window-wide state, enabled-state sync, and global
key handling.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QTextEdit,
)

from ulm_track_correction_gui.core.accumulation_checkpoint import AccumulationCheckpoint
from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.dataset_queue import DatasetQueue
from ulm_track_correction_gui.gui.history_dialog import TrackHistoryDialog
from ulm_track_correction_gui.gui.main_window_datasetlist import DatasetListMixin
from ulm_track_correction_gui.gui.main_window_editing import EditingMixin
from ulm_track_correction_gui.gui.main_window_playback import PlaybackMixin
from ulm_track_correction_gui.gui.main_window_session import SessionIoMixin
from ulm_track_correction_gui.gui.main_window_tracklist import TrackListMixin
from ulm_track_correction_gui.gui.main_window_ui import WindowUiMixin
from ulm_track_correction_gui.io.editlog import EditLogLock, EditLogWriter


class MainWindow(
    WindowUiMixin,
    DatasetListMixin,
    SessionIoMixin,
    EditingMixin,
    TrackListMixin,
    PlaybackMixin,
    QMainWindow,
):
    """A minimal, track-centric GUI shell.

    Editing modes should build on this class rather than reintroducing the old
    parameter-tuning GUI layout. Keyboard-first navigation: Left/Right step
    frames, Up/Down step tracks, V = verify, X = flag, 1--8 = toggle the
    current occupants of fixed display slots, and F = fit view.
    """

    def __init__(self, *, dataset_settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("ULM Track Correction")
        self.resize(1540, 820)

        self.session: CorrectionSession | None = None
        self.current_path: Path | None = None
        self.selected_track_id: int | None = None
        self.editlog_writer: EditLogWriter | None = None
        self._editlog_lock: EditLogLock | None = None
        self._editlog_baseline_fingerprint: str | None = None
        self._autosave_blocked = False
        self.undo_manager: UndoManager | None = None
        self.accumulation_checkpoint: AccumulationCheckpoint | None = None
        self.correction_mode = "inspect"
        self._movie_export_in_progress = False
        self._movie_watermark_preview_suppressed = False
        self._mode_button_was_active_on_press = False
        self._row_by_track_id: dict[int, int] = {}
        self._suppress_frame_jump = False
        self._history_dialog: TrackHistoryDialog | None = None
        self._add_point_used_params: dict = {}
        self._add_point_candidate_frame: int | None = None
        self._add_point_refresh_timer = QTimer(self)
        self._add_point_refresh_timer.setSingleShot(True)
        self._add_point_refresh_timer.setInterval(180)
        self._add_point_refresh_timer.timeout.connect(
            self.refresh_add_point_candidates
        )

        self.dataset_settings = (
            dataset_settings
            if dataset_settings is not None
            else QSettings("ULMTrackCorrection", "ULMTrackCorrectionGUI")
        )
        self.dataset_queue = DatasetQueue()
        self._current_dataset_path: str | None = None
        self._session_dirty = False
        self._active_track_ids: set[int] | None = None
        self._verified_track_ids: set[int] | None = None
        self._dataset_row_by_path: dict[str, int] = {}
        self._dataset_persist_timer = QTimer(self)
        self._dataset_persist_timer.setSingleShot(True)
        self._dataset_persist_timer.timeout.connect(self.persist_dataset_queue)

        self._build_widgets()
        self._build_layout()
        self._build_menu()
        self._slot_shortcut_filter_installed = False
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._slot_shortcut_filter_installed = True
        self.restore_dataset_queue()
        self.populate_dataset_list()
        self._sync_enabled_state()

    def _sync_enabled_state(self) -> None:
        has_session = self.session is not None
        can_edit = has_session and not self._autosave_blocked
        if not can_edit and self.correction_mode != "inspect":
            # A persistent tool must not leave the viewer accepting edits when
            # autosave safety has disabled the editing controls.
            self.mode_buttons["inspect"].setChecked(True)
        selected_active = (
            has_session
            and self.selected_track_id is not None
            and 0 <= self.selected_track_id < self.session.n_tracks
            and self.session.is_track_active(self.selected_track_id)
        )
        selected_points = (
            self.session.track_summary(self.selected_track_id)["n_points"]
            if selected_active
            else 0
        )
        active_nonempty_count = (
            len(self.session.active_nonempty_track_ids) if has_session else 0
        )
        self.track_list.setEnabled(has_session)
        self.sort_combo.setEnabled(has_session)
        self.filter_combo.setEnabled(has_session)
        self.refresh_frame_scope_controls()
        self.frame_spin.setEnabled(has_session)
        self.mark_verified_button.setEnabled(
            can_edit and selected_active and selected_points > 0
        )
        self.flag_button.setEnabled(can_edit and selected_active)
        self.history_button.setEnabled(selected_active)
        self.play_button.setEnabled(has_session)
        self.export_movie_button.setEnabled(
            has_session and not self._movie_export_in_progress
        )
        self.watermark_preview_checkbox.setEnabled(
            has_session and not self._movie_export_in_progress
        )
        self.watermark_font_size_label.setEnabled(
            has_session and not self._movie_export_in_progress
        )
        self.watermark_font_size_spin.setEnabled(
            has_session and not self._movie_export_in_progress
        )
        self.fps_spin.setEnabled(has_session)
        can_add_point = (
            can_edit
            and self.session is not None
            and self.session.image_stack is not None
        )
        self.add_point_button.setEnabled(can_add_point)
        self.add_point_panel.setEnabled(can_add_point)
        self.initialize_track_button.setEnabled(can_edit)
        self.merge_button.setEnabled(can_edit and active_nonempty_count >= 2)
        self.split_button.setEnabled(
            can_edit and selected_active and selected_points >= 2
        )
        self.delete_track_button.setEnabled(can_edit and selected_active)
        self.initialize_track_action.setEnabled(can_edit)
        self.split_track_action.setEnabled(
            can_edit and selected_active and selected_points >= 2
        )
        self.delete_track_action.setEnabled(can_edit and selected_active)
        for mode, button in self.mode_buttons.items():
            button.setEnabled(can_add_point if mode == "add_point" else can_edit)
        self.add_point_action.setEnabled(can_add_point)
        can_undo = (
            can_edit and self.undo_manager is not None and self.undo_manager.can_undo
        )
        can_redo = (
            can_edit and self.undo_manager is not None and self.undo_manager.can_redo
        )
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)
        self.load_checkpoint_action.setEnabled(has_session)
        self._sync_dataset_controls()
        self.refresh_movie_watermark_preview()

    def set_correction_mode(self, mode: str) -> None:
        if mode == "add_point" and (
            self.session is None or self.session.image_stack is None
        ):
            self.mode_buttons["inspect"].setChecked(True)
            self.status_label.setText(
                "Add Point needs the raw bubble movie in the current session."
            )
            return
        was_add_point = self.correction_mode == "add_point"
        self.correction_mode = mode
        self.viewer.set_interaction_mode(mode)
        if mode == "add_point":
            self.stop_playback()
            self.refresh_add_point_candidates()
        else:
            self._add_point_refresh_timer.stop()
            self._add_point_candidate_frame = None
            self._add_point_used_params = {}
            self.viewer.set_manual_candidate_placement_enabled(False)
            self.viewer.set_manual_detection_eraser_enabled(False)
            self.viewer.clear_candidate_overlay()
            if was_add_point:
                self.add_point_panel.set_eraser_active(False, notify=False)
                self.add_point_panel.set_status(
                    "Ready · enter Add Point to localize the current frame."
                )
            if self.session is not None:
                self.status_label.setText(f"Mode: {mode}.")

    def toggle_correction_mode(self, mode: str) -> None:
        """Activate a tool, or return to Inspect when its shortcut repeats."""

        if mode not in self.mode_buttons:
            raise ValueError(f"Unknown correction mode: {mode}")
        target_mode = (
            "inspect"
            if mode != "inspect" and self.correction_mode == mode
            else mode
        )
        self.mode_buttons[target_mode].setChecked(True)

    def remember_correction_mode_button_press(self, mode: str) -> None:
        """Remember whether a state button was already active when pressed."""

        self._mode_button_was_active_on_press = self.correction_mode == mode

    def finish_correction_mode_button_click(self, mode: str) -> None:
        """Let a second click on an active edit-tool button return to Inspect.

        Qt keeps the checked button in an exclusive group checked when it is
        clicked again. Remembering the pre-click state gives the buttons the
        same toggle behavior as repeating their A/R/M shortcuts.
        """

        was_active = self._mode_button_was_active_on_press
        self._mode_button_was_active_on_press = False
        if mode != "inspect" and was_active:
            self.mode_buttons["inspect"].setChecked(True)

    def keyPressEvent(self, event) -> None:
        if self.session is not None:
            key = event.key()
            if key == Qt.Key_Left:
                self.step_frame(-1)
                return
            if key == Qt.Key_Right:
                self.step_frame(1)
                return
            if key == Qt.Key_Up:
                self.step_track(-1)
                return
            if key == Qt.Key_Down:
                self.step_track(1)
                return
        super().keyPressEvent(event)

    @staticmethod
    def _keyboard_focus_accepts_text_input() -> bool:
        focus = QApplication.focusWidget()
        return isinstance(
            focus,
            (QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit),
        ) or (isinstance(focus, QComboBox) and focus.isEditable())

    def eventFilter(self, watched, event) -> bool:
        """Route Add Point and 1--8 context keys without stealing text entry."""

        is_active_keypress = (
            event.type() == QEvent.KeyPress and QApplication.activeWindow() is self
        )
        if is_active_keypress:
            if self.handle_manual_candidate_key(event):
                return True
            method_steps = {
                Qt.Key_Comma: -1,
                Qt.Key_Period: 1,
            }
            if (
                event.key() in method_steps
                and event.modifiers() == Qt.NoModifier
                and not event.isAutoRepeat()
                and not self._keyboard_focus_accepts_text_input()
                and self.cycle_add_point_method(method_steps[event.key()])
            ):
                return True

        digit_keys = (
            Qt.Key_1,
            Qt.Key_2,
            Qt.Key_3,
            Qt.Key_4,
            Qt.Key_5,
            Qt.Key_6,
            Qt.Key_7,
            Qt.Key_8,
        )
        if (
            is_active_keypress
            and event.key() in digit_keys
            and not event.isAutoRepeat()
            and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier)
        ):
            if not self._keyboard_focus_accepts_text_input():
                self.display_panel.toggle_slot(digit_keys.index(event.key()) + 1)
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        if self._dataset_persist_timer.isActive():
            self._dataset_persist_timer.stop()
            self.persist_dataset_queue()
        self._close_editlog()
        if self._slot_shortcut_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._slot_shortcut_filter_installed = False
        super().closeEvent(event)
