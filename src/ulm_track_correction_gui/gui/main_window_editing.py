"""Correction editing behavior of the main window.

Everything here mutates the session exclusively through ``CorrectionSession``
methods so each edit lands in the edit log (autosave) and the undo stack via
``_on_session_edit`` (the single dispatch point).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_FLAGGED,
    TRACK_STATUS_VERIFIED,
    CorrectionEdit,
    DetectionAssignmentConflict,
    new_group_id,
)
from ulm_track_correction_gui.core.manual_localization import sample_manual_intensity
from ulm_track_correction_gui.core.pala_contracts import (
    ensure_mat_tracking,
    find_matching_detection,
)
from ulm_track_correction_gui.gui.add_point_panel import (
    MANUAL_NUDGE_STEP_PX,
    MANUAL_PLACEMENT_METHOD,
)
from ulm_track_correction_gui.gui.confirmation_dialog import ask_yes_no
from ulm_track_correction_gui.gui.history_dialog import TrackHistoryDialog
from ulm_track_correction_gui.gui.merge_dialog import MergeTracksDialog
from ulm_track_correction_gui.gui.split_dialog import SplitTrackDialog
from ulm_track_correction_gui.reference_processing.wrappers import localization_wrapper


STRUCTURAL_EDIT_ACTIONS = {
    "initialize_track",
    "remove_initialized_track",
    "set_track_active",
}


def _deleted_track_id_in_group(edits: list[CorrectionEdit]) -> int | None:
    """Identify a Delete Track group without confusing merge or split retirement."""

    track_ids = {edit.track_id for edit in edits if edit.track_id is not None}
    retired_ids = {
        edit.track_id
        for edit in edits
        if edit.action == "set_track_active"
        and edit.payload.get("before") is True
        and edit.payload.get("after") is False
    }
    if len(track_ids) == 1 and retired_ids == track_ids:
        return next(iter(track_ids))
    return None


class EditingMixin:
    """Click tools, dialogs, undo/redo, and status marking."""

    def _on_session_edit(self, edit: CorrectionEdit) -> None:
        """Single dispatch point for every session edit (autosave + undo)."""

        if self.undo_manager is not None:
            self.undo_manager.record(edit)
        self.on_dataset_session_edit(edit)
        if self.editlog_writer is not None:
            try:
                self.editlog_writer.append(edit)
            except Exception as exc:
                failed_writer = self.editlog_writer
                self.editlog_writer = None
                try:
                    failed_writer.close()
                except Exception:
                    pass
                self._autosave_blocked = True
                self._sync_enabled_state()
                QMessageBox.critical(
                    self,
                    "Autosave failed; editing disabled",
                    "The edit was applied in memory and remains undoable, but it "
                    "could not be fsynced to the sidecar. Further editing is "
                    "disabled. Save an atomic snapshot to a different destination "
                    f"and resolve the storage error before continuing.\n\n{exc}",
                )

    def _refresh_after_edit(self, track_ids) -> None:
        self.viewer.invalidate_track_geometry(track_ids)
        if self.session is not None:
            self.viewer.set_checkpoint_track_id_map(
                self.session.current_to_source_track_ids()
            )
        self.refresh_frame_scope_controls()
        self.viewer.render()
        for track_id in {t for t in track_ids if t is not None}:
            if 0 <= track_id < self.session.n_tracks:
                self._update_track_item(track_id)
        self._sync_enabled_state()

    def _refresh_after_structural_edit(
        self,
        track_ids,
        *,
        preferred_track_id: int | None = None,
        select_first_visible_if_none: bool = False,
    ) -> None:
        """Rebuild identity-dependent GUI state after create/retire operations."""

        if self.session is None:
            return
        affected = {int(track_id) for track_id in track_ids if track_id is not None}
        self.viewer.invalidate_track_geometry(affected)
        self.viewer.set_checkpoint_track_id_map(
            self.session.current_to_source_track_ids()
        )

        candidates = [
            track_id
            for track_id in affected
            if 0 <= track_id < self.session.n_tracks
            and self.session.is_track_active(track_id)
        ]
        current = self.selected_track_id
        if preferred_track_id in candidates:
            selected = preferred_track_id
        elif (
            current is not None
            and 0 <= current < self.session.n_tracks
            and self.session.is_track_active(current)
        ):
            selected = current
        else:
            frame_idx = self.frame_spin.value() - 1
            selected = next(
                (
                    track_id
                    for track_id in sorted(candidates)
                    if self.session.local_idx_for_track_frame(track_id, frame_idx)
                    is not None
                ),
                min(candidates) if candidates else None,
            )

        self.selected_track_id = selected
        self.populate_track_list()
        if selected is None and select_first_visible_if_none and self.track_list.count():
            first_item = self.track_list.item(0)
            selected = int(first_item.data(Qt.UserRole))
            self.selected_track_id = selected
        signals_were_blocked = self.track_list.blockSignals(True)
        try:
            row = self._row_by_track_id.get(selected, -1)
            self.track_list.setCurrentRow(row)
            if row < 0:
                self.track_list.clearSelection()
        finally:
            self.track_list.blockSignals(signals_were_blocked)
        self.viewer.set_selected_track(selected)
        self.refresh_frame_scope_controls()
        self.viewer.render()
        self._sync_enabled_state()

    def _confirm(self, title: str, text: str) -> bool:
        return ask_yes_no(self, title, text)

    def on_detection_clicked(self, frame_idx: int, local_idx: int) -> None:
        if self.session is None:
            return
        mode = self.correction_mode
        if mode == "assign":
            self._assign_clicked_detection(frame_idx, local_idx)
        elif mode == "remove":
            self._remove_clicked_detection(frame_idx, local_idx)
        else:
            return

    def _is_track_verified(self, track_id: int) -> bool:
        return (
            self.session is not None
            and self.session.track_status.get(track_id) == TRACK_STATUS_VERIFIED
        )

    def _flag_demoted_track(self, track_id: int, group_id: str, reason: str) -> None:
        """Verified tracks whose points were taken/removed lose their status.

        Sharing the caller's group_id keeps the demotion inside the same undo
        group as the edit that caused it.
        """

        self.session.set_track_status(
            track_id,
            TRACK_STATUS_FLAGGED,
            note=f"Auto-flagged: was verified, then {reason}",
            group_id=group_id,
        )

    def _steal_detection(
        self,
        track_id: int,
        frame_idx: int,
        local_idx: int,
        owner_track_id: int,
    ) -> None:
        """Steal a detection after user confirmation, demoting a verified owner."""

        owner_was_verified = self._is_track_verified(owner_track_id)
        target_was_verified = self._is_track_verified(track_id)
        gid = new_group_id()
        self.session.assign_detection(
            track_id, frame_idx, local_idx, steal=True, group_id=gid
        )
        if owner_was_verified:
            self._flag_demoted_track(
                owner_track_id,
                gid,
                f"its frame-{frame_idx + 1} point was stolen by track {track_id}.",
            )
        if target_was_verified:
            self._flag_demoted_track(
                track_id,
                gid,
                f"its frame-{frame_idx + 1} assignment was changed by a steal.",
            )
        self._refresh_after_edit([track_id, owner_track_id])
        demoted = " (now flagged for re-review)" if owner_was_verified else ""
        self.status_label.setText(
            f"Stole detection from track {owner_track_id}{demoted} for track "
            f"{track_id} at frame {frame_idx + 1}."
        )

    def _assign_clicked_detection(self, frame_idx: int, local_idx: int) -> None:
        if self.selected_track_id is None:
            self.status_label.setText("Select a track before assigning.")
            return
        track_id = self.selected_track_id
        target_was_verified = self._is_track_verified(track_id)
        current_local_idx = self.session.local_idx_for_track_frame(track_id, frame_idx)
        assignment_changes = current_local_idx != local_idx
        if (
            target_was_verified
            and assignment_changes
            and not self._confirm(
                "Modify a VERIFIED track?",
                f"Track {track_id} is VERIFIED. Changing its frame-{frame_idx + 1} "
                "assignment will FLAG it and require a complete re-review. Continue?",
            )
        ):
            return
        gid = new_group_id()
        try:
            self.session.assign_detection(
                track_id,
                frame_idx,
                local_idx,
                group_id=gid,
            )
        except DetectionAssignmentConflict as exc:
            if self._is_track_verified(exc.owner_track_id):
                title = "Conflict with a VERIFIED track"
                text = (
                    f"This detection belongs to track {exc.owner_track_id}, "
                    f"which is already VERIFIED.\n\n"
                    f"Steal it for track {track_id}? Track "
                    f"{exc.owner_track_id} will be FLAGGED and must be "
                    f"reviewed again from scratch."
                )
            else:
                title = "Detection already assigned"
                text = (
                    f"This detection belongs to track {exc.owner_track_id}.\n"
                    f"Steal it for track {track_id}?"
                )
            if not self._confirm(title, text):
                return
            self._steal_detection(track_id, frame_idx, local_idx, exc.owner_track_id)
            return
        if target_was_verified and assignment_changes:
            self._flag_demoted_track(
                track_id,
                gid,
                f"its frame-{frame_idx + 1} assignment was changed.",
            )
        self._refresh_after_edit([track_id])
        flagged = (
            " Track is now FLAGGED for re-review."
            if target_was_verified and assignment_changes
            else ""
        )
        self.status_label.setText(
            f"Assigned detection {local_idx} to track {track_id} at frame "
            f"{frame_idx + 1}.{flagged}"
        )

    def _remove_clicked_detection(self, frame_idx: int, local_idx: int) -> None:
        owner = self.session.track_id_for_detection(frame_idx, local_idx)
        if owner is None:
            self.status_label.setText("This detection is not assigned to any track.")
            return
        owner_was_verified = self._is_track_verified(owner)
        if owner_was_verified:
            title = "Remove a VERIFIED track point?"
            text = (
                f"This detection belongs to VERIFIED track {owner}.\n\n"
                "Remove its assignment anyway? The track will be FLAGGED and "
                "must be reviewed again from scratch."
            )
            if not self._confirm(title, text):
                return
        elif owner != self.selected_track_id:
            title = "Remove other track's point?"
            text = (
                f"This detection belongs to track {owner} (not the "
                "selected track). Remove its assignment anyway?"
            )
            if not self._confirm(title, text):
                return
        gid = new_group_id()
        self.session.remove_assignment(owner, frame_idx, group_id=gid)
        if owner_was_verified:
            self._flag_demoted_track(
                owner,
                gid,
                f"its frame-{frame_idx + 1} assignment was removed.",
            )
        self._refresh_after_edit([owner])
        demoted = " (now flagged for re-review)" if owner_was_verified else ""
        self.status_label.setText(
            f"Removed track {owner}'s assignment at frame {frame_idx + 1}{demoted}."
        )

    # ---------------------------------------------------------- Add Point

    def cycle_add_point_method(self, step: int) -> bool:
        """Select the previous or next Add Point method, wrapping at the ends."""

        if step not in (-1, 1):
            raise ValueError("Add Point method step must be -1 or 1.")
        combo = self.add_point_panel.method_combo
        if self.correction_mode != "add_point" or not combo.isEnabled():
            return False
        count = combo.count()
        if count < 2:
            return False
        combo.setCurrentIndex((combo.currentIndex() + step) % count)
        return True

    def initialize_new_track(self) -> None:
        if self.session is None:
            return
        self.mode_buttons["inspect"].setChecked(True)
        track_id = self.session.initialize_track()
        self._refresh_after_structural_edit([track_id], preferred_track_id=track_id)
        self.status_label.setText(
            f"Initialized active empty track {track_id}. Use Assign or Add Point "
            "to add its first detection."
        )

    def schedule_add_point_localization(self, status: str | None = None) -> None:
        """Debounce parameter edits while the automatic Add Point tool is active."""

        if self.correction_mode != "add_point":
            return
        self._add_point_candidate_frame = None
        self._add_point_used_params = {}
        self.viewer.clear_candidate_overlay()
        if self._is_add_point_eraser_active():
            self._prepare_add_point_eraser()
            return
        if self._is_manual_add_point_method():
            self._prepare_manual_candidate_placement()
            return
        self.viewer.set_manual_candidate_placement_enabled(False)
        self.viewer.set_manual_detection_eraser_enabled(False)
        self.add_point_panel.set_status(
            status or "Parameters changed · refreshing candidates…"
        )
        self._add_point_refresh_timer.start()

    def refresh_add_point_candidates(self) -> None:
        """Localize the raw current frame and publish candidates to the viewer."""

        if self.correction_mode != "add_point" or self.session is None:
            return
        self._add_point_refresh_timer.stop()
        stack = self.session.image_stack
        if stack is None:
            return
        frame_idx = min(self.frame_spin.value() - 1, stack.shape[2] - 1)
        params = self.add_point_panel.current_params()
        if self._is_add_point_eraser_active():
            self._prepare_add_point_eraser()
            return
        if self._is_manual_add_point_method():
            self._prepare_manual_candidate_placement()
            return
        self.viewer.set_manual_candidate_placement_enabled(False)
        self.viewer.set_manual_detection_eraser_enabled(False)
        self.add_point_panel.set_status(
            f"Localizing raw frame {frame_idx + 1}…"
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            candidates = ensure_mat_tracking(
                localization_wrapper(stack[:, :, frame_idx], params)
            )
        except Exception as exc:
            self._add_point_candidate_frame = None
            self._add_point_used_params = {}
            self.viewer.set_candidate_overlay(
                frame_idx,
                ensure_mat_tracking([]),
                self.add_point_panel.candidate_style(),
            )
            message = f"Localization failed: {exc}"
            self.add_point_panel.set_status(message, error=True)
            self.status_label.setText(message)
            return
        finally:
            QApplication.restoreOverrideCursor()

        if (
            self.correction_mode != "add_point"
            or self.frame_spin.value() - 1 != frame_idx
        ):
            return
        self._add_point_candidate_frame = frame_idx
        self._add_point_used_params = dict(params)
        self.viewer.set_candidate_overlay(
            frame_idx,
            candidates,
            self.add_point_panel.candidate_style(),
        )
        count = int(candidates.shape[0])
        if count:
            self.add_point_panel.set_status(
                f"{count} candidate(s) on frame {frame_idx + 1} · click one in the image."
            )
            self.status_label.setText(
                f"Add Point: click one of {count} candidates on frame {frame_idx + 1}."
            )
        else:
            self.add_point_panel.set_status(
                "No candidates found · expand settings to adjust the method or parameters."
            )
            self.status_label.setText(
                f"Add Point found no candidates on frame {frame_idx + 1}."
            )

    def on_candidate_style_changed(self, style: dict) -> None:
        """Restyle the temporary overlay without rerunning localization."""

        self.viewer.update_candidate_style(style)

    def on_candidate_clicked(self, frame_idx: int, candidate_idx: int) -> None:
        """Apply one viewer candidate while keeping Add Point active."""

        if (
            self.correction_mode != "add_point"
            or self._is_add_point_eraser_active()
            or self._is_manual_add_point_method()
            or frame_idx != self._add_point_candidate_frame
        ):
            return
        candidates = self.viewer.candidate_rows
        if candidate_idx < 0 or candidate_idx >= candidates.shape[0]:
            return
        row = candidates[candidate_idx]
        self._apply_candidate(
            frame_idx,
            float(row[0]),
            float(row[1]),
            float(row[2]),
            dict(self._add_point_used_params),
        )

    def _is_manual_add_point_method(self) -> bool:
        return self.add_point_panel.current_method() == MANUAL_PLACEMENT_METHOD

    def _is_add_point_eraser_active(self) -> bool:
        return self.add_point_panel.eraser_active()

    def _add_point_eraser_prompt(self, frame_idx: int) -> str:
        return (
            f"Eraser · Frame {frame_idx + 1}: click an unassigned Add Point cross; "
            "assigned points are protected."
        )

    def on_add_point_eraser_toggled(self, active: bool) -> None:
        """Toggle the Add Point sub-tool for discarding stored manual rows."""

        if active:
            if self.correction_mode != "add_point":
                self.mode_buttons["add_point"].setChecked(True)
                return
            self._prepare_add_point_eraser()
            return
        if self.correction_mode == "add_point":
            self.refresh_add_point_candidates()

    def _prepare_add_point_eraser(self) -> None:
        """Pause candidate creation while the manual-point eraser is active."""

        if self.session is None:
            return
        self._add_point_refresh_timer.stop()
        self._add_point_candidate_frame = None
        self._add_point_used_params = {}
        self.viewer.set_manual_candidate_placement_enabled(False)
        self.viewer.set_manual_detection_eraser_enabled(True)
        self.viewer.clear_candidate_overlay()
        prompt = self._add_point_eraser_prompt(self.frame_spin.value() - 1)
        self.add_point_panel.set_status(prompt)
        self.status_label.setText(prompt)

    def on_manual_detection_erase_requested(
        self,
        frame_idx: int,
        local_idx: int,
    ) -> None:
        """Discard a clicked, unassigned Add Point detection."""

        if (
            self.session is None
            or self.correction_mode != "add_point"
            or not self._is_add_point_eraser_active()
            or frame_idx != self.frame_spin.value() - 1
        ):
            return
        owner = self.session.track_id_for_detection(frame_idx, local_idx)
        if owner is not None:
            message = (
                f"Detection {local_idx} is assigned to track {owner}. Use Remove "
                "first, then return to the eraser."
            )
            self.add_point_panel.set_status(message, error=True)
            self.status_label.setText(message)
            return
        try:
            self.session.discard_manual_detection(frame_idx, local_idx)
        except (IndexError, ValueError) as exc:
            self.add_point_panel.set_status(str(exc), error=True)
            self.status_label.setText(str(exc))
            return
        self.viewer.render()
        self._sync_enabled_state()
        message = (
            f"Erased added detection {local_idx} on frame {frame_idx + 1}. "
            "It will be omitted from saved data; Undo restores it."
        )
        self.add_point_panel.set_status(message)
        self.status_label.setText(message)

    def _manual_placement_prompt(self, frame_idx: int) -> str:
        return (
            f"Manual Placement · Frame {frame_idx + 1}: click the image to place a draft; "
            "Shift+W/A/S/D to nudge; Enter to confirm; Backspace to clear; Esc to exit."
        )

    def _prepare_manual_candidate_placement(self) -> None:
        """Enter manual placement without running a localization algorithm."""

        if self.session is None or self.session.image_stack is None:
            return
        self._add_point_refresh_timer.stop()
        frame_idx = min(
            self.frame_spin.value() - 1,
            self.session.image_stack.shape[2] - 1,
        )
        self._add_point_candidate_frame = frame_idx
        self._add_point_used_params = self.add_point_panel.current_params()
        self.viewer.set_manual_detection_eraser_enabled(False)
        self.viewer.set_manual_candidate_placement_enabled(True)
        self.viewer.set_candidate_overlay(
            frame_idx,
            ensure_mat_tracking([]),
            self.add_point_panel.candidate_style(),
        )
        prompt = self._manual_placement_prompt(frame_idx)
        self.add_point_panel.set_status(prompt)
        self.status_label.setText(prompt)

    def on_manual_candidate_placed(
        self,
        frame_idx: int,
        z: float,
        x: float,
    ) -> None:
        """Place or reposition the provisional human-defined point."""

        if (
            self.correction_mode != "add_point"
            or self._is_add_point_eraser_active()
            or not self._is_manual_add_point_method()
            or frame_idx != self.frame_spin.value() - 1
        ):
            return
        self._set_manual_candidate_position(frame_idx, z, x)

    def _set_manual_candidate_position(
        self,
        frame_idx: int,
        z: float,
        x: float,
    ) -> None:
        stack = self.session.image_stack
        if stack is None:
            return
        nz, nx = stack.shape[:2]
        z = float(np.clip(z, 1.0, nz))
        x = float(np.clip(x, 1.0, nx))
        intensity = sample_manual_intensity(stack[:, :, frame_idx], z, x)
        self._add_point_candidate_frame = frame_idx
        self._add_point_used_params = self.add_point_panel.current_params()
        self.viewer.set_candidate_overlay(
            frame_idx,
            np.asarray([[intensity, z, x, float(frame_idx + 1)]], dtype=float),
            self.add_point_panel.candidate_style(),
        )
        message = (
            f"Manual Placement draft · Frame {frame_idx + 1}, z={z:.2f}, x={x:.2f} · "
            "Shift+W/A/S/D to nudge; Enter to confirm; Backspace to clear."
        )
        self.add_point_panel.set_status(message)
        self.status_label.setText(message)

    def _clear_manual_candidate(self) -> None:
        if self.session is None:
            return
        frame_idx = self.frame_spin.value() - 1
        self._add_point_candidate_frame = frame_idx
        self._add_point_used_params = self.add_point_panel.current_params()
        self.viewer.set_candidate_overlay(
            frame_idx,
            ensure_mat_tracking([]),
            self.add_point_panel.candidate_style(),
        )

    def on_add_point_track_context_changed(self) -> None:
        """Discard a draft that could otherwise be confirmed into another track."""

        if (
            self.correction_mode != "add_point"
            or self._is_add_point_eraser_active()
            or not self._is_manual_add_point_method()
        ):
            return
        had_draft = self.viewer.candidate_rows.shape == (1, 4)
        if had_draft:
            self._clear_manual_candidate()
        prompt = self._manual_placement_prompt(self.frame_spin.value() - 1)
        message = (
            "Track selection changed; the human-defined draft was cleared. " + prompt
            if had_draft
            else prompt
        )
        self.add_point_panel.set_status(message)
        self.status_label.setText(message)

    def handle_manual_candidate_key(self, event) -> bool:
        """Handle context-only manual placement keys from the app event filter."""

        if (
            self.correction_mode != "add_point"
            or self._is_add_point_eraser_active()
            or not self._is_manual_add_point_method()
        ):
            return False

        key = event.key()
        modifiers = event.modifiers()
        directions = {
            Qt.Key_W: (-MANUAL_NUDGE_STEP_PX, 0.0),
            Qt.Key_A: (0.0, -MANUAL_NUDGE_STEP_PX),
            Qt.Key_S: (MANUAL_NUDGE_STEP_PX, 0.0),
            Qt.Key_D: (0.0, MANUAL_NUDGE_STEP_PX),
        }
        if modifiers == Qt.ShiftModifier and key in directions:
            candidates = self.viewer.candidate_rows
            if candidates.shape != (1, 4):
                message = self._manual_placement_prompt(self.frame_spin.value() - 1)
                self.add_point_panel.set_status(message)
                self.status_label.setText(message)
                return True
            dz, dx = directions[key]
            row = candidates[0]
            self._set_manual_candidate_position(
                self.frame_spin.value() - 1,
                float(row[1]) + dz,
                float(row[2]) + dx,
            )
            return True

        plain_modifiers = (Qt.NoModifier, Qt.KeypadModifier)
        if key in (Qt.Key_Return, Qt.Key_Enter) and modifiers in plain_modifiers:
            candidates = self.viewer.candidate_rows
            if candidates.shape != (1, 4):
                message = self._manual_placement_prompt(self.frame_spin.value() - 1)
                self.add_point_panel.set_status(message)
                self.status_label.setText(message)
                return True
            row = candidates[0]
            applied = self._apply_candidate(
                self.frame_spin.value() - 1,
                float(row[0]),
                float(row[1]),
                float(row[2]),
                dict(self._add_point_used_params),
                source="manual_placement",
            )
            if applied:
                applied_status = self.status_label.text()
                self._clear_manual_candidate()
                self.add_point_panel.set_status(
                    "Point confirmed · Click the image to place the next draft."
                )
                self.status_label.setText(
                    applied_status
                    + " Manual Placement remains active; click to place the next draft."
                )
            return True

        if key == Qt.Key_Backspace and modifiers == Qt.NoModifier:
            self._clear_manual_candidate()
            prompt = self._manual_placement_prompt(self.frame_spin.value() - 1)
            self.add_point_panel.set_status("Draft cleared · " + prompt)
            self.status_label.setText("Draft cleared · " + prompt)
            return True
        return False

    def _apply_candidate(
        self,
        frame_idx: int,
        intensity: float,
        z: float,
        x: float,
        params: dict,
        *,
        source: str = "localization_candidates",
    ) -> bool:
        """Insert a chosen candidate, reusing an existing detection if it matches.

        Candidates often include the original localization results; if the
        chosen one coincides with an existing row, assign that row instead of
        appending a duplicate (the localization result is then unchanged).
        """

        match = find_matching_detection(
            self.session.localized_by_frame,
            frame_idx,
            z,
            x,
            excluded_local_indices={
                local_idx
                for discarded_frame, local_idx in (
                    self.session.discarded_manual_detections
                )
                if discarded_frame == frame_idx
            },
        )
        if match is not None:
            if self.selected_track_id is None:
                self.status_label.setText(
                    f"Candidate matches existing detection {match}; select a "
                    "track to assign it."
                )
                return False
            self._assign_clicked_detection(frame_idx, match)
            return (
                self.session.local_idx_for_track_frame(
                    self.selected_track_id,
                    frame_idx,
                )
                == match
            )

        track_id = self.selected_track_id
        track_was_verified = (
            track_id is not None and self._is_track_verified(track_id)
        )
        if track_was_verified and not self._confirm(
            "Modify a VERIFIED track?",
            f"Track {track_id} is VERIFIED. Adding this point will FLAG it and "
            "require a complete re-review. Continue?",
        ):
            return False
        gid = new_group_id()
        try:
            local_idx = self.session.append_manual_detection(
                frame_idx,
                intensity,
                z,
                x,
                track_id=track_id,
                group_id=gid,
                extra_payload={
                    "source": source,
                    "params": params,
                },
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not add point", str(exc))
            return False
        if track_was_verified:
            self._flag_demoted_track(
                track_id,
                gid,
                f"a candidate point was added at frame {frame_idx + 1}.",
            )
        self._refresh_after_edit([track_id])
        assigned = "" if track_id is None else f" and assigned to track {track_id}"
        self.status_label.setText(
            f"Added detection {local_idx} at frame {frame_idx + 1} "
            f"(z={z:.2f}, x={x:.2f}, {params.get('method', '?')}){assigned}. "
            "Localization result modified."
        )

        return True

    # ------------------------------------------------------------- dialogs

    def open_merge_dialog(self) -> None:
        """Merge two tracks chosen by ID."""

        if self.session is None or len(self.session.active_nonempty_track_ids) < 2:
            return
        self.mode_buttons["inspect"].setChecked(True)
        dialog = MergeTracksDialog(self.session, self.selected_track_id, self)
        if dialog.exec() != MergeTracksDialog.Accepted:
            return
        target_id = dialog.target_id
        source_id = dialog.source_id
        verified = [
            track_id
            for track_id in (target_id, source_id)
            if self._is_track_verified(track_id)
        ]
        if verified and not self._confirm(
            "Merge VERIFIED track?",
            "This merge structurally changes VERIFIED track(s) "
            f"{', '.join(str(track_id) for track_id in verified)}. Continue?",
        ):
            return
        try:
            result = self.session.merge_tracks(target_id, source_id)
        except Exception as exc:
            QMessageBox.warning(self, "Could not merge", str(exc))
            return
        self._refresh_after_structural_edit(
            [target_id, source_id], preferred_track_id=target_id
        )
        dropped = (
            f", {result['dropped']} overlapping point(s) unassigned"
            if result["dropped"]
            else ""
        )
        self.status_label.setText(
            f"Merged track {source_id} into track {target_id}: "
            f"{result['moved']} point(s) moved{dropped}. Saved Track IDs stay "
            "unchanged."
        )

    def open_split_dialog(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        parent_id = self.selected_track_id
        if not self.session.is_track_active(parent_id):
            return
        self.mode_buttons["inspect"].setChecked(True)
        try:
            dialog = SplitTrackDialog(
                self.session,
                parent_id,
                self.frame_spin.value(),
                self,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Could not split", str(exc))
            return
        if dialog.exec() != SplitTrackDialog.Accepted:
            return
        if self._is_track_verified(parent_id) and not self._confirm(
            "Split VERIFIED track?",
            f"Track {parent_id} is VERIFIED. Splitting retires it and creates "
            "two new FLAGGED tracks. Continue?",
        ):
            return
        try:
            result = self.session.split_track(parent_id, dialog.split_after_frame)
        except Exception as exc:
            QMessageBox.warning(self, "Could not split", str(exc))
            return
        current_frame = self.frame_spin.value() - 1
        preferred = (
            result["first_child_id"]
            if current_frame <= result["split_after_frame"]
            else result["second_child_id"]
        )
        self._refresh_after_structural_edit(
            [parent_id, result["first_child_id"], result["second_child_id"]],
            preferred_track_id=preferred,
        )
        self.status_label.setText(
            f"Split track {parent_id} into flagged tracks "
            f"{result['first_child_id']} ({result['first_points']} points) and "
            f"{result['second_child_id']} ({result['second_points']} points)."
        )

    def delete_selected_track(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        track_id = self.selected_track_id
        if not self.session.is_track_active(track_id):
            return
        summary = self.session.track_summary(track_id)
        warning = (
            "\n\nThis track is VERIFIED; deleting it invalidates that review."
            if self._is_track_verified(track_id)
            else ""
        )
        if not self._confirm(
            "Delete entire track?",
            f"Retire track {track_id} and release its {summary['n_points']} "
            "assignment(s)? Localization detections are preserved and become "
            f"unassigned.{warning}\n\nThis operation can be undone.",
        ):
            return
        self.mode_buttons["inspect"].setChecked(True)
        try:
            result = self.session.delete_track(track_id)
        except Exception as exc:
            QMessageBox.warning(self, "Could not delete track", str(exc))
            return
        self.selected_track_id = None
        self._refresh_after_structural_edit(
            [track_id],
            select_first_visible_if_none=True,
        )
        self.status_label.setText(
            f"Deleted track {track_id}; {result['removed']} detection(s) are now "
            "unassigned."
        )

    def open_track_history(self, track_id: int | None = None) -> None:
        """Show the audit history of one track (read-only, non-modal)."""

        if self.session is None:
            return
        tid = self.selected_track_id if track_id is None else track_id
        if tid is None:
            self.status_label.setText("Select a track to view its edit history.")
            return
        self._close_history_dialog()
        dialog = TrackHistoryDialog(self.session, tid, self)
        dialog.jump_to_frame.connect(self.set_frame_1based)
        self._history_dialog = dialog
        dialog.show()

    def _close_history_dialog(self) -> None:
        if self._history_dialog is not None:
            self._history_dialog.close()
            self._history_dialog = None

    # ---------------------------------------------------------- undo / redo

    def undo(self) -> None:
        if self.undo_manager is None:
            return
        group = self.undo_manager.undo()
        if group is None:
            self.status_label.setText("Nothing to undo.")
            return
        affected = [edit.track_id for edit in group]
        if any(edit.action in STRUCTURAL_EDIT_ACTIONS for edit in group):
            self._refresh_after_structural_edit(
                affected,
                preferred_track_id=_deleted_track_id_in_group(group),
            )
        else:
            self._refresh_after_edit(affected)
        self.status_label.setText(f"Undid {group[-1].action}.")

    def redo(self) -> None:
        if self.undo_manager is None:
            return
        group = self.undo_manager.redo()
        if group is None:
            self.status_label.setText("Nothing to redo.")
            return
        affected = [edit.track_id for edit in group]
        if any(edit.action in STRUCTURAL_EDIT_ACTIONS for edit in group):
            self._refresh_after_structural_edit(
                affected,
                select_first_visible_if_none=(
                    _deleted_track_id_in_group(group) is not None
                ),
            )
        else:
            self._refresh_after_edit(affected)
        self.status_label.setText(f"Redid {group[-1].action}.")

    # -------------------------------------------------------------- status

    def mark_selected_verified(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        track_id = self.selected_track_id
        self.session.set_track_status(track_id, TRACK_STATUS_VERIFIED)
        self._update_track_item(track_id)
        self._sync_enabled_state()
        self.viewer.render()  # path color depends on verified status
        self.status_label.setText(f"Track {track_id} marked verified.")

    def flag_selected(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        track_id = self.selected_track_id
        self.session.set_track_status(track_id, TRACK_STATUS_FLAGGED)
        self._update_track_item(track_id)
        self._sync_enabled_state()
        self.viewer.render()  # a previously verified track loses its color
        self.status_label.setText(f"Track {track_id} flagged.")
