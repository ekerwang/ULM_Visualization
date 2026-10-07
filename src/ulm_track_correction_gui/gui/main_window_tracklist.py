"""Track list population, filtering, and selection for the main window."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QMenu

from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_EDITED,
    TRACK_STATUS_FLAGGED,
    TRACK_STATUS_UNREVIEWED,
    TRACK_STATUS_VERIFIED,
)
from ulm_track_correction_gui.gui.track_list_delegate import SUMMARY_ROLE

SORT_OPTIONS = (
    "Track ID ↑",
    "Track ID ↓",
    "Points ↑",
    "Points ↓",
    "Gaps ↑",
    "Gaps ↓",
    "Start frame ↑",
    "Start frame ↓",
)
_SORT_SPECS = {
    "Track ID ↑": ("track_id", False),
    "Track ID ↓": ("track_id", True),
    "Points ↑": ("n_points", False),
    "Points ↓": ("n_points", True),
    "Gaps ↑": ("n_gaps", False),
    "Gaps ↓": ("n_gaps", True),
    "Start frame ↑": ("start_frame", False),
    "Start frame ↓": ("start_frame", True),
}
FILTER_UNVERIFIED = "Unverified"
FILTER_OPTIONS = (
    "All",
    FILTER_UNVERIFIED,
    TRACK_STATUS_UNREVIEWED,
    TRACK_STATUS_EDITED,
    TRACK_STATUS_VERIFIED,
    TRACK_STATUS_FLAGGED,
)


def _sort_track_summaries(
    summaries: list[dict[str, Any]], sort_mode: str
) -> list[dict[str, Any]]:
    """Return summaries ordered by the requested field and direction.

    Track ID is an ascending tie-breaker so every mode stays deterministic.
    Active empty drafts have no start frame and remain at the end in both
    start-frame directions.
    """

    field, descending = _SORT_SPECS[sort_mode]

    def sort_key(summary: dict[str, Any]) -> tuple[bool, int, int]:
        value = summary[field]
        if value is None:
            return (True, 0, summary["track_id"])
        primary = -value if descending else value
        return (False, primary, summary["track_id"])

    return sorted(summaries, key=sort_key)


class TrackListMixin:
    """Keeps the track list in sync with the session (single-row updates)."""

    def _update_track_count_label(self) -> None:
        """Show the number of rows produced by the active track filter."""

        count = self.track_list.count()
        filter_name = self.filter_combo.currentText()
        self.track_count_label.setText(f"{count:,} shown")
        self.track_count_label.setToolTip(
            f"{count:,} active track(s) match the {filter_name!r} filter."
        )

    def populate_track_list(self) -> None:
        self.track_list.blockSignals(True)
        self.track_list.clear()
        self._row_by_track_id = {}
        if self.session is None:
            self._update_track_count_label()
            self.track_list.blockSignals(False)
            return

        summaries = [
            self.session.track_summary(track_id)
            for track_id in self.session.active_track_ids
        ]

        status_filter = self.filter_combo.currentText()
        if status_filter == FILTER_UNVERIFIED:
            # everything still needing attention: all statuses except verified
            summaries = [
                s for s in summaries if s["status"] != TRACK_STATUS_VERIFIED
            ]
        elif status_filter != "All":
            summaries = [s for s in summaries if s["status"] == status_filter]

        summaries = _sort_track_summaries(summaries, self.sort_combo.currentText())

        for row, summary in enumerate(summaries):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, summary["track_id"])
            item.setData(SUMMARY_ROLE, summary)
            if summary["note"]:
                item.setToolTip(summary["note"])
            self.track_list.addItem(item)
            self._row_by_track_id[summary["track_id"]] = row

        if self.selected_track_id in self._row_by_track_id:
            self.track_list.setCurrentRow(self._row_by_track_id[self.selected_track_id])
        self._update_track_count_label()
        self.track_list.blockSignals(False)

    def _update_track_item(self, track_id: int) -> None:
        """Refresh a single row after a status change; avoids rebuilding the
        whole list, which is too slow for thousands of tracks."""

        if self.session is None:
            return
        updated_track_was_selected = self.selected_track_id == track_id
        summary = self.session.track_summary(track_id)
        row = self._row_by_track_id.get(track_id)
        if (
            self.filter_combo.currentText() != "All"
            or not summary["active"]
            or row is None
        ):
            # The row may no longer match the active filter, the track may
            # have emptied out (merge), or it may need to reappear (undo of a
            # merge); rebuild the list.
            self.populate_track_list()
            if updated_track_was_selected and track_id not in self._row_by_track_id:
                if self.track_list.count() > 0:
                    # Status-filter review is a continuous workflow. Advance to
                    # the first remaining row before track-derived frame scopes
                    # can lose their valid selection and fall back to Full stack.
                    self.track_list.setCurrentRow(0)
                else:
                    self.on_track_selected(None)
            return
        item = self.track_list.item(row)
        if item is not None:
            item.setData(SUMMARY_ROLE, summary)
            if summary["note"]:
                item.setToolTip(summary["note"])

    def on_track_selected(self, current: QListWidgetItem | None) -> None:
        if current is None or self.session is None:
            self.selected_track_id = None
            self.viewer.set_selected_track(None)
            self._sync_enabled_state()
            self.on_add_point_track_context_changed()
            return

        self.selected_track_id = int(current.data(Qt.UserRole))
        self.viewer.set_selected_track(self.selected_track_id)
        summary = self.session.track_summary(self.selected_track_id)
        # A new selection changes both track-derived slider ranges. Recompute
        # them before jumping so an active Track range/±N scope clamps against
        # the new track rather than the previous one.
        self.refresh_frame_scope_controls()
        if summary["start_frame"] is not None and not self._suppress_frame_jump:
            self.set_frame_1based(summary["start_frame"] + 1)
        self.status_label.setText(self._track_status_text(self.selected_track_id))
        self._sync_enabled_state()
        self.on_add_point_track_context_changed()

    def clear_track_selection(self) -> None:
        """Clear list and viewer selection after an Inspect background click."""

        if self.session is None:
            return
        signals_were_blocked = self.track_list.blockSignals(True)
        try:
            self.track_list.setCurrentRow(-1)
            self.track_list.clearSelection()
        finally:
            self.track_list.blockSignals(signals_were_blocked)
        self.on_track_selected(None)
        self.status_label.setText(
            f"Frame {self.frame_spin.value()}: no track selected."
        )

    def on_track_picked(self, track_id: int) -> None:
        """Viewer picking selects the track without changing the current frame."""

        if self.session is None:
            return
        row = self._row_by_track_id.get(track_id)
        self._suppress_frame_jump = True
        try:
            if row is not None:
                self.track_list.setCurrentRow(row)
            else:
                # hidden by the active filter: select it directly anyway
                self.selected_track_id = track_id
                self.viewer.set_selected_track(track_id)
                self.status_label.setText(
                    self._track_status_text(track_id) + " (hidden by filter)"
                )
                self._sync_enabled_state()
        finally:
            self._suppress_frame_jump = False

    def _track_status_text(self, track_id: int) -> str:
        summary = self.session.track_summary(track_id)
        if summary["start_frame"] is None:
            return f"Track {track_id}: active empty draft — assign or add a point."
        return (
            f"Track {track_id}: {summary['n_points']} pts, "
            f"{summary['n_gaps']} gaps, frames "
            f"{summary['start_frame'] + 1}–{summary['end_frame'] + 1}, "
            f"{summary['status']}."
        )

    def _show_track_context_menu(self, pos) -> None:
        item = self.track_list.itemAt(pos)
        if item is None or self.session is None:
            return
        self.track_list.setCurrentItem(item)
        track_id = int(item.data(Qt.UserRole))
        menu = QMenu(self.track_list)
        history_action = menu.addAction("Edit History…")
        menu.addSeparator()
        verify_action = menu.addAction("Mark Verified")
        flag_action = menu.addAction("Flag")
        menu.addSeparator()
        split_action = menu.addAction("Split Track…")
        split_action.setEnabled(self.session.track_summary(track_id)["n_points"] >= 2)
        delete_action = menu.addAction("Delete Track…")
        chosen = menu.exec(self.track_list.viewport().mapToGlobal(pos))
        if chosen is history_action:
            self.open_track_history(track_id)
        elif chosen is verify_action:
            self.mark_selected_verified()
        elif chosen is flag_action:
            self.flag_selected()
        elif chosen is split_action:
            self.open_split_dialog()
        elif chosen is delete_action:
            self.delete_selected_track()

    def step_track(self, delta: int) -> None:
        count = self.track_list.count()
        if count == 0:
            return
        row = self.track_list.currentRow()
        row = 0 if row < 0 else max(0, min(count - 1, row + delta))
        self.track_list.setCurrentRow(row)
