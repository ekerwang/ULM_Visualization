"""Persistent dataset queue behavior for the main window."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
)

from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_VERIFIED,
    CorrectionEdit,
)
from ulm_track_correction_gui.core.dataset_queue import (
    DATASET_STATUS_COMPLETED,
    DATASET_STATUS_FLAGGED,
    DATASET_STATUS_IN_PROGRESS,
    DATASET_STATUS_PENDING,
    DatasetQueue,
    DatasetQueueEntry,
    canonical_dataset_path,
)
from ulm_track_correction_gui.gui.confirmation_dialog import ask_yes_no
from ulm_track_correction_gui.gui.dataset_list_delegate import DATASET_SUMMARY_ROLE
from ulm_track_correction_gui.io.mat_io import save_correction_session_atomic


DATASET_QUEUE_SETTINGS_KEY = "datasetQueue/json-v1"
DATASET_NAME_FILTER_ALL_RESULTS = (
    "All correction result MAT files (*_results.mat)"
)
DATASET_NAME_FILTER_UP_RESULTS = "Up result MAT files (*_up_results.mat)"
DATASET_NAME_FILTER_DOWN_RESULTS = "Down result MAT files (*_down_results.mat)"
DATASET_NAME_FILTERS = (
    DATASET_NAME_FILTER_ALL_RESULTS,
    DATASET_NAME_FILTER_UP_RESULTS,
    DATASET_NAME_FILTER_DOWN_RESULTS,
    "MAT files (*.mat)",
    "All files (*)",
)


class DatasetImportDialog(QFileDialog):
    """Multi-file chooser with direction filters and reliable select-all keys."""

    def __init__(
        self,
        parent=None,
        *,
        start_dir: str = "",
        title: str = "Add datasets",
    ) -> None:
        super().__init__(parent, title, start_dir)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.ExistingFiles)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setNameFilters(list(DATASET_NAME_FILTERS))
        self.selectNameFilter(DATASET_NAME_FILTER_ALL_RESULTS)

        self._file_views: list[QAbstractItemView] = []
        for object_name in ("listView", "treeView"):
            view = self.findChild(QAbstractItemView, object_name)
            if view is not None:
                view.installEventFilter(self)
                self._file_views.append(view)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched in self._file_views
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_A
            and event.modifiers() in (Qt.ControlModifier, Qt.MetaModifier)
        ):
            watched.selectAll()
            return True
        return super().eventFilter(watched, event)


class DatasetListMixin:
    """Imports, persists, switches, saves, and renders dataset queue entries."""

    # ------------------------------------------------------------- persistence

    def restore_dataset_queue(self) -> None:
        raw = self.dataset_settings.value(DATASET_QUEUE_SETTINGS_KEY, "")
        if raw in (None, ""):
            self.dataset_queue = DatasetQueue()
            return
        try:
            self.dataset_queue = DatasetQueue.from_json(str(raw))
        except ValueError as exc:
            self.dataset_queue = DatasetQueue()
            self.dataset_settings.remove(DATASET_QUEUE_SETTINGS_KEY)
            self.status_label.setText(f"Dataset queue settings were ignored: {exc}")

    def persist_dataset_queue(self) -> None:
        self.dataset_settings.setValue(
            DATASET_QUEUE_SETTINGS_KEY,
            self.dataset_queue.to_json(),
        )
        self.dataset_settings.sync()

    def schedule_dataset_queue_persist(self) -> None:
        self._dataset_persist_timer.start(250)

    # ---------------------------------------------------------------- display

    def populate_dataset_list(self, select_path: str | Path | None = None) -> None:
        selected_path = None
        current_item = self.dataset_list.currentItem()
        if current_item is not None:
            selected_path = current_item.data(Qt.UserRole)
        if select_path is not None:
            selected_path = canonical_dataset_path(select_path)

        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        self._dataset_row_by_path = {}
        for row, entry in enumerate(self.dataset_queue.entries):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry.source_path)
            item.setData(DATASET_SUMMARY_ROLE, self._dataset_summary(entry))
            item.setToolTip(self._dataset_tooltip(entry))
            self.dataset_list.addItem(item)
            self._dataset_row_by_path[entry.source_path] = row
        if selected_path in self._dataset_row_by_path:
            self.dataset_list.setCurrentRow(self._dataset_row_by_path[selected_path])
        self.dataset_list.blockSignals(False)
        self._update_dataset_header()
        self._sync_dataset_controls()

    def _dataset_summary(self, entry: DatasetQueueEntry) -> dict:
        return {
            "name": entry.source.name,
            "status": entry.status,
            "verified_tracks": entry.verified_tracks,
            "active_tracks": entry.active_tracks,
            "missing": not entry.source.is_file(),
            "is_current": entry.source_path == self._current_dataset_path,
        }

    def _dataset_tooltip(self, entry: DatasetQueueEntry) -> str:
        saved = entry.last_saved_at or "Never"
        return (
            f"Source: {entry.source_path}\n"
            f"Corrected: {entry.corrected_path}\n"
            f"Status: {entry.status.replace('_', ' ')}\n"
            f"Last saved: {saved}"
        )

    def _update_dataset_header(self) -> None:
        self.dataset_header.setText(
            f"Datasets · {self.dataset_queue.completed_count()} / "
            f"{len(self.dataset_queue)} completed"
        )

    def _update_dataset_item(self, source_path: str | Path) -> None:
        key = canonical_dataset_path(source_path)
        row = self._dataset_row_by_path.get(key)
        entry = self.dataset_queue.entry_for_path(key)
        if row is None or entry is None:
            return
        item = self.dataset_list.item(row)
        if item is None:
            return
        item.setData(DATASET_SUMMARY_ROLE, self._dataset_summary(entry))
        item.setToolTip(self._dataset_tooltip(entry))
        self._update_dataset_header()
        self.dataset_list.viewport().update()

    def on_dataset_selected(self, current: QListWidgetItem | None) -> None:
        self._sync_dataset_controls()

    def _sync_dataset_controls(self) -> None:
        if not hasattr(self, "dataset_list"):
            return
        selected = self.dataset_list.currentItem()
        selected_path = None if selected is None else selected.data(Qt.UserRole)
        is_current_selection = selected_path == self._current_dataset_path
        self.remove_dataset_button.setEnabled(
            selected_path is not None and not is_current_selection
        )
        self.remove_all_datasets_button.setEnabled(bool(self.dataset_queue.entries))

        current_entry = self.current_dataset_entry()
        current_matches_session = (
            current_entry is not None
            and self.session is not None
            and self.current_path is not None
            and not self._autosave_blocked
            and canonical_dataset_path(self.current_path)
            == current_entry.source_path
        )
        self.save_progress_button.setEnabled(current_matches_session)

        next_available = False
        if current_entry is not None:
            current_index = self.dataset_queue.index_for_path(current_entry.source_path)
            next_available = (
                current_index is not None
                and self.dataset_queue.next_index(current_index) is not None
            )
        self.next_dataset_button.setEnabled(next_available)

    # ----------------------------------------------------------------- import

    def import_dataset_list(self) -> None:
        start_dir = ""
        if self.current_path is not None:
            start_dir = str(self.current_path.parent)
        dialog = DatasetImportDialog(self, start_dir=start_dir)
        if dialog.exec() == QFileDialog.Accepted:
            self.add_dataset_paths(dialog.selectedFiles(), report_dialog=True)

    def add_dataset_paths(
        self,
        paths: list[str | Path],
        *,
        report_dialog: bool = False,
    ) -> dict[str, int]:
        accepted: list[str | Path] = []
        skipped_checkpoint = 0
        skipped_corrected = 0
        skipped_non_mat = 0
        for path in paths:
            candidate = Path(path).expanduser().resolve(strict=False)
            lower_name = candidate.name.lower()
            if candidate.suffix.lower() != ".mat":
                skipped_non_mat += 1
            elif lower_name.endswith("_accumulation_checkpoint.mat"):
                skipped_checkpoint += 1
            elif candidate.parent.name.lower() == "corrected":
                skipped_corrected += 1
            else:
                accepted.append(candidate)

        previous_count = len(self.dataset_queue)
        added, duplicates = self.dataset_queue.add_paths(accepted)
        select_path = (
            self.dataset_queue.entries[previous_count].source_path if added else None
        )
        self.persist_dataset_queue()
        self.populate_dataset_list(select_path=select_path)

        skipped = skipped_checkpoint + skipped_corrected + skipped_non_mat
        message = (
            f"Imported {added} dataset(s); {duplicates} duplicate(s) kept unchanged; "
            f"{skipped} unsupported selection(s) skipped."
        )
        self.status_label.setText(message)
        if report_dialog and skipped:
            QMessageBox.information(
                self,
                "Dataset import summary",
                message
                + "\n\nCheckpoint MAT files and corrected outputs are not "
                "dataset queue inputs.",
            )
        return {
            "added": added,
            "duplicates": duplicates,
            "skipped_checkpoint": skipped_checkpoint,
            "skipped_corrected": skipped_corrected,
            "skipped_non_mat": skipped_non_mat,
        }

    def remove_selected_dataset(self) -> None:
        item = self.dataset_list.currentItem()
        if item is None:
            return
        source_path = str(item.data(Qt.UserRole))
        if source_path == self._current_dataset_path:
            return
        entry = self.dataset_queue.entry_for_path(source_path)
        if entry is None:
            return
        confirmed = ask_yes_no(
            self,
            "Remove dataset from list?",
            f"Remove {entry.source.name} from the dataset list?\n\n"
            "No MAT or corrected files will be deleted.",
        )
        if not confirmed:
            return
        index = self.dataset_queue.index_for_path(source_path)
        if index is None:
            return
        self.dataset_queue.remove_at(index)
        self.persist_dataset_queue()
        if self.dataset_queue.entries:
            select_index = min(index, len(self.dataset_queue.entries) - 1)
            select_path = self.dataset_queue.entries[select_index].source_path
        else:
            select_path = None
        self.populate_dataset_list(select_path=select_path)

    def remove_all_datasets(self) -> bool:
        count = len(self.dataset_queue)
        if count == 0 or not self._confirm_remove_all_datasets(count):
            return False

        self.dataset_queue = DatasetQueue()
        self.persist_dataset_queue()
        self.populate_dataset_list()
        self.status_label.setText(
            f"Removed {count} dataset(s) from the list. No files were deleted."
        )
        return True

    def _confirm_remove_all_datasets(self, count: int) -> bool:
        current_note = ""
        if self.current_dataset_entry() is not None:
            current_note = (
                "\n\nThe currently open dataset will remain open, but Save progress "
                "will be unavailable until it is added to the queue again."
            )
        return ask_yes_no(
            self,
            "Remove all datasets from list?",
            f"Remove all {count} datasets and their queue progress from the list?"
            f"{current_note}\n\nNo MAT or corrected files will be deleted.",
        )

    # --------------------------------------------------------- session linkage

    def current_dataset_entry(self) -> DatasetQueueEntry | None:
        if self._current_dataset_path is None:
            return None
        return self.dataset_queue.entry_for_path(self._current_dataset_path)

    def on_dataset_session_loaded(
        self,
        path: str | Path,
        *,
        recovered_edits: bool,
    ) -> None:
        key = canonical_dataset_path(path)
        entry = self.dataset_queue.entry_for_path(key)
        self._current_dataset_path = None if entry is None else key
        self._session_dirty = bool(recovered_edits)
        self._active_track_ids = None
        self._verified_track_ids = None

        if entry is not None:
            if entry.status == DATASET_STATUS_PENDING or (
                recovered_edits
                and entry.status
                in {DATASET_STATUS_COMPLETED, DATASET_STATUS_FLAGGED}
            ):
                entry.status = DATASET_STATUS_IN_PROGRESS
            self._refresh_current_dataset_progress()
            self.persist_dataset_queue()
        self.populate_dataset_list(select_path=key if entry is not None else None)

    def on_dataset_session_edit(self, edit: CorrectionEdit) -> None:
        self._session_dirty = True
        if self.session is not None:
            self.session.metadata["dataset_review_state"] = (
                DATASET_STATUS_IN_PROGRESS
            )
        entry = self.current_dataset_entry()
        if entry is None:
            return
        if entry.status != DATASET_STATUS_IN_PROGRESS:
            entry.status = DATASET_STATUS_IN_PROGRESS
        affected = () if edit.track_id is None else (edit.track_id,)
        self._refresh_current_dataset_progress(affected)
        self._update_dataset_item(entry.source_path)
        self._sync_dataset_controls()
        self.schedule_dataset_queue_persist()

    def _refresh_current_dataset_progress(self, affected_track_ids=()) -> None:
        if self.session is None:
            return
        entry = self.current_dataset_entry()
        if entry is None:
            return
        if self._active_track_ids is None:
            self._active_track_ids = set(self.session.active_nonempty_track_ids)
            self._verified_track_ids = {
                track_id
                for track_id in self._active_track_ids
                if self.session.track_status.get(track_id) == TRACK_STATUS_VERIFIED
            }
        else:
            if self._verified_track_ids is None:
                self._verified_track_ids = set()
            for track_id in affected_track_ids:
                if track_id < 0 or track_id >= self.session.n_tracks:
                    continue
                if (
                    self.session.is_track_active(track_id)
                    and np.any(~np.isnan(self.session.tracks[track_id]))
                ):
                    self._active_track_ids.add(track_id)
                else:
                    self._active_track_ids.discard(track_id)
                if (
                    track_id in self._active_track_ids
                    and self.session.track_status.get(track_id)
                    == TRACK_STATUS_VERIFIED
                ):
                    self._verified_track_ids.add(track_id)
                else:
                    self._verified_track_ids.discard(track_id)
        entry.set_progress(
            len(self._verified_track_ids),
            len(self._active_track_ids),
        )

    # --------------------------------------------------------------- switching

    def on_dataset_double_clicked(self, item: QListWidgetItem) -> None:
        source_path = item.data(Qt.UserRole)
        index = self.dataset_queue.index_for_path(source_path)
        if index is not None:
            self.request_dataset_load(index)

    def load_next_dataset(self) -> None:
        current_entry = self.current_dataset_entry()
        if current_entry is None:
            return
        current_index = self.dataset_queue.index_for_path(current_entry.source_path)
        if current_index is None:
            return
        next_index = self.dataset_queue.next_index(current_index)
        if next_index is not None:
            self.request_dataset_load(next_index)

    def request_dataset_load(self, index: int) -> bool:
        if index < 0 or index >= len(self.dataset_queue.entries):
            return False
        entry = self.dataset_queue.entries[index]
        if not entry.source.is_file():
            QMessageBox.warning(
                self,
                "Dataset file is missing",
                f"The source MAT file no longer exists:\n{entry.source_path}",
            )
            return False

        if self.session is not None and self._session_dirty:
            choice = self._choose_dirty_dataset_switch(entry)
            if choice == "cancel":
                return False
            if choice == "save" and not self.save_dataset_progress():
                return False
        elif not self._confirm_dataset_load(entry):
            return False

        return bool(self.open_path(entry.source_path))

    def _confirm_dataset_load(self, entry: DatasetQueueEntry) -> bool:
        return ask_yes_no(
            self,
            "Load dataset?",
            f"Load this dataset?\n\n{entry.source.name}",
        )

    def _choose_dirty_dataset_switch(self, entry: DatasetQueueEntry) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Current dataset has newer edits")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            "The current session has changed since its last Save progress.\n\n"
            f"Load {entry.source.name}?"
        )
        save_button = box.addButton("Save & load", QMessageBox.AcceptRole)
        load_button = box.addButton("Load without saving", QMessageBox.DestructiveRole)
        cancel_button = box.addButton(QMessageBox.Cancel)
        save_button.setEnabled(self.current_dataset_entry() is not None)
        box.setDefaultButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_button:
            return "save"
        if clicked is load_button:
            return "load"
        return "cancel"

    # ------------------------------------------------------------------- save

    def save_dataset_progress(self) -> bool:
        entry = self.current_dataset_entry()
        if self.session is None or self.current_path is None or entry is None:
            return False
        if self._autosave_blocked:
            QMessageBox.warning(
                self,
                "Save progress disabled",
                "This source does not have an exclusive, writable autosave lease. "
                "Use Save As for an emergency snapshot, then resolve the sidecar "
                "or concurrent-session problem before continuing.",
            )
            return False
        if canonical_dataset_path(self.current_path) != entry.source_path:
            return False
        if entry.corrected_path.resolve(strict=False) == self.current_path.resolve(
            strict=False
        ):
            QMessageBox.critical(
                self,
                "Source MAT is protected",
                "The corrected/ destination resolves to the source MAT. Fix the "
                "directory or symlink before saving progress.",
            )
            return False
        self._refresh_current_dataset_progress()
        self.session.metadata["dataset_review_state"] = DATASET_STATUS_IN_PROGRESS
        try:
            save_correction_session_atomic(self.session, entry.corrected_path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save dataset progress", str(exc))
            return False

        requested_status = self._choose_dataset_status_after_save(entry.corrected_path)
        if (
            requested_status == DATASET_STATUS_COMPLETED
            and not self._confirm_incomplete_dataset_completion()
        ):
            requested_status = DATASET_STATUS_IN_PROGRESS

        self.session.metadata["dataset_review_state"] = requested_status
        try:
            save_correction_session_atomic(self.session, entry.corrected_path)
        except Exception as exc:
            self.session.metadata["dataset_review_state"] = DATASET_STATUS_IN_PROGRESS
            QMessageBox.critical(
                self,
                "Could not persist dataset review state",
                "A valid in-progress snapshot remains on disk, but the requested "
                f"review classification was not saved.\n\n{exc}",
            )
            return False

        entry.status = requested_status
        entry.last_saved_at = datetime.now(timezone.utc).isoformat()
        self._session_dirty = False
        self.persist_dataset_queue()
        self._update_dataset_item(entry.source_path)
        self._sync_dataset_controls()
        self.status_label.setText(
            f"Saved dataset progress to {entry.corrected_path}."
        )
        return True

    def _confirm_incomplete_dataset_completion(self) -> bool:
        if self.session is None:
            return False
        unresolved_ids = [
            track_id
            for track_id in self.session.active_nonempty_track_ids
            if self.session.track_status.get(track_id) != TRACK_STATUS_VERIFIED
        ]
        empty_ids = self.session.active_empty_track_ids
        if not unresolved_ids and not empty_ids:
            return True

        details = []
        if unresolved_ids:
            details.append(
                f"{len(unresolved_ids)} active non-empty track(s) are not verified"
            )
        if empty_ids:
            details.append(f"{len(empty_ids)} active track draft(s) are empty")
        verified_count = len(self.session.active_nonempty_track_ids) - len(
            unresolved_ids
        )
        return ask_yes_no(
            self,
            "Complete with unresolved tracks?",
            "This dataset has not been fully reviewed:\n\n- "
            + "\n- ".join(details)
            + f"\n\nOnly the {verified_count} verified track(s) will be included "
            "in the corrected MAT. The remaining tracks will not be exported.\n\n"
            "Mark this dataset completed anyway?",
        )

    def _choose_dataset_status_after_save(self, corrected_path: Path) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Dataset progress saved")
        box.setIcon(QMessageBox.Information)
        box.setText(
            f"Saved corrected result:\n{corrected_path}\n\n"
            "How should this dataset appear in the queue?"
        )
        completed_button = box.addButton("Mark completed", QMessageBox.AcceptRole)
        flagged_button = box.addButton("Flag for later", QMessageBox.ActionRole)
        progress_button = box.addButton("Keep in progress", QMessageBox.RejectRole)
        box.setDefaultButton(progress_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is completed_button:
            return DATASET_STATUS_COMPLETED
        if clicked is flagged_button:
            return DATASET_STATUS_FLAGGED
        return DATASET_STATUS_IN_PROGRESS
