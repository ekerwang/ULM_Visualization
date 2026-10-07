"""Session open/save and autosave edit-log wiring for the main window."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.dataset_accumulation import (
    DatasetAccumulator,
    DatasetAccumulationSettings,
)
from ulm_track_correction_gui.gui.confirmation_dialog import ask_yes_no
from ulm_track_correction_gui.gui.layer_order import LAYER_LABELS
from ulm_track_correction_gui.gui.main_window_datasetlist import DatasetImportDialog
from ulm_track_correction_gui.io.accumulation_checkpoint_io import (
    discover_accumulation_checkpoint,
    load_accumulation_checkpoint,
)
from ulm_track_correction_gui.io.editlog import (
    EditLogIntegrityError,
    EditLogLock,
    EditLogWriter,
    editlog_path_for,
    read_edit_log_bundle,
    replay_edits,
)
from ulm_track_correction_gui.io.mat_io import (
    load_correction_session,
    save_correction_session_atomic,
)


class SessionIoMixin:
    """Loads/saves MAT sessions and manages the autosave edit log."""

    def open_mat(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open correction input",
            "",
            "MAT files (*.mat);;All files (*)",
        )
        if not path:
            return
        self.open_path(path)

    def open_path(self, path: str) -> bool:
        try:
            session = load_correction_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load MAT file", str(exc))
            return False

        self.stop_playback()
        self._close_editlog()
        self._close_history_dialog()
        self.session = session
        self.current_path = Path(path)
        self._editlog_baseline_fingerprint = session.state_fingerprint()
        self.session.metadata.setdefault(
            "source_correction_fingerprint",
            self._editlog_baseline_fingerprint,
        )
        self.session.metadata["active_editlog_baseline_fingerprint"] = (
            self._editlog_baseline_fingerprint
        )
        self._autosave_blocked = False
        self.selected_track_id = None
        self.undo_manager = UndoManager(session)
        self.mode_buttons["inspect"].setChecked(True)

        recovered_edits = self._offer_editlog_recovery()
        self._attach_editlog_writer()
        self.session.edit_listener = self._on_session_edit

        self.viewer.set_session(self.session)
        self._load_discovered_accumulation_checkpoint()
        self.populate_track_list()
        self.configure_frame_controls()
        self.on_dataset_session_loaded(path, recovered_edits=recovered_edits)
        self.status_label.setText(
            f"Loaded {self.session.n_tracks} tracks across "
            f"{self.session.n_frames} frames."
        )
        self._report_import_warnings()
        self._sync_enabled_state()
        self.viewer.setFocus()
        return True

    def _load_discovered_accumulation_checkpoint(self) -> None:
        if self.session is None or self.current_path is None:
            return
        discovery = discover_accumulation_checkpoint(
            self.current_path,
            self.session.metadata,
        )
        if discovery.warnings:
            self.session.metadata.setdefault("import_warnings", []).extend(
                discovery.warnings
            )
        if discovery.path is None:
            self.accumulation_checkpoint = None
            self.viewer.set_accumulation_checkpoint(None)
            self.display_panel.configure_track_plotting(
                (),
                "No companion Batch checkpoint was found. Velocity overlays "
                "depend on the checkpoint schema and contribution mode.",
            )
            self.display_panel.configure_dataset_plotting(
                False,
                "No companion Batch checkpoint was found; its PALA grid and "
                f"parameters are required for {LAYER_LABELS['dataset_plot']}.",
            )
            return
        self._activate_accumulation_checkpoint(discovery.path, report_dialog=False)

    def _activate_accumulation_checkpoint(
        self,
        path: str | Path,
        *,
        report_dialog: bool,
    ) -> bool:
        if self.session is None:
            return False
        try:
            checkpoint = load_accumulation_checkpoint(path)
            if self.session.image_stack is not None:
                checkpoint.validate_input_shape(self.session.image_stack.shape[:2])
            track_id_map = self.session.current_to_source_track_ids()
        except Exception as exc:
            message = f"Accumulation checkpoint overlay is unavailable: {exc}"
            self.accumulation_checkpoint = None
            self.viewer.set_accumulation_checkpoint(None)
            self.display_panel.configure_track_plotting((), message)
            self.display_panel.configure_dataset_plotting(False, message)
            if report_dialog:
                QMessageBox.warning(self, "Could not load checkpoint", message)
            else:
                self.session.metadata.setdefault("import_warnings", []).append(message)
            return False

        self.accumulation_checkpoint = checkpoint
        self.session.metadata["accumulation_checkpoint_path"] = str(
            checkpoint.source_path.resolve(strict=False)
        )
        if checkpoint.mode == "none":
            message = (
                f"The schema v{checkpoint.schema_version} Batch checkpoint uses "
                "perTrackContributionMode=none; "
                f"{LAYER_LABELS['track_plot']} is unavailable."
            )
        else:
            count = len(checkpoint.supported_track_maps)
            count_text = {2: "two", 4: "four"}.get(count, str(count))
            message = (
                f"Schema v{checkpoint.schema_version} {checkpoint.mode} checkpoint "
                f"loaded; {count_text} selected-track maps are available."
            )
        self.display_panel.configure_track_plotting(
            checkpoint.supported_track_maps,
            message,
            velocity_display_max=float(checkpoint.params["velocityDisplayMax"]),
        )
        try:
            DatasetAccumulationSettings.from_checkpoint(checkpoint)
        except ValueError as exc:
            self.display_panel.configure_dataset_plotting(
                False,
                f"{LAYER_LABELS['dataset_plot']} is unavailable: {exc}",
            )
        else:
            self.display_panel.configure_dataset_plotting(
                True,
                "Ready to accumulate every current track on the original PALA grid. "
                "Enable this layer to build it.",
            )
            self.viewer.set_dataset_plot_map(
                str(self.display_panel.dataset_plot_map_combo.currentData())
            )
        self.viewer.set_track_plot_map(
            str(self.display_panel.track_plot_map_combo.currentData())
        )
        self.viewer.set_accumulation_checkpoint(checkpoint, track_id_map)
        if report_dialog:
            self.status_label.setText(f"Loaded accumulation checkpoint: {path}")
        return True

    def load_accumulation_checkpoint_manually(self) -> None:
        if self.session is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Batch accumulation checkpoint",
            str(self.current_path.parent if self.current_path is not None else ""),
            "MAT files (*.mat);;All files (*)",
        )
        if path:
            self._activate_accumulation_checkpoint(path, report_dialog=True)

    def import_multiple_dataset_accumulation(self) -> None:
        """Explicitly replace the retained multi-dataset ULM snapshot."""

        start_dir = str(
            self.current_path.parent if self.current_path is not None else ""
        )
        dialog = DatasetImportDialog(
            self,
            start_dir=start_dir,
            title=f"Import datasets for {LAYER_LABELS['multiple_dataset_plot']}",
        )
        if dialog.exec() != QFileDialog.Accepted:
            return
        selected_paths = dialog.selectedFiles()
        if not selected_paths:
            return

        paths: list[Path] = []
        seen: set[str] = set()
        for selected_path in selected_paths:
            path = Path(selected_path).resolve(strict=False)
            key = str(path)
            if key not in seen:
                paths.append(path)
                seen.add(key)

        def loaded_datasets():
            for index, path in enumerate(paths, start=1):
                self.status_label.setText(
                    f"Importing accumulation dataset {index}/{len(paths)}: {path.name}"
                )
                QApplication.processEvents()
                try:
                    session = load_correction_session(path)
                    discovery = discover_accumulation_checkpoint(
                        path,
                        session.metadata,
                    )
                    if discovery.path is None:
                        detail = (
                            f" ({'; '.join(discovery.warnings)})"
                            if discovery.warnings
                            else ""
                        )
                        raise ValueError(
                            "no companion Batch accumulation checkpoint was found"
                            f"{detail}"
                        )
                    checkpoint = load_accumulation_checkpoint(discovery.path)
                except Exception as exc:
                    raise ValueError(f"{path.name}: {exc}") from exc
                yield session, checkpoint

        old_status = self.status_label.text()
        self.display_panel.import_multiple_dataset_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            accumulator = DatasetAccumulator.build_multiple(loaded_datasets())
        except Exception as exc:
            self.status_label.setText(old_status)
            QMessageBox.warning(
                self,
                "Could not import multiple datasets",
                f"The existing {LAYER_LABELS['multiple_dataset_plot']} was kept "
                "unchanged.\n\n"
                f"{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.display_panel.import_multiple_dataset_button.setEnabled(True)

        self.viewer.set_multiple_dataset_accumulation(
            accumulator,
            tuple(str(path) for path in paths),
        )
        self.display_panel.configure_multiple_dataset_plotting(
            True,
            f"Imported {accumulator.dataset_count} datasets with "
            f"{accumulator.track_count} tracks. This snapshot is retained when "
            "the current dataset changes.",
        )
        self.viewer.set_multiple_dataset_plot_map(
            str(self.display_panel.multiple_dataset_plot_map_combo.currentData())
        )
        self.display_panel.multiple_dataset_plot_group.setChecked(True)
        self.status_label.setText(
            f"Imported {accumulator.dataset_count} datasets into "
            f"{LAYER_LABELS['multiple_dataset_plot']} "
            f"({accumulator.track_count} tracks)."
        )

    def _offer_editlog_recovery(self) -> bool:
        if self.session is None or self.current_path is None:
            return False
        log_path = editlog_path_for(self.current_path)
        try:
            self._editlog_lock = EditLogLock(log_path)
        except OSError as exc:
            self._autosave_blocked = True
            QMessageBox.warning(
                self,
                "Editing disabled: dataset already in use",
                "Another process holds the correction sidecar lock. This session "
                "is read-only so its recovery log cannot be renamed or split "
                f"between processes.\n\n{exc}",
            )
            return False
        try:
            bundle = read_edit_log_bundle(log_path)
        except EditLogIntegrityError as exc:
            archived = self._archive_editlog(log_path, "invalid")
            QMessageBox.warning(
                self,
                "Edit log rejected",
                "The edit log failed its integrity checks and was NOT replayed. "
                f"It was preserved as:\n{archived.name}\n\n{exc}",
            )
            return False
        if not bundle.edits:
            return False
        if (
            bundle.baseline_fingerprint is not None
            and bundle.baseline_fingerprint != self._editlog_baseline_fingerprint
        ):
            archived = self._archive_editlog(log_path, "baseline-mismatch")
            QMessageBox.warning(
                self,
                "Edit log belongs to different data",
                "The sidecar fingerprint does not match the loaded localization/track "
                "baseline. Nothing was replayed. The log was preserved as:\n"
                f"{archived.name}",
            )
            return False
        edits = bundle.edits
        legacy_warning = (
            "\n\nThis log has no source fingerprint. Verify the "
            "source file carefully before accepting recovery."
            if bundle.baseline_fingerprint is None
            else ""
        )
        tail_warning = (
            "\n\nA corrupt final tail was detected. Only the validated prefix will "
            "be replayed; the writer will archive and repair the tail."
            if bundle.corrupt_tail_offset is not None
            else ""
        )
        recover = ask_yes_no(
            self,
            "Recover previous edits?",
            f"Found an edit log with {len(edits)} recorded edit(s) for this "
            f"file:\n{log_path.name}\n\nReplay them onto the loaded data?"
            f"{legacy_warning}{tail_warning}",
        )
        if recover:
            try:
                replay_edits(self.session, edits)
            except Exception as exc:
                archived = self._archive_editlog(log_path, "failed-replay")
                QMessageBox.warning(
                    self,
                    "Recovery rejected",
                    "Replay failed transactionally; the loaded session was left "
                    "unchanged. The log was preserved as "
                    f"{archived.name}.\n\n{exc}",
                )
                return False
            return True
        else:
            self._archive_editlog(log_path, "discarded")
        return False

    @staticmethod
    def _archive_editlog(log_path: Path, reason: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        archived = log_path.with_name(f"{log_path.name}.{reason}-{stamp}")
        suffix = 1
        while archived.exists():
            archived = log_path.with_name(
                f"{log_path.name}.{reason}-{stamp}-{suffix}"
            )
            suffix += 1
        log_path.rename(archived)
        return archived

    def _attach_editlog_writer(self) -> None:
        if self.session is None or self.current_path is None:
            return
        if self._autosave_blocked and self._editlog_lock is None:
            return
        lock = self._editlog_lock
        self._editlog_lock = None
        try:
            self.editlog_writer = EditLogWriter(
                editlog_path_for(self.current_path),
                baseline_fingerprint=self._editlog_baseline_fingerprint,
                source_path=self.current_path,
                lock=lock,
            )
            self._autosave_blocked = False
        except (OSError, ValueError) as exc:
            self.editlog_writer = None
            self._autosave_blocked = True
            QMessageBox.warning(
                self,
                "Editing disabled: autosave unavailable",
                f"Could not open the edit log for writing; edits will NOT be "
                "allowed until this is resolved.\n\n"
                f"{exc}",
            )

    def _close_editlog(self) -> None:
        if self.session is not None:
            self.session.edit_listener = None
        if self.editlog_writer is not None:
            self.editlog_writer.close()
            self.editlog_writer = None
        if self._editlog_lock is not None:
            self._editlog_lock.close()
            self._editlog_lock = None

    def _report_import_warnings(self) -> None:
        if self.session is None:
            return
        warnings = list(self.session.metadata.get("import_warnings", []))
        if warnings:
            QMessageBox.warning(self, "Import warnings", "\n".join(warnings))

    def save_mat_as(self) -> None:
        if self.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save corrected result",
            "",
            "MAT files (*.mat);;All files (*)",
        )
        if not path:
            return
        destination = Path(path).resolve(strict=False)
        protected = (
            {self.current_path.resolve(strict=False)}
            if self.current_path is not None
            else set()
        )
        source_path = self.session.metadata.get("source_path")
        if source_path:
            protected.add(Path(source_path).expanduser().resolve(strict=False))
        if destination in protected:
            QMessageBox.critical(
                self,
                "Source MAT is protected",
                "The corrected export cannot overwrite its source MAT. Choose a "
                "different filename or use Save progress, which writes into corrected/.",
            )
            return
        verification_issues = self.session.integrity_issues(require_all_verified=True)
        if verification_issues and not self._confirm(
            "Export verified tracks only?",
            "This session still has unresolved tracks:\n\n- "
            + "\n- ".join(verification_issues)
            + "\n\nOnly verified tracks will be included in the corrected MAT. "
            "Export anyway?",
        ):
            return
        try:
            save_correction_session_atomic(self.session, destination)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save MAT file", str(exc))
            return
        self.status_label.setText(f"Saved corrected result to {path}.")
