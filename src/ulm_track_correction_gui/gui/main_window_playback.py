"""Frame navigation and playback behavior of the main window."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)

from ulm_track_correction_gui.gui.confirmation_dialog import ask_yes_no
from ulm_track_correction_gui.gui.main_window_ui import (
    SCOPE_FULL_STACK,
    SCOPE_TRACK_PADDED,
    SCOPE_TRACK_RANGE,
)
from ulm_track_correction_gui.gui.movie_export import (
    MovieWatermark,
    Mp4FrameWriter,
    annotate_movie_frame,
    normalized_mp4_path,
)


class MovieExportCancelled(RuntimeError):
    """Raised when the user cancels an in-progress movie export."""


class PlaybackMixin:
    """Synchronize scope sliders, the absolute frame editor, and playback."""

    def configure_frame_controls(self) -> None:
        """Reset navigation for a newly loaded session."""

        self.playback_scope = SCOPE_FULL_STACK
        self.frame_slider = self.scope_sliders[SCOPE_FULL_STACK]
        self.scope_buttons[SCOPE_FULL_STACK].setChecked(True)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(1)
        self.frame_spin.blockSignals(False)
        self.refresh_frame_scope_controls(current_frame=1)
        self._set_current_frame(1)

    def _selected_track_span_1based(self) -> tuple[int, int] | None:
        if self.session is None or self.selected_track_id is None:
            return None
        summary = self.session.track_summary(self.selected_track_id)
        if summary["start_frame"] is None:
            return None
        return summary["start_frame"] + 1, summary["end_frame"] + 1

    def refresh_frame_scope_controls(self, *, current_frame: int | None = None) -> None:
        """Recompute all three ranges after session, track, edit, or N changes."""

        has_session = self.session is not None
        max_frame = max(1, self.session.n_frames) if has_session else 1
        track_span = self._selected_track_span_1based()

        if track_span is None and self.playback_scope != SCOPE_FULL_STACK:
            # Track-dependent scopes cannot represent a frame without a valid
            # selected track. Fall back without recursively re-entering this
            # method through the radio-button signal.
            self.playback_scope = SCOPE_FULL_STACK
            self.frame_slider = self.scope_sliders[SCOPE_FULL_STACK]
            button = self.scope_buttons[SCOPE_FULL_STACK]
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)

        if current_frame is None:
            current_frame = self.frame_spin.value()
        current_frame = max(1, min(int(current_frame), max_frame))

        ranges: dict[str, tuple[int, int]] = {
            SCOPE_FULL_STACK: (1, max_frame),
            SCOPE_TRACK_RANGE: (1, 1),
            SCOPE_TRACK_PADDED: (1, 1),
        }
        if track_span is not None:
            track_start, track_end = track_span
            padding = self.pad_spin.value()
            ranges[SCOPE_TRACK_RANGE] = track_span
            ranges[SCOPE_TRACK_PADDED] = (
                max(1, track_start - padding),
                min(max_frame, track_end + padding),
            )

        for scope, slider in self.scope_sliders.items():
            start, end = ranges[scope]
            slider.blockSignals(True)
            slider.setRange(start, end)
            slider.setValue(max(start, min(current_frame, end)))
            slider.blockSignals(False)
        padded_slider = self.scope_sliders[SCOPE_TRACK_PADDED]
        if track_span is None:
            padded_slider.set_track_span(None, None)
        else:
            padded_slider.set_track_span(*track_span)

        active_start, active_end = ranges[self.playback_scope]
        active_frame = max(active_start, min(current_frame, active_end))
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(active_start, active_end)
        self.frame_spin.setValue(active_frame)
        self.frame_spin.blockSignals(False)

        track_available = has_session and track_span is not None
        availability = {
            SCOPE_FULL_STACK: has_session,
            SCOPE_TRACK_RANGE: track_available,
            SCOPE_TRACK_PADDED: track_available,
        }
        for scope in self.scope_sliders:
            available = availability[scope]
            active = scope == self.playback_scope
            self.scope_buttons[scope].setEnabled(available)
            self.scope_sliders[scope].setEnabled(available and active)
            if hasattr(self, "scope_actions"):
                self.scope_actions[scope].setEnabled(available)
                self.scope_actions[scope].setChecked(active)
        self.scope_stack.setCurrentWidget(self.scope_sliders[self.playback_scope])
        self.scope_context_label.setEnabled(has_session)
        self.padding_controls.setVisible(
            track_available and self.playback_scope == SCOPE_TRACK_PADDED
        )
        self.pad_spin.setEnabled(
            track_available and self.playback_scope == SCOPE_TRACK_PADDED
        )

        if has_session and active_frame != current_frame:
            self.viewer.set_frame(active_frame - 1)
            self._update_gap_hint(active_frame - 1)
        self._update_scope_context(active_frame, track_span, max_frame)

    def _update_scope_context(
        self,
        current_frame: int,
        track_span: tuple[int, int] | None,
        max_frame: int,
    ) -> None:
        """Update the single secondary readout for the active timeline."""

        if self.session is None:
            self.scope_context_label.setText("No stack loaded")
            return
        if self.playback_scope == SCOPE_FULL_STACK or track_span is None:
            self.scope_context_label.setText(f"of {max_frame}")
            return

        track_start, track_end = track_span
        track_length = track_end - track_start + 1
        if self.playback_scope == SCOPE_TRACK_RANGE:
            relative_frame = current_frame - track_start + 1
            self.scope_context_label.setText(
                f"{relative_frame} / {track_length} in track range"
            )
            return

        if current_frame < track_start:
            region = "before track"
        elif current_frame > track_end:
            region = "after track"
        else:
            region = "inside track"
        self.scope_context_label.setText(
            f"Track frames {track_start}–{track_end}  ·  {region}"
        )

    def set_playback_scope(self, scope: str) -> None:
        if scope not in self.scope_sliders:
            raise ValueError(f"Unknown playback scope: {scope}")
        if scope != SCOPE_FULL_STACK and self._selected_track_span_1based() is None:
            return
        self.stop_playback()
        self.playback_scope = scope
        self.frame_slider = self.scope_sliders[scope]
        button = self.scope_buttons[scope]
        if not button.isChecked():
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
        self.scope_stack.setCurrentWidget(self.frame_slider)
        self.refresh_frame_scope_controls()
        self._set_current_frame(self.frame_spin.value())

    def on_track_padding_changed(self, _value: int) -> None:
        self.refresh_frame_scope_controls()
        self._set_current_frame(self.frame_spin.value())

    def on_scope_slider_changed(self, scope: str, value: int) -> None:
        if scope == self.playback_scope:
            self._set_current_frame(value)

    def on_frame_changed(self, value: int) -> None:
        """Compatibility handler for callers using the active frame slider."""

        self._set_current_frame(value)

    def on_frame_spin_changed(self, value: int) -> None:
        self._set_current_frame(value)

    def _set_current_frame(self, frame_1based: int) -> None:
        start, end = self._playback_range()
        value = max(start, min(int(frame_1based), end))

        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        for slider in self.scope_sliders.values():
            slider.blockSignals(True)
            slider.setValue(max(slider.minimum(), min(value, slider.maximum())))
            slider.blockSignals(False)

        self.viewer.set_frame(value - 1)
        if self.correction_mode == "add_point":
            self.schedule_add_point_localization(
                f"Frame {value} · refreshing candidates…"
            )
        self._update_gap_hint(value - 1)
        max_frame = max(1, self.session.n_frames) if self.session is not None else 1
        self._update_scope_context(
            value,
            self._selected_track_span_1based(),
            max_frame,
        )
        self.refresh_movie_watermark_preview()

    def _update_gap_hint(self, frame_idx: int) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        summary = self.session.track_summary(self.selected_track_id)
        start, end = summary["start_frame"], summary["end_frame"]
        if start is None or not (start <= frame_idx <= end):
            return
        if self.session.local_idx_for_track_frame(self.selected_track_id, frame_idx) is None:
            self.status_label.setText(
                f"Frame {frame_idx + 1} is a GAP in track "
                f"{self.selected_track_id} — assign or manually localize."
            )

    def set_frame_1based(self, frame_1based: int) -> None:
        self._set_current_frame(frame_1based)

    def step_frame(self, delta: int) -> None:
        self.set_frame_1based(self.frame_spin.value() + delta)

    # ------------------------------------------------------------- playback

    def export_movie(self) -> None:
        """Ask for an MP4 path and export the active playback scope once."""

        if self.session is None or self._movie_export_in_progress:
            return
        self.stop_playback()
        if self.current_path is not None:
            initial_path = self.current_path.with_name(
                f"{self.current_path.stem}_movie.mp4"
            )
        else:
            initial_path = Path("ulm_track_movie.mp4")
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Movie",
            str(initial_path),
            "MP4 video (*.mp4)",
        )
        if not selected_path:
            return
        requested_path = Path(selected_path).expanduser()
        output_path = normalized_mp4_path(requested_path)
        if output_path != requested_path and output_path.exists():
            if not ask_yes_no(
                self,
                "Replace existing movie?",
                f"{output_path.name} already exists. Replace it?",
            ):
                return

        start, end = self._playback_range()
        progress = QProgressDialog(
            "Exporting movie…",
            "Cancel",
            0,
            end - start + 1,
            self,
        )
        progress.setWindowTitle("Export Movie")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()

        self._movie_export_in_progress = True
        self._sync_enabled_state()

        def report_progress(completed: int, total: int) -> bool:
            progress.setMaximum(total)
            progress.setValue(completed)
            self.status_label.setText(
                f"Exporting movie frame {completed} of {total}…"
            )
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            saved_path = self._export_movie_to_path(
                output_path,
                progress_callback=report_progress,
            )
        except MovieExportCancelled:
            self.status_label.setText("Movie export canceled.")
            return
        except Exception as exc:
            self.status_label.setText("Movie export failed.")
            QMessageBox.critical(self, "Could not export movie", str(exc))
            return
        finally:
            progress.close()
            self._movie_export_in_progress = False
            self._sync_enabled_state()

        self.status_label.setText(f"Movie exported to {saved_path.name}.")
        QMessageBox.information(
            self,
            "Movie export complete",
            f"Movie saved to:\n{saved_path}",
        )

    def _movie_watermark(self, output_path: Path | None = None) -> MovieWatermark:
        if self.current_path is not None:
            filename = self.current_path.name
        elif self.session is not None and self.session.metadata.get("source_path"):
            filename = Path(str(self.session.metadata["source_path"])).name
        else:
            filename = output_path.name if output_path is not None else "movie.mp4"

        track_id = self.selected_track_id
        track_start = None
        track_end = None
        if (
            self.session is not None
            and track_id is not None
            and 0 <= track_id < self.session.n_tracks
        ):
            summary = self.session.track_summary(track_id)
            if summary["start_frame"] is not None:
                track_start = int(summary["start_frame"]) + 1
                track_end = int(summary["end_frame"]) + 1
        else:
            track_id = None
        return MovieWatermark(
            filename=filename,
            track_id=track_id,
            track_start_frame=track_start,
            track_end_frame=track_end,
        )

    def refresh_movie_watermark_preview(self, *_args) -> None:
        """Synchronize the fixed viewer preview with current export metadata."""

        controls_ready = hasattr(self, "watermark_preview_checkbox")
        visible = (
            controls_ready
            and self.session is not None
            and self.watermark_preview_checkbox.isChecked()
            and not self._movie_watermark_preview_suppressed
        )
        watermark = self._movie_watermark() if visible else None
        frame_number = self.frame_spin.value() if controls_ready else 1
        font_size_px = (
            self.watermark_font_size_spin.value() if controls_ready else 12
        )
        self.viewer.set_movie_watermark_preview(
            watermark,
            frame_number,
            font_size_px,
            visible=visible,
        )

    def _export_movie_to_path(
        self,
        path: str | Path,
        *,
        progress_callback: Callable[[int, int], bool] | None = None,
    ) -> Path:
        """Export the visible viewer for every frame in the active scope."""

        if self.session is None:
            raise RuntimeError("Open a MAT session before exporting a movie.")
        output_path = normalized_mp4_path(path)
        if not output_path.parent.exists():
            raise FileNotFoundError(
                f"Movie output folder does not exist: {output_path.parent}"
            )

        start, end = self._playback_range()
        frame_numbers = range(start, end + 1)
        total = end - start + 1
        fps = int(self.fps_spin.value())
        original_frame = int(self.frame_spin.value())
        watermark = self._movie_watermark(output_path)
        font_size_px = int(self.watermark_font_size_spin.value())
        preview_was_suppressed = self._movie_watermark_preview_suppressed
        self._movie_watermark_preview_suppressed = True
        self.refresh_movie_watermark_preview()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".mp4",
            dir=str(output_path.parent),
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        writer: Mp4FrameWriter | None = None

        try:
            for completed, frame_number in enumerate(frame_numbers, start=1):
                self._set_current_frame(frame_number)
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                source = self.viewer.viewport().grab().toImage()
                frame = annotate_movie_frame(
                    source,
                    watermark,
                    frame_number,
                    font_size_px,
                )
                if writer is None:
                    writer = Mp4FrameWriter(
                        temporary_path,
                        (frame.width(), frame.height()),
                        fps,
                    )
                writer.append_frame(frame)
                if progress_callback is not None and not progress_callback(
                    completed,
                    total,
                ):
                    raise MovieExportCancelled
            if writer is None:
                raise RuntimeError("The active playback scope contains no frames.")
            writer.close()
            writer = None
            os.replace(temporary_path, output_path)
        except Exception:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            self._set_current_frame(original_frame)
            self._movie_watermark_preview_suppressed = preview_was_suppressed
            self.refresh_movie_watermark_preview()
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        return output_path

    def _apply_fps(self) -> None:
        self.play_timer.setInterval(max(1, round(1000 / self.fps_spin.value())))

    def on_play_toggled(self, playing: bool) -> None:
        if playing and self.session is None:
            self.play_button.setChecked(False)
            return
        self.play_button.setText("Pause (Space)" if playing else "Play (Space)")
        if playing:
            self._apply_fps()
            start, end = self._playback_range()
            if not (start <= self.frame_spin.value() <= end):
                self.set_frame_1based(start)
            self.play_timer.start()
        else:
            self.play_timer.stop()

    def stop_playback(self) -> None:
        if self.play_button.isChecked():
            self.play_button.setChecked(False)

    def _playback_range(self) -> tuple[int, int]:
        """Active slider's 1-based inclusive frame range."""

        slider = self.scope_sliders[self.playback_scope]
        return slider.minimum(), slider.maximum()

    def _on_play_tick(self) -> None:
        if self.session is None:
            self.stop_playback()
            return
        start, end = self._playback_range()
        following = self.frame_spin.value() + 1
        if following > end or following < start:
            following = start
        self.set_frame_1based(following)
