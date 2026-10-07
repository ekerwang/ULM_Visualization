"""Split-track dialog with a frame-boundary preview."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ulm_track_correction_gui.core.correction_session import CorrectionSession


class SplitTrackDialog(QDialog):
    """Choose the last frame belonging to the first split child."""

    def __init__(
        self,
        session: CorrectionSession,
        track_id: int,
        current_frame: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Split track")
        self._session = session
        self.track_id = int(track_id)

        summary = session.track_summary(self.track_id)
        start = summary["start_frame"]
        end = summary["end_frame"]
        if start is None or end is None or start >= end:
            raise ValueError("A track needs points on at least two frames to be split.")

        self.boundary_spin = QSpinBox()
        self.boundary_spin.setRange(start + 1, end)
        self.boundary_spin.setValue(max(start + 1, min(int(current_frame), end)))
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Parent track", QLabel(str(self.track_id)))
        form.addRow("Split after frame", self.boundary_spin)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Ok).setText("Split")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.preview_label)
        layout.addWidget(
            QLabel(
                "The parent is retired. Two new track IDs are created and both "
                "are flagged for review."
            )
        )
        layout.addWidget(self.button_box)

        self.boundary_spin.valueChanged.connect(self._refresh)
        self._refresh()

    @property
    def split_after_frame(self) -> int:
        """Return the zero-based frame index used by ``CorrectionSession``."""

        return int(self.boundary_spin.value()) - 1

    def _refresh(self) -> None:
        track = self._session.tracks[self.track_id]
        frames = np.flatnonzero(~np.isnan(track))
        boundary = self.split_after_frame
        first_count = int(np.sum(frames <= boundary))
        second_count = int(np.sum(frames > boundary))
        first_id = self._session.n_tracks
        second_id = first_id + 1
        self.preview_label.setText(
            f"Track {first_id}: {first_count} point(s), frames ≤ {boundary + 1}\n"
            f"Track {second_id}: {second_count} point(s), frames > {boundary + 1}"
        )
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(
            first_count > 0 and second_count > 0
        )
