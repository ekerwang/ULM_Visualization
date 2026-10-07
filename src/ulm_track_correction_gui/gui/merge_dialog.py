"""Merge-tracks dialog: pick the two tracks by ID with live summaries.

Typing IDs (rather than hunting in the list) is the primary flow because real
datasets have thousands of tracks. Live summary labels under each ID confirm
the right tracks were entered before committing.
"""

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


def _describe(session: CorrectionSession, track_id: int) -> str:
    summary = session.track_summary(track_id)
    if summary["start_frame"] is None:
        return "empty track"
    return (
        f"{summary['n_points']} pts · {summary['n_gaps']} gaps · frames "
        f"{summary['start_frame'] + 1}–{summary['end_frame'] + 1} · "
        f"{summary['status']}"
    )


class MergeTracksDialog(QDialog):
    """Collects (target_id, source_id) for CorrectionSession.merge_tracks."""

    def __init__(
        self,
        session: CorrectionSession,
        target_id: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge tracks")
        self._session = session

        max_id = max(session.n_tracks - 1, 0)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, max_id)
        self.source_spin = QSpinBox()
        self.source_spin.setRange(0, max_id)
        if target_id is not None:
            self.target_spin.setValue(int(target_id))
        self.target_info = QLabel()
        self.source_info = QLabel()
        self.overlap_label = QLabel()
        self.overlap_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Merge into track", self.target_spin)
        form.addRow("", self.target_info)
        form.addRow("Absorb track", self.source_spin)
        form.addRow("", self.source_info)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.button(QDialogButtonBox.Ok).setText("Merge")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.overlap_label)
        layout.addWidget(
            QLabel(
                "The absorbed track's points move into the target; overlapping\n"
                "frames keep the target's point. Saved Track IDs stay unchanged."
            )
        )
        layout.addWidget(self.button_box)

        self.target_spin.valueChanged.connect(self._refresh)
        self.source_spin.valueChanged.connect(self._refresh)
        self._refresh()

    @property
    def target_id(self) -> int:
        return int(self.target_spin.value())

    @property
    def source_id(self) -> int:
        return int(self.source_spin.value())

    def overlap_count(self) -> int:
        if self.target_id == self.source_id:
            return 0
        target = self._session.tracks[self.target_id]
        source = self._session.tracks[self.source_id]
        return int(np.sum(~np.isnan(target) & ~np.isnan(source)))

    def _refresh(self) -> None:
        session = self._session
        self.target_info.setText(_describe(session, self.target_id))
        self.source_info.setText(_describe(session, self.source_id))

        distinct = self.target_id != self.source_id
        target_ok = session.track_summary(self.target_id)["start_frame"] is not None
        source_ok = session.track_summary(self.source_id)["start_frame"] is not None
        valid = distinct and target_ok and source_ok

        if not distinct:
            self.overlap_label.setText("Pick two different tracks.")
        elif not (target_ok and source_ok):
            self.overlap_label.setText("Both tracks must have points.")
        else:
            overlap = self.overlap_count()
            self.overlap_label.setText(
                "No overlapping frames."
                if overlap == 0
                else (
                    f"⚠ {overlap} overlapping frame(s): the target keeps its "
                    "points there; the absorbed track's points at those frames "
                    "become unassigned."
                )
            )
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(valid)
