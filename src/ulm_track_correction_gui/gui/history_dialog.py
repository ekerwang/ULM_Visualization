"""Read-only, per-track edit history dialog.

Shows a snapshot of the track's edit-log entries in plain language (built by
``core.edit_history``). Non-modal so the user can keep it open while stepping
frames; double-clicking an entry jumps the main viewer to that frame.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.edit_history import track_history_entries

JUMP_FRAME_ROLE = int(Qt.ItemDataRole.UserRole)


class TrackHistoryDialog(QDialog):
    """Audit view of everything the user did to one track."""

    jump_to_frame = Signal(int)  # 1-based frame number

    def __init__(
        self,
        session: CorrectionSession,
        track_id: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.track_id = track_id
        self.setWindowTitle(f"Track {track_id} — Edit History")
        self.resize(600, 440)

        entries = track_history_entries(session, track_id)
        status = session.track_status.get(track_id, "unreviewed")
        if entries:
            header_text = (
                f"Track {track_id} · status: {status} · "
                f"{len(entries)} operation(s), newest first"
            )
        else:
            header_text = (
                f"Track {track_id} · status: {status} · "
                "no edits recorded in this session"
            )
        header = QLabel(header_text)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Time", "Frame", "What happened"])
        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 70)
        self.tree.header().setStretchLastSection(True)
        self.tree.setAlternatingRowColors(True)
        self._populate(entries)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        hint = QLabel("Double-click an entry to jump to its frame.")
        hint.setStyleSheet("color: gray;")
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(hint)
        bottom.addStretch(1)
        bottom.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.tree, stretch=1)
        layout.addLayout(bottom)

    def _populate(self, entries: list[dict]) -> None:
        bold = QFont(self.tree.font())
        bold.setBold(True)
        for entry in entries:
            top = QTreeWidgetItem(
                [entry["time"], entry["frames"], entry["summary"]]
            )
            if entry["jump_frame"] is not None:
                top.setData(0, JUMP_FRAME_ROLE, entry["jump_frame"])
            if entry["details"]:
                top.setFont(2, bold)
                for detail in entry["details"]:
                    child = QTreeWidgetItem(
                        [detail["time"], detail["frame"], detail["text"]]
                    )
                    if detail["jump_frame"] is not None:
                        child.setData(0, JUMP_FRAME_ROLE, detail["jump_frame"])
                    top.addChild(child)
            self.tree.addTopLevelItem(top)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        frame = item.data(0, JUMP_FRAME_ROLE)
        if frame is not None:
            self.jump_to_frame.emit(int(frame))
