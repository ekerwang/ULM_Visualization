"""Two-line, status-colored rendering of track list items.

Information hierarchy instead of one pipe-separated string: the track ID is
the primary line (bold) with the review status as a colored label on the
right; counts and frame range form a dimmer secondary line. A thin bar on the
left edge repeats the status color so the list can be scanned by color alone.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

SUMMARY_ROLE = int(Qt.ItemDataRole.UserRole) + 1

STATUS_COLORS = {
    "unreviewed": "#8a8f98",
    "edited": "#e0a63c",
    "verified": "#3fb950",
    "flagged": "#e5484d",
}


class TrackItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        summary = index.data(SUMMARY_ROLE)
        if not summary:
            super().paint(painter, option, index)
            return

        painter.save()
        rect = option.rect
        selected = bool(option.state & QStyle.State_Selected)
        palette = option.palette
        if selected:
            painter.fillRect(rect, palette.highlight())

        status = summary.get("status", "unreviewed")
        status_color = QColor(STATUS_COLORS.get(status, "#8a8f98"))
        painter.fillRect(rect.x(), rect.y() + 3, 3, rect.height() - 6, status_color)

        primary = (
            palette.highlightedText().color() if selected else palette.text().color()
        )
        secondary = QColor(primary)
        secondary.setAlpha(140)

        line_height = (rect.height() - 8) // 2
        line1 = QRect(rect.x() + 10, rect.y() + 4, rect.width() - 16, line_height)
        line2 = QRect(
            rect.x() + 10,
            rect.y() + 4 + line_height,
            rect.width() - 16,
            line_height,
        )

        bold = QFont(option.font)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(primary)
        painter.drawText(
            line1,
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Track {summary['track_id']}",
        )

        status_font = QFont(option.font)
        status_font.setPointSizeF(max(option.font.pointSizeF() * 0.85, 6.0))
        painter.setFont(status_font)
        painter.setPen(status_color)
        painter.drawText(line1, Qt.AlignRight | Qt.AlignVCenter, status)

        start = summary.get("start_frame")
        if start is None:
            detail = "empty"
        else:
            detail = (
                f"{summary['n_points']} pts · {summary['n_gaps']} gaps · "
                f"frames {start + 1}–{summary['end_frame'] + 1}"
            )
        painter.setPen(secondary)
        painter.drawText(line2, Qt.AlignLeft | Qt.AlignVCenter, detail)

        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        metrics = option.fontMetrics
        return QSize(option.rect.width(), metrics.height() * 2 + 10)
