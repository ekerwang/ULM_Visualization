"""Two-line rendering for the persistent dataset correction queue."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QStyle, QStyledItemDelegate


DATASET_SUMMARY_ROLE = int(Qt.ItemDataRole.UserRole) + 1

DATASET_STATUS_COLORS = {
    "pending": "#8a8f98",
    "in_progress": "#2f81f7",
    "completed": "#3fb950",
    "flagged": "#e5484d",
}

DATASET_STATUS_LABELS = {
    "pending": "pending",
    "in_progress": "in progress",
    "completed": "completed",
    "flagged": "flagged",
}

CURRENT_COLOR = "#58a6ff"


class DatasetItemDelegate(QStyledItemDelegate):
    """Paint queue state with the same hierarchy as the track list."""

    def paint(self, painter, option, index) -> None:
        summary = index.data(DATASET_SUMMARY_ROLE)
        if not summary:
            super().paint(painter, option, index)
            return

        painter.save()
        rect = option.rect
        selected = bool(option.state & QStyle.State_Selected)
        palette = option.palette
        if selected:
            painter.fillRect(rect, palette.highlight())

        status = summary.get("status", "pending")
        status_color = QColor(DATASET_STATUS_COLORS.get(status, "#8a8f98"))
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

        status_label = DATASET_STATUS_LABELS.get(status, status)
        status_font = QFont(option.font)
        status_font.setPointSizeF(max(option.font.pointSizeF() * 0.85, 6.0))
        painter.setFont(status_font)
        painter.setPen(status_color)
        painter.drawText(line1, Qt.AlignRight | Qt.AlignVCenter, status_label)

        bold = QFont(option.font)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(primary)
        status_width = painter.fontMetrics().horizontalAdvance(status_label) + 16
        filename = painter.fontMetrics().elidedText(
            str(summary.get("name", "")),
            Qt.ElideMiddle,
            max(24, line1.width() - status_width),
        )
        painter.drawText(line1, Qt.AlignLeft | Qt.AlignVCenter, filename)

        if summary.get("missing"):
            detail = "Source file missing"
        elif summary.get("active_tracks") is None:
            detail = "Not loaded yet"
        else:
            detail = (
                f"{summary['verified_tracks']} / {summary['active_tracks']} "
                "verified"
            )

        painter.setFont(option.font)
        painter.setPen(secondary)
        current_width = 0
        if summary.get("is_current"):
            current_font = QFont(option.font)
            current_font.setBold(True)
            current_font.setPointSizeF(max(option.font.pointSizeF() * 0.78, 6.0))
            painter.setFont(current_font)
            painter.setPen(QColor(CURRENT_COLOR))
            painter.drawText(line2, Qt.AlignRight | Qt.AlignVCenter, "CURRENT")
            current_width = painter.fontMetrics().horizontalAdvance("CURRENT") + 14

        painter.setFont(option.font)
        painter.setPen(secondary)
        detail = painter.fontMetrics().elidedText(
            detail,
            Qt.ElideRight,
            max(24, line2.width() - current_width),
        )
        painter.drawText(line2, Qt.AlignLeft | Qt.AlignVCenter, detail)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        metrics = option.fontMetrics
        return QSize(option.rect.width(), metrics.height() * 2 + 10)
