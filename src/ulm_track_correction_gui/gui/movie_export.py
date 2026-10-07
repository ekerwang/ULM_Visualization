"""MP4 encoding and watermark rendering for viewer movie exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


DEFAULT_WATERMARK_FONT_SIZE_PX = 12
MIN_WATERMARK_FONT_SIZE_PX = 8
MAX_WATERMARK_FONT_SIZE_PX = 48


@dataclass(frozen=True)
class MovieWatermark:
    """Dataset and selected-track labels repeated on every exported frame."""

    filename: str
    track_id: int | None
    track_start_frame: int | None
    track_end_frame: int | None

    def lines(self, frame_number: int) -> tuple[str, str, str, str]:
        track_text = "N/A" if self.track_id is None else str(self.track_id)
        if self.track_start_frame is None or self.track_end_frame is None:
            span_text = "N/A"
        else:
            span_text = f"{self.track_start_frame}–{self.track_end_frame}"
        return (
            f"File: {self.filename}",
            f"Track ID: {track_text}",
            f"Track frames: {span_text}",
            f"Frame Number: {int(frame_number)}",
        )


def normalized_mp4_path(path: str | Path) -> Path:
    """Return *path* with the required MP4 extension."""

    output_path = Path(path).expanduser()
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    return output_path


def draw_movie_watermark(
    painter: QPainter,
    width: int,
    height: int,
    watermark: MovieWatermark,
    frame_number: int,
    font_size_px: int,
) -> None:
    """Paint the shared preview/export watermark in viewport coordinates."""

    width = int(width)
    height = int(height)
    font_size_px = max(
        MIN_WATERMARK_FONT_SIZE_PX,
        min(MAX_WATERMARK_FONT_SIZE_PX, int(font_size_px)),
    )
    if width <= 0 or height <= 0:
        return

    painter.save()
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    font = QFont()
    font.setPixelSize(font_size_px)
    painter.setFont(font)
    metrics = QFontMetrics(font)
    margin = max(7, font.pixelSize() // 2)
    maximum_text_width = max(1, width - 2 * margin)
    visible_lines = tuple(
        metrics.elidedText(line, Qt.ElideMiddle, maximum_text_width)
        for line in watermark.lines(frame_number)
    )
    baseline = margin + metrics.ascent()
    line_spacing = metrics.lineSpacing()
    background_padding = 4
    background_width = max(metrics.horizontalAdvance(line) for line in visible_lines)
    painter.fillRect(
        margin - background_padding,
        margin - background_padding,
        background_width + 2 * background_padding,
        len(visible_lines) * line_spacing + 2 * background_padding,
        QColor(0, 0, 0, 145),
    )
    shadow_pen = QPen(QColor(0, 0, 0, 220))
    text_pen = QPen(Qt.white)
    for visible_line in visible_lines:
        painter.setPen(shadow_pen)
        painter.drawText(margin + 1, baseline + 1, visible_line)
        painter.setPen(text_pen)
        painter.drawText(margin, baseline, visible_line)
        baseline += line_spacing
    painter.restore()


def annotate_movie_frame(
    source: QImage,
    watermark: MovieWatermark,
    frame_number: int,
    font_size_px: int = DEFAULT_WATERMARK_FONT_SIZE_PX,
) -> QImage:
    """Copy a viewport image, pad it for H.264, and paint the white watermark."""

    if source.isNull():
        raise ValueError("The viewer did not produce an image for movie export.")

    converted = source.convertToFormat(QImage.Format_RGBA8888)
    converted.setDevicePixelRatio(1.0)
    width = converted.width() + converted.width() % 2
    height = converted.height() + converted.height() % 2
    frame = QImage(width, height, QImage.Format_RGBA8888)
    frame.fill(Qt.black)

    painter = QPainter(frame)
    painter.drawImage(0, 0, converted)
    draw_movie_watermark(
        painter,
        width,
        height,
        watermark,
        frame_number,
        font_size_px,
    )
    painter.end()
    return frame


class MovieWatermarkPreview(QWidget):
    """Mouse-transparent viewport overlay using the export watermark renderer."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("movieWatermarkPreview")
        self.setAccessibleName("Movie watermark preview")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.watermark: MovieWatermark | None = None
        self.frame_number = 1
        self.font_size_px = DEFAULT_WATERMARK_FONT_SIZE_PX
        self.hide()

    def set_content(
        self,
        watermark: MovieWatermark | None,
        frame_number: int,
        font_size_px: int,
        *,
        visible: bool,
    ) -> None:
        self.watermark = watermark
        self.frame_number = int(frame_number)
        self.font_size_px = int(font_size_px)
        self.setVisible(bool(visible and watermark is not None))
        if self.isVisible():
            self.raise_()
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.watermark is None:
            return
        painter = QPainter(self)
        draw_movie_watermark(
            painter,
            self.width(),
            self.height(),
            self.watermark,
            self.frame_number,
            self.font_size_px,
        )
        painter.end()


class Mp4FrameWriter:
    """Small lazy wrapper around imageio-ffmpeg's streaming writer."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: int) -> None:
        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.fps = int(fps)
        self._writer = None
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("Movie frame dimensions must be positive.")
        if self.size[0] % 2 or self.size[1] % 2:
            raise ValueError("Movie frame dimensions must be even for H.264 encoding.")
        if self.fps <= 0:
            raise ValueError("Movie frame rate must be positive.")

    def _open(self) -> None:
        if self._writer is not None:
            return
        try:
            import imageio_ffmpeg
        except ImportError as exc:  # pragma: no cover - packaging/runtime guard
            raise RuntimeError(
                "MP4 export requires imageio-ffmpeg. Reinstall the application "
                "dependencies and try again."
            ) from exc
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path),
            self.size,
            pix_fmt_in="rgba",
            pix_fmt_out="yuv420p",
            fps=self.fps,
            quality=8,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        self._writer.send(None)

    def append_frame(self, frame: QImage) -> None:
        if (frame.width(), frame.height()) != self.size:
            raise ValueError(
                "The viewer size changed during movie export; keep the window "
                "at a fixed size and try again."
            )
        rgba = frame.convertToFormat(QImage.Format_RGBA8888)
        rgba.setDevicePixelRatio(1.0)
        self._open()
        self._writer.send(bytes(rgba.constBits()))

    def close(self) -> None:
        if self._writer is None:
            return
        writer = self._writer
        self._writer = None
        writer.close()
