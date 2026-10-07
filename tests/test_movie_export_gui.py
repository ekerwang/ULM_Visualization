import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.gui import main_window_playback as playback_module
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui.main_window_playback import MovieExportCancelled
from ulm_track_correction_gui.gui.main_window_ui import SCOPE_TRACK_RANGE
from ulm_track_correction_gui.gui.movie_export import (
    DEFAULT_WATERMARK_FONT_SIZE_PX,
    MovieWatermark,
    Mp4FrameWriter,
    annotate_movie_frame,
    normalized_mp4_path,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session() -> CorrectionSession:
    track = np.asarray([np.nan, 0.0, 0.0, np.nan])
    stack = np.zeros((20, 30, 4), dtype=np.float32)
    for frame_idx in range(stack.shape[2]):
        stack[:, :, frame_idx] = frame_idx + 1
    return CorrectionSession(
        localized_by_frame={
            1: np.asarray([[1.0, 7.0, 8.0, 2.0]]),
            2: np.asarray([[1.0, 8.0, 9.0, 3.0]]),
        },
        tracks=[track],
        image_stack=stack,
    )


@pytest.fixture
def window(app, tmp_path):
    win = MainWindow()
    win.session = make_session()
    win.current_path = tmp_path / "sample_results.mat"
    win.viewer.set_session(win.session)
    win.populate_track_list()
    win.configure_frame_controls()
    win._sync_enabled_state()
    win.resize(900, 620)
    win.show()
    app.processEvents()
    win.track_list.setCurrentRow(0)
    win.scope_buttons[SCOPE_TRACK_RANGE].click()
    app.processEvents()
    yield win
    win.close()


def test_watermark_lines_and_even_h264_frame(app):
    watermark = MovieWatermark("sample.mat", 17, 4, 22)
    assert watermark.lines(9) == (
        "File: sample.mat",
        "Track ID: 17",
        "Track frames: 4–22",
        "Frame Number: 9",
    )
    image = QImage(101, 81, QImage.Format_RGBA8888)
    image.fill(0)
    annotated = annotate_movie_frame(image, watermark, 9)
    assert (annotated.width(), annotated.height()) == (102, 82)
    assert annotated.format() == QImage.Format_RGBA8888


def test_viewer_preview_tracks_frame_track_and_font_size(window, app):
    preview = window.viewer.movie_watermark_preview
    assert not preview.isVisible()
    assert window.watermark_font_size_spin.value() == DEFAULT_WATERMARK_FONT_SIZE_PX

    window.watermark_preview_checkbox.setChecked(True)
    app.processEvents()
    assert preview.isVisible()
    assert preview.watermark == MovieWatermark("sample_results.mat", 0, 2, 3)
    assert preview.frame_number == window.frame_spin.value()
    assert preview.font_size_px == DEFAULT_WATERMARK_FONT_SIZE_PX
    assert preview.geometry() == window.viewer.viewport().rect()

    window.watermark_font_size_spin.setValue(27)
    window.set_frame_1based(2)
    app.processEvents()
    assert preview.isVisible()
    assert preview.font_size_px == 27
    assert preview.frame_number == 2

    window.clear_track_selection()
    app.processEvents()
    assert preview.watermark == MovieWatermark(
        "sample_results.mat",
        None,
        None,
        None,
    )

    window.track_list.setCurrentRow(0)
    window.scope_buttons[SCOPE_TRACK_RANGE].click()
    window.watermark_font_size_spin.setValue(DEFAULT_WATERMARK_FONT_SIZE_PX)
    window.watermark_preview_checkbox.setChecked(False)
    app.processEvents()
    assert not preview.isVisible()


def test_normalized_mp4_path_replaces_other_suffix():
    assert normalized_mp4_path("movie") == Path("movie.mp4")
    assert normalized_mp4_path("movie.MP4") == Path("movie.MP4")
    assert normalized_mp4_path("movie.avi") == Path("movie.mp4")


def test_active_scope_export_uses_fps_watermarks_and_restores_frame(
    window,
    app,
    tmp_path,
    monkeypatch,
):
    writers = []
    watermark_calls = []
    real_annotate = playback_module.annotate_movie_frame

    class FakeWriter:
        def __init__(self, path, size, fps):
            self.path = Path(path)
            self.size = size
            self.fps = fps
            self.frames = []
            writers.append(self)

        def append_frame(self, frame):
            assert (frame.width(), frame.height()) == self.size
            self.frames.append((frame.width(), frame.height()))

        def close(self):
            self.path.write_bytes(b"fake mp4")

    def recording_annotate(source, watermark, frame_number, font_size_px):
        assert not window.viewer.movie_watermark_preview.isVisible()
        watermark_calls.append(
            (frame_number, watermark.lines(frame_number), font_size_px)
        )
        return real_annotate(source, watermark, frame_number, font_size_px)

    monkeypatch.setattr(playback_module, "Mp4FrameWriter", FakeWriter)
    monkeypatch.setattr(playback_module, "annotate_movie_frame", recording_annotate)
    window.fps_spin.setValue(23)
    window.watermark_font_size_spin.setValue(19)
    window.watermark_preview_checkbox.setChecked(True)
    window.set_frame_1based(3)
    progress = []

    output = window._export_movie_to_path(
        tmp_path / "track_preview",
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        )
        or True,
    )

    assert output == tmp_path / "track_preview.mp4"
    assert output.read_bytes() == b"fake mp4"
    assert len(writers) == 1
    assert writers[0].fps == 23
    assert len(writers[0].frames) == 2
    assert all(width % 2 == height % 2 == 0 for width, height in writers[0].frames)
    assert progress == [(1, 2), (2, 2)]
    assert [call[0] for call in watermark_calls] == [2, 3]
    assert watermark_calls[0][1] == (
        "File: sample_results.mat",
        "Track ID: 0",
        "Track frames: 2–3",
        "Frame Number: 2",
    )
    assert watermark_calls[1][1][-1] == "Frame Number: 3"
    assert [call[2] for call in watermark_calls] == [19, 19]
    assert window.frame_spin.value() == 3
    assert window.viewer.frame_idx == 2
    assert window.viewer.movie_watermark_preview.isVisible()
    assert window.viewer.movie_watermark_preview.font_size_px == 19
    assert list(tmp_path.glob(".track_preview-*.mp4")) == []
    window.watermark_preview_checkbox.setChecked(False)
    window.watermark_font_size_spin.setValue(DEFAULT_WATERMARK_FONT_SIZE_PX)


def test_cancel_removes_partial_movie_and_restores_frame(
    window,
    app,
    tmp_path,
    monkeypatch,
):
    class FakeWriter:
        def __init__(self, path, size, fps):
            self.path = Path(path)

        def append_frame(self, frame):
            pass

        def close(self):
            self.path.write_bytes(b"partial")

    monkeypatch.setattr(playback_module, "Mp4FrameWriter", FakeWriter)
    window.set_frame_1based(3)
    output = tmp_path / "canceled.mp4"

    with pytest.raises(MovieExportCancelled):
        window._export_movie_to_path(
            output,
            progress_callback=lambda completed, total: False,
        )

    assert not output.exists()
    assert window.frame_spin.value() == 3
    assert window.viewer.frame_idx == 2
    assert list(tmp_path.glob(".canceled-*.mp4")) == []


def test_export_button_dialog_and_completion_message(
    window,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "chosen.mp4"
    export_calls = []
    messages = []

    monkeypatch.setattr(
        playback_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output), "MP4 video (*.mp4)"),
    )

    def fake_export(path, *, progress_callback=None):
        export_calls.append(Path(path))
        return Path(path)

    monkeypatch.setattr(window, "_export_movie_to_path", fake_export)
    monkeypatch.setattr(
        playback_module.QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    assert window.export_movie_button.text() == "Export Movie"
    assert window.export_movie_button.isEnabled()
    window.export_movie()

    assert export_calls == [output]
    assert messages == [
        ("Movie export complete", f"Movie saved to:\n{output}"),
    ]
    assert window.status_label.text() == "Movie exported to chosen.mp4."
    assert window.export_movie_button.isEnabled()


def test_mp4_writer_creates_h264_container(app, tmp_path):
    output = tmp_path / "writer.mp4"
    writer = Mp4FrameWriter(output, (32, 24), 12)
    for color in (0xFF0000FF, 0x00FF00FF):
        image = QImage(32, 24, QImage.Format_RGBA8888)
        image.fill(color)
        writer.append_frame(image)
    writer.close()

    assert output.stat().st_size > 100
    assert b"ftyp" in output.read_bytes()[:64]
