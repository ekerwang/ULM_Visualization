import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMessageBox

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.dataset_queue import (
    DATASET_STATUS_COMPLETED,
    DATASET_STATUS_FLAGGED,
    DATASET_STATUS_IN_PROGRESS,
)
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui import main_window_datasetlist, main_window_session
from ulm_track_correction_gui.gui.main_window_datasetlist import (
    DATASET_NAME_FILTER_ALL_RESULTS,
    DATASET_NAME_FILTER_DOWN_RESULTS,
    DATASET_NAME_FILTER_UP_RESULTS,
    DatasetImportDialog,
)
from ulm_track_correction_gui.io.mat_io import load_correction_session


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session() -> CorrectionSession:
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [10.0, 2.0, 3.0, 1.0],
                    [11.0, 4.0, 5.0, 1.0],
                ]
            )
        },
        tracks=[np.asarray([0.0]), np.asarray([1.0]), np.asarray([np.nan])],
        image_stack=np.ones((5, 6, 1), dtype=np.float32),
    )


@pytest.fixture
def window(app, tmp_path):
    settings = QSettings(str(tmp_path / "queue.ini"), QSettings.IniFormat)
    settings.clear()
    win = MainWindow(dataset_settings=settings)
    yield win
    win.close()
    settings.clear()


def attach_session(window, source_path: Path) -> None:
    window.session = make_session()
    window.current_path = source_path
    window.session.edit_listener = window._on_session_edit
    window.on_dataset_session_loaded(source_path, recovered_edits=False)
    window._sync_enabled_state()


def test_corrupt_qsettings_is_ignored_without_loading_a_session(app, tmp_path):
    settings = QSettings(str(tmp_path / "corrupt.ini"), QSettings.IniFormat)
    settings.setValue("datasetQueue/json-v1", "not-json")

    win = MainWindow(dataset_settings=settings)
    try:
        assert len(win.dataset_queue) == 0
        assert win.session is None
        assert settings.value("datasetQueue/json-v1") is None
        assert "settings were ignored" in win.status_label.text()
    finally:
        win.close()


def test_import_is_path_only_filters_companions_and_persists(window, tmp_path):
    source = tmp_path / "one_results.mat"
    source.touch()
    checkpoint = tmp_path / "one_accumulation_checkpoint.mat"
    checkpoint.touch()
    corrected_dir = tmp_path / "corrected"
    corrected_dir.mkdir()
    corrected = corrected_dir / source.name
    corrected.touch()
    text_file = tmp_path / "notes.txt"
    text_file.touch()

    report = window.add_dataset_paths(
        [source, checkpoint, corrected, text_file, source],
        report_dialog=False,
    )

    assert report == {
        "added": 1,
        "duplicates": 1,
        "skipped_checkpoint": 1,
        "skipped_corrected": 1,
        "skipped_non_mat": 1,
    }
    assert window.session is None
    assert window.dataset_list.count() == 1
    assert window.main_splitter.count() == 4
    assert window.import_dataset_button.text() == "Add dataset"
    assert window.remove_all_datasets_button.text() == "Remove all"
    assert window.remove_all_datasets_button.isEnabled()

    restored = MainWindow(dataset_settings=window.dataset_settings)
    try:
        assert [entry.source_path for entry in restored.dataset_queue.entries] == [
            str(source.resolve())
        ]
    finally:
        restored.close()


def test_dataset_import_dialog_filters_direction_and_selects_all_visible_files(
    app,
    tmp_path,
):
    for name in (
        "first_up_results.mat",
        "second_up_results.mat",
        "first_down_results.mat",
        "first_up_accumulation_checkpoint.mat",
    ):
        (tmp_path / name).touch()

    dialog_title = "Import datasets for Multi Dataset Accumulation (ULM)"
    dialog = DatasetImportDialog(start_dir=str(tmp_path), title=dialog_title)
    try:
        assert dialog.windowTitle() == dialog_title
        assert dialog.nameFilters()[:3] == [
            DATASET_NAME_FILTER_ALL_RESULTS,
            DATASET_NAME_FILTER_UP_RESULTS,
            DATASET_NAME_FILTER_DOWN_RESULTS,
        ]
        dialog.selectNameFilter(DATASET_NAME_FILTER_UP_RESULTS)
        dialog.show()

        view = dialog.findChild(QAbstractItemView, "treeView")
        assert view is not None
        for _ in range(100):
            app.processEvents()
            if view.model().rowCount(view.rootIndex()) == 2:
                break
            QTest.qWait(10)
        assert view.model().rowCount(view.rootIndex()) == 2

        view.setFocus()
        QTest.keyClick(view, Qt.Key_A, Qt.MetaModifier)
        app.processEvents()
        assert {Path(path).name for path in dialog.selectedFiles()} == {
            "first_up_results.mat",
            "second_up_results.mat",
        }

        dialog.selectNameFilter(DATASET_NAME_FILTER_DOWN_RESULTS)
        for _ in range(100):
            app.processEvents()
            if view.model().rowCount(view.rootIndex()) == 1:
                break
            QTest.qWait(10)
        view.clearSelection()
        view.setFocus()
        QTest.keyClick(view, Qt.Key_A, Qt.ControlModifier)
        app.processEvents()
        assert [Path(path).name for path in dialog.selectedFiles()] == [
            "first_down_results.mat"
        ]
    finally:
        dialog.close()


def test_remove_all_clears_persisted_queue_but_keeps_open_session(
    window,
    tmp_path,
    monkeypatch,
):
    sources = [tmp_path / f"{index}_results.mat" for index in range(2)]
    for source in sources:
        source.touch()
    window.add_dataset_paths(sources)
    attach_session(window, sources[0])
    original_session = window.session

    monkeypatch.setattr(window, "_confirm_remove_all_datasets", lambda count: False)
    assert not window.remove_all_datasets()
    assert len(window.dataset_queue) == 2

    monkeypatch.setattr(window, "_confirm_remove_all_datasets", lambda count: count == 2)
    assert window.remove_all_datasets()
    assert len(window.dataset_queue) == 0
    assert window.dataset_list.count() == 0
    assert window.dataset_header.text() == "Datasets · 0 / 0 completed"
    assert not window.remove_all_datasets_button.isEnabled()
    assert not window.save_progress_button.isEnabled()
    assert window.session is original_session
    assert window.current_path == sources[0]
    assert window.current_dataset_entry() is None
    assert "No files were deleted" in window.status_label.text()

    restored = MainWindow(dataset_settings=window.dataset_settings)
    try:
        assert len(restored.dataset_queue) == 0
    finally:
        restored.close()

    window.add_dataset_paths([sources[0]])
    assert window.current_dataset_entry() is not None
    assert window.save_progress_button.isEnabled()


def test_left_panel_tabs_restore_collapsed_dataset_and_track_panels(window, app):
    window.resize(1540, 820)
    window.show()
    app.processEvents()

    assert window.main_splitter.count() == 4
    assert window.left_panel_tabs.count() == 2
    assert window.left_panel_tabs.tabText(0) == "● Datasets"
    assert window.left_panel_tabs.tabText(1) == "● Tracks"
    assert window.left_panel_tabs.tabData(0) == "expanded"
    assert window.left_panel_tabs.tabData(1) == "expanded"
    assert window.main_splitter.sizes()[0] > 0
    assert window.main_splitter.sizes()[1] > 0

    window.collapse_dataset_button.click()
    app.processEvents()
    assert window.main_splitter.sizes()[0] == 0
    assert not window.left_panel_tabs.isHidden()
    assert window.left_panel_tabs.tabText(0) == "○ Datasets"
    assert window.left_panel_tabs.tabData(0) == "collapsed"
    assert window.left_panel_tabs.tabText(1) == "● Tracks"
    assert window.left_panel_tabs.tabData(1) == "expanded"

    QTest.mouseClick(
        window.left_panel_tabs,
        Qt.LeftButton,
        pos=window.left_panel_tabs.tabRect(0).center(),
    )
    app.processEvents()
    assert window.main_splitter.sizes()[0] >= window.dataset_panel.minimumWidth()
    assert window.left_panel_tabs.tabText(0) == "● Datasets"
    assert window.left_panel_tabs.tabData(0) == "expanded"

    # Direct splitter collapse remains recoverable, not just the header action.
    sizes = window.main_splitter.sizes()
    sizes[0] = 0
    window.main_splitter.setSizes(sizes)
    window.main_splitter.splitterMoved.emit(0, 1)
    app.processEvents()
    assert window.left_panel_tabs.tabText(0) == "○ Datasets"
    assert window.left_panel_tabs.tabData(0) == "collapsed"
    QTest.mouseClick(
        window.left_panel_tabs,
        Qt.LeftButton,
        pos=window.left_panel_tabs.tabRect(0).center(),
    )
    app.processEvents()
    assert window.main_splitter.sizes()[0] >= window.dataset_panel.minimumWidth()

    window.collapse_track_button.click()
    app.processEvents()
    assert window.main_splitter.sizes()[1] == 0
    assert window.left_panel_tabs.tabText(1) == "○ Tracks"
    assert window.left_panel_tabs.tabData(1) == "collapsed"

    QTest.mouseClick(
        window.left_panel_tabs,
        Qt.LeftButton,
        pos=window.left_panel_tabs.tabRect(1).center(),
    )
    app.processEvents()
    assert window.main_splitter.sizes()[1] >= window.track_panel.minimumWidth()
    assert window.left_panel_tabs.tabText(1) == "● Tracks"
    assert window.left_panel_tabs.tabData(1) == "expanded"


def test_single_click_does_not_load_and_double_click_requires_confirmation(
    window,
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "one_results.mat"
    source.touch()
    window.add_dataset_paths([source])
    calls = []
    monkeypatch.setattr(window, "open_path", lambda path: calls.append(path) or True)

    window.dataset_list.setCurrentRow(0)
    assert calls == []

    monkeypatch.setattr(window, "_confirm_dataset_load", lambda entry: False)
    window.on_dataset_double_clicked(window.dataset_list.item(0))
    assert calls == []

    monkeypatch.setattr(window, "_confirm_dataset_load", lambda entry: True)
    window.on_dataset_double_clicked(window.dataset_list.item(0))
    assert calls == [str(source.resolve())]


@pytest.mark.parametrize(
    ("key", "expected"),
    ((Qt.Key_Y, True), (Qt.Key_N, False)),
)
def test_dataset_load_confirmation_handles_y_n_from_focused_button(
    window,
    tmp_path,
    key,
    expected,
):
    source = tmp_path / f"key_{int(key)}_results.mat"
    source.touch()
    window.add_dataset_paths([source])
    entry = window.dataset_queue.entry_for_path(source)
    targeted_dialogs = []

    def press_confirmation_key():
        dialog = QApplication.activeModalWidget()
        targeted_dialogs.append(dialog)
        QTest.keyClick(dialog.yes_button, key)

    QTimer.singleShot(0, press_confirmation_key)
    assert window._confirm_dataset_load(entry) is expected
    assert targeted_dialogs
    assert targeted_dialogs[0] is not None


def test_missing_or_invalid_target_does_not_replace_current_session(
    window,
    tmp_path,
    monkeypatch,
):
    current = tmp_path / "current_results.mat"
    invalid = tmp_path / "invalid_results.mat"
    current.touch()
    invalid.touch()
    window.add_dataset_paths([current, invalid])
    attach_session(window, current)
    original_session = window.session
    warnings = []
    criticals = []
    monkeypatch.setattr(window, "_confirm_dataset_load", lambda entry: True)
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args: criticals.append(args),
    )

    assert not window.request_dataset_load(1)
    assert criticals
    assert warnings == []
    assert window.session is original_session
    assert window.current_path == current
    assert window.current_dataset_entry().source_path == str(current.resolve())


def test_next_loads_literal_next_row_even_when_flagged_or_completed(
    window,
    tmp_path,
    monkeypatch,
):
    sources = [tmp_path / f"{index}_results.mat" for index in range(3)]
    for source in sources:
        source.touch()
    window.add_dataset_paths(sources)
    window.dataset_queue.entries[1].status = DATASET_STATUS_COMPLETED
    window.dataset_queue.entries[2].status = DATASET_STATUS_FLAGGED
    attach_session(window, sources[0])

    loaded = []
    monkeypatch.setattr(window, "_confirm_dataset_load", lambda entry: True)
    monkeypatch.setattr(window, "open_path", lambda path: loaded.append(path) or True)
    window.load_next_dataset()

    assert loaded == [str(sources[1].resolve())]
    window._current_dataset_path = str(sources[2].resolve())
    window._sync_dataset_controls()
    assert not window.next_dataset_button.isEnabled()


def test_dirty_switch_supports_cancel_load_without_save_and_save_then_load(
    window,
    tmp_path,
    monkeypatch,
):
    first = tmp_path / "first_results.mat"
    second = tmp_path / "second_results.mat"
    first.touch()
    second.touch()
    window.add_dataset_paths([first, second])
    attach_session(window, first)
    window._session_dirty = True
    opened = []
    monkeypatch.setattr(window, "open_path", lambda path: opened.append(path) or True)

    monkeypatch.setattr(window, "_choose_dirty_dataset_switch", lambda entry: "cancel")
    assert not window.request_dataset_load(1)
    assert opened == []

    monkeypatch.setattr(window, "_choose_dirty_dataset_switch", lambda entry: "load")
    assert window.request_dataset_load(1)
    assert opened == [str(second.resolve())]

    opened.clear()
    saved = []
    window._session_dirty = True
    monkeypatch.setattr(window, "_choose_dirty_dataset_switch", lambda entry: "save")
    monkeypatch.setattr(
        window,
        "save_dataset_progress",
        lambda: saved.append(True) or True,
    )
    assert window.request_dataset_load(1)
    assert saved == [True]
    assert opened == [str(second.resolve())]


@pytest.mark.parametrize(
    "status",
    [DATASET_STATUS_COMPLETED, DATASET_STATUS_FLAGGED, DATASET_STATUS_IN_PROGRESS],
)
def test_save_progress_uses_corrected_same_name_and_updates_manual_state(
    window,
    tmp_path,
    monkeypatch,
    status,
):
    source = tmp_path / f"{status}_results.mat"
    source.write_bytes(b"original source remains untouched")
    window.add_dataset_paths([source])
    attach_session(window, source)
    window.session.set_track_status(0, "verified")
    if status == DATASET_STATUS_COMPLETED:
        window.session.set_track_status(1, "verified")
    monkeypatch.setattr(
        window,
        "_choose_dataset_status_after_save",
        lambda path: status,
    )

    assert window.save_dataset_progress()

    entry = window.dataset_queue.entry_for_path(source)
    assert entry.status == status
    expected_verified = 2 if status == DATASET_STATUS_COMPLETED else 1
    assert (entry.verified_tracks, entry.active_tracks) == (expected_verified, 2)
    assert entry.last_saved_at is not None
    assert source.read_bytes() == b"original source remains untouched"
    assert entry.corrected_path == tmp_path / "corrected" / source.name
    assert entry.corrected_path.exists()
    assert load_correction_session(entry.corrected_path).track_status[0] == "verified"
    saved = load_correction_session(entry.corrected_path)
    assert saved.metadata["dataset_review_state"] == status
    assert saved.metadata["verification_complete"] is (
        status == DATASET_STATUS_COMPLETED
    )
    assert saved.active_nonempty_track_ids == (
        [0, 1] if status == DATASET_STATUS_COMPLETED else [0]
    )
    assert saved.metadata["exported_verified_track_ids"] == (
        [0, 1] if status == DATASET_STATUS_COMPLETED else [0]
    )
    assert not window._session_dirty


@pytest.mark.parametrize(
    ("confirm_completion", "expected_status"),
    [
        (True, DATASET_STATUS_COMPLETED),
        (False, DATASET_STATUS_IN_PROGRESS),
    ],
)
def test_incomplete_completion_warns_but_respects_user_choice(
    window,
    tmp_path,
    monkeypatch,
    confirm_completion,
    expected_status,
):
    source = tmp_path / "not_ready_results.mat"
    source.write_bytes(b"source")
    window.add_dataset_paths([source])
    attach_session(window, source)
    window.session.set_track_status(0, "verified")
    monkeypatch.setattr(
        window,
        "_choose_dataset_status_after_save",
        lambda path: DATASET_STATUS_COMPLETED,
    )
    prompts = []
    monkeypatch.setattr(
        main_window_datasetlist,
        "ask_yes_no",
        lambda parent, title, text: prompts.append((title, text))
        or confirm_completion,
    )

    assert window.save_dataset_progress()

    entry = window.dataset_queue.entry_for_path(source)
    assert entry.status == expected_status
    saved = load_correction_session(entry.corrected_path)
    assert saved.metadata["dataset_review_state"] == expected_status
    assert saved.metadata["verification_complete"] is False
    assert saved.metadata["completion_override"] is confirm_completion
    assert saved.active_nonempty_track_ids == [0]
    assert saved.metadata["exported_verified_track_ids"] == [0]
    assert prompts and prompts[0][0] == "Complete with unresolved tracks?"
    assert "Only the 1 verified track(s) will be included" in prompts[0][1]


def test_save_as_refuses_to_overwrite_source(window, tmp_path, monkeypatch):
    source = tmp_path / "source_results.mat"
    source.write_bytes(b"precious source")
    attach_session(window, source)
    monkeypatch.setattr(
        main_window_session.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(source), "MAT files (*.mat)"),
    )
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args: messages.append(args[2]),
    )
    monkeypatch.setattr(
        main_window_session,
        "save_correction_session_atomic",
        lambda *args: pytest.fail("source overwrite reached the writer"),
    )

    window.save_mat_as()

    assert source.read_bytes() == b"precious source"
    assert messages and "cannot overwrite" in messages[0]


def test_mid_session_autosave_failure_keeps_edit_undoable_and_blocks_more_edits(
    window,
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source_results.mat"
    source.write_bytes(b"source")
    window.add_dataset_paths([source])
    attach_session(window, source)
    window.undo_manager = UndoManager(window.session)

    class BrokenWriter:
        def append(self, edit):
            raise RuntimeError("simulated serialization failure")

        def close(self):
            return None

    window.editlog_writer = BrokenWriter()
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args: messages.append(args[2]),
    )

    window.session.set_track_status(0, "verified")

    assert window.session.track_status[0] == "verified"
    assert window.undo_manager.can_undo
    assert window._session_dirty
    assert window._autosave_blocked
    assert not window.mark_verified_button.isEnabled()
    assert not window.save_progress_button.isEnabled()
    assert messages and "remains undoable" in messages[0]


def test_editing_completed_dataset_returns_it_to_in_progress(window, tmp_path):
    source = tmp_path / "complete_results.mat"
    source.touch()
    window.add_dataset_paths([source])
    attach_session(window, source)
    entry = window.dataset_queue.entry_for_path(source)
    entry.status = DATASET_STATUS_COMPLETED
    window._session_dirty = False

    window.session.set_track_status(0, "verified")

    assert entry.status == DATASET_STATUS_IN_PROGRESS
    assert (entry.verified_tracks, entry.active_tracks) == (1, 2)
    assert window._session_dirty
