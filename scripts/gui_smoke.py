"""Offscreen end-to-end smoke test of the full GUI wiring.

Run with ``python scripts/gui_smoke.py``. The script exercises GUI features
against the synthetic demo dataset and saves a screenshot for visual review;
extend it whenever a feature is added — it is the project's regression
harness for everything pytest cannot reach.

Env overrides: DEMO_MAT (input dataset), SMOKE_OUT_DIR (artifacts dir).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = Path(os.environ.get("SMOKE_OUT_DIR", tempfile.mkdtemp(prefix="gui_smoke_")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
source_demo = Path(os.environ.get("DEMO_MAT", str(ROOT / "demo" / "demo_results.mat")))
if not source_demo.exists():
    source_demo = OUT_DIR / "generated_source_results.mat"
    print(f"demo data missing; generating isolated copy {source_demo} ...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_demo_data.py"),
            "--out",
            str(source_demo),
        ],
        check=True,
    )

# Work only on an isolated demo copy: GUI smoke writes an editlog and a
# companion checkpoint, neither of which should touch source data.
demo_path = OUT_DIR / "checkpoint_demo_results.mat"
shutil.copy2(source_demo, demo_path)
DEMO = str(demo_path)


def _write_smoke_checkpoint(result_path: Path) -> Path:
    import numpy as np
    from scipy.io import savemat

    from ulm_track_correction_gui.io.accumulation_checkpoint_io import (
        checkpoint_path_for_result,
    )
    from ulm_track_correction_gui.io.mat_io import load_correction_session

    session = load_correction_session(result_path)
    scale_z, scale_x = 2.0, 3.0
    map_shape = (
        int(np.ceil((session.image_stack.shape[0] - 1) * scale_z)) + 1,
        int(np.ceil((session.image_stack.shape[1] - 1) * scale_x)) + 1,
    )
    track_ids = []
    hard_rows = []
    hard_cols = []
    hard_offsets = [0]
    aa_rows = []
    aa_cols = []
    aa_weights = []
    aa_offsets = [0]
    for track_id in range(session.n_tracks):
        refs = session.point_refs_for_track(track_id)
        if not refs:
            continue
        track_ids.append(track_id)
        hard_pixels = set()
        aa_weights_by_pixel = {}
        for ref in refs:
            localization_row = session.localization_row_for_ref(ref)
            z, x = localization_row[1:3]
            row = int(round((z - 1.0) * scale_z))
            col = int(round((x - 1.0) * scale_x))
            hard_pixels.add((row, col))
            for drow, dcol, weight in ((0, 0, 1.0), (0, 1, 0.5), (1, 0, 0.5)):
                aa_pixel = (
                    min(map_shape[0] - 1, row + drow),
                    min(map_shape[1] - 1, col + dcol),
                )
                aa_weights_by_pixel[aa_pixel] = (
                    aa_weights_by_pixel.get(aa_pixel, 0.0) + weight
                )
        for row, col in sorted(hard_pixels):
            hard_rows.append(row)
            hard_cols.append(col)
        hard_offsets.append(len(hard_rows))
        for (row, col), weight in sorted(aa_weights_by_pixel.items()):
            aa_rows.append(row)
            aa_cols.append(col)
            aa_weights.append(weight)
        aa_offsets.append(len(aa_rows))

    hard_count = len(hard_rows)
    aa_weight_array = np.asarray(aa_weights, dtype=np.float32)
    zeros = np.zeros(map_shape, dtype=np.float32)
    checkpoint_path = checkpoint_path_for_result(result_path)
    savemat(
        checkpoint_path,
        {
            "checkpointSchemaVersion": np.int32(2),
            "checkpointKind": "pala_accumulation",
            "perTrackContributionMode": "full",
            "inputShape": np.asarray(session.image_stack.shape[:2], dtype=np.uint64),
            "mapShape": np.asarray(map_shape, dtype=np.uint64),
            "inputPixelSizeZUm": np.float64(20.0),
            "inputPixelSizeXUm": np.float64(30.0),
            "ulmPixelSizeZUm": np.float64(10.0),
            "ulmPixelSizeXUm": np.float64(10.0),
            "ulmScaleZ": np.float64(scale_z),
            "ulmScaleX": np.float64(scale_x),
            "rasterRowColumnIndexBase": np.int8(0),
            "sparseOffsetsIndexBase": np.int8(0),
            "sparsePixelIndexConvention": "row_col",
            "axisOrder": "z_x",
            "counterPower": np.float64(1.0 / 3.0),
            "counterDisplayMax": np.float64(8.0),
            "counterAADisplayMax": np.float64(4.0),
            "velocityDisplayMax": np.float64(15.0),
            "accumulation_param": {
                "method": "pala",
                "movingAverageSpan": 1,
                "maxSamplingStepUlmpx": 0.8,
                "inputPixelSizeZUm": 20.0,
                "inputPixelSizeXUm": 30.0,
                "ulmPixelSizeZUm": 10.0,
                "ulmPixelSizeXUm": 10.0,
                "prfHz": 1000.0,
                "antiAliasSize": 0.5,
                "counterPower": 1.0 / 3.0,
                "counterDisplayMax": 8.0,
                "counterAADisplayMax": 4.0,
                "velocityDisplayMax": 15.0,
            },
            "mapCounter": zeros,
            "mapCounter_AA": zeros.copy(),
            "VelocityDisplacement": zeros.copy(),
            "VelocityTrackMean": zeros.copy(),
            "trackIds": np.asarray(track_ids, dtype=np.int64),
            "hardOffsets": np.asarray(hard_offsets, dtype=np.uint64),
            "hardRows": np.asarray(hard_rows, dtype=np.uint32),
            "hardCols": np.asarray(hard_cols, dtype=np.uint32),
            "aaOffsets": np.asarray(aa_offsets, dtype=np.uint64),
            "aaRows": np.asarray(aa_rows, dtype=np.uint32),
            "aaCols": np.asarray(aa_cols, dtype=np.uint32),
            "aaDensityWeight": aa_weight_array,
            "hardLocalVelocityNumerator": np.full(
                hard_count, 5.0, dtype=np.float32
            ),
            "hardTrackMeanVelocityNumerator": np.full(
                hard_count, 7.0, dtype=np.float32
            ),
        },
    )
    return checkpoint_path


checkpoint_path = _write_smoke_checkpoint(demo_path)
print("checkpoint:", checkpoint_path)

from PySide6.QtCore import QPointF, QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGraphicsPathItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
)

app = QApplication([])

from ulm_track_correction_gui.gui.display_panel import OVERLAY_LAYERS  # noqa: E402
from ulm_track_correction_gui.gui.confirmation_dialog import (  # noqa: E402
    YesNoMessageBox,
)
from ulm_track_correction_gui.gui.display_compression import (  # noqa: E402
    DISPLAY_MODE_DB,
    DISPLAY_MODE_POWER,
)
from ulm_track_correction_gui.gui.main_window import MainWindow  # noqa: E402
from ulm_track_correction_gui.gui.main_window_datasetlist import (  # noqa: E402
    DATASET_NAME_FILTER_UP_RESULTS,
    DatasetImportDialog,
)
from ulm_track_correction_gui.gui.main_window_ui import (  # noqa: E402
    SCOPE_FULL_STACK,
    SCOPE_TRACK_PADDED,
    SCOPE_TRACK_RANGE,
    SCOPE_SHORTCUTS,
)
from ulm_track_correction_gui.gui.layer_order import DEFAULT_LAYER_ORDER  # noqa: E402
from ulm_track_correction_gui.gui.viewer import (  # noqa: E402
    ALL_LINES_Z,
    CANDIDATE_Z,
    DATASET_PLOT_Z,
    IMAGE_Z,
    LOCALIZATION_Z,
    MULTIPLE_DATASET_PLOT_Z,
    SELECTED_POINT_Z,
    TRACK_PATH_Z,
    TRACK_PLOT_Z,
)

queue_settings = QSettings(
    str(OUT_DIR / "dataset_queue.ini"),
    QSettings.IniFormat,
)
queue_settings.clear()
win = MainWindow(dataset_settings=queue_settings)
win.resize(1540, 900)
win.show()

# --- all Yes/No confirmation keyboard contracts ----------------------------
for key, expected in (
    (Qt.Key_Y, QMessageBox.Yes),
    (Qt.Key_N, QMessageBox.No),
    (Qt.Key_Return, QMessageBox.Yes),
    (Qt.Key_Enter, QMessageBox.Yes),
    (Qt.Key_Escape, QMessageBox.No),
):
    confirmation = YesNoMessageBox(win, "Confirm", "Proceed?")
    confirmation.show()
    app.processEvents()
    assert confirmation.defaultButton() is confirmation.yes_button
    assert confirmation.escapeButton() is confirmation.no_button
    assert QApplication.focusWidget() is confirmation.yes_button
    if key == Qt.Key_Y:
        confirmation_shot = OUT_DIR / "yes_no_confirmation.png"
        assert confirmation.grab().save(str(confirmation_shot))
    QTest.keyClick(QApplication.focusWidget(), key)
    app.processEvents()
    assert confirmation.result() == expected
    assert not confirmation.isVisible()
print("Yes/No confirmations: Y/N + Enter/Esc + Yes default focus OK")

# --- dataset queue + load demo data -----------------------------------------
# The shared app-owned chooser keeps Add dataset and Multiple Dataset
# Accumulation filtering/select-all behavior consistent across platforms.
dialog_dir = OUT_DIR / "dataset_dialog"
dialog_dir.mkdir(exist_ok=True)
for name in (
    "first_up_results.mat",
    "second_up_results.mat",
    "first_down_results.mat",
):
    (dialog_dir / name).touch()
dataset_dialog_title = "Import datasets for Multi Dataset Accumulation (ULM)"
dataset_dialog = DatasetImportDialog(
    start_dir=str(dialog_dir),
    title=dataset_dialog_title,
)
assert dataset_dialog.windowTitle() == dataset_dialog_title
dataset_dialog.selectNameFilter(DATASET_NAME_FILTER_UP_RESULTS)
dataset_dialog.show()
dialog_view = dataset_dialog.findChild(QAbstractItemView, "treeView")
assert dialog_view is not None
for _ in range(100):
    app.processEvents()
    if dialog_view.model().rowCount(dialog_view.rootIndex()) == 2:
        break
    QTest.qWait(10)
assert dialog_view.model().rowCount(dialog_view.rootIndex()) == 2
dialog_view.setFocus()
QTest.keyClick(dialog_view, Qt.Key_A, Qt.MetaModifier)
app.processEvents()
assert {Path(path).name for path in dataset_dialog.selectedFiles()} == {
    "first_up_results.mat",
    "second_up_results.mat",
}
dataset_dialog_shot = OUT_DIR / "dataset_import_dialog.png"
assert dataset_dialog.grab().save(str(dataset_dialog_shot))
dataset_dialog.close()
print(
    "shared dataset import filters + Command/Ctrl+A: OK,",
    dataset_dialog_shot,
)

# move any editlog left by a previous smoke run aside, or the recovery
# QMessageBox blocks forever under the offscreen platform
_stale_log = Path(str(DEMO) + ".editlog.jsonl")
if _stale_log.exists():
    shutil.move(str(_stale_log), str(OUT_DIR / "demo.editlog.prev.jsonl"))

queue_report = win.add_dataset_paths([DEMO, checkpoint_path])
assert queue_report["added"] == 1
assert queue_report["skipped_checkpoint"] == 1
assert win.session is None, "importing the dataset list must not load MAT data"
assert win.dataset_list.count() == 1
assert win.import_dataset_button.text() == "Add dataset"
assert win.remove_all_datasets_button.text() == "Remove all"
assert win.remove_all_datasets_button.isEnabled()
win._confirm_remove_all_datasets = lambda count: count == 1
assert win.remove_all_datasets()
assert win.dataset_list.count() == 0
assert not win.remove_all_datasets_button.isEnabled()
queue_report = win.add_dataset_paths([DEMO, checkpoint_path])
assert queue_report["added"] == 1
assert win.dataset_list.count() == 1


def _press_dataset_confirmation_y():
    confirmation = QApplication.activeModalWidget()
    assert isinstance(confirmation, YesNoMessageBox)
    QTest.keyClick(confirmation.yes_button, Qt.Key_Y)


QTimer.singleShot(0, _press_dataset_confirmation_y)
assert win._confirm_dataset_load(win.dataset_queue.entries[0])
print("dataset-list load confirmation: Y accepts the active modal dialog OK")

assert win.main_splitter.count() == 4
assert win.left_panel_tabs.count() == 2
assert win.left_panel_tabs.tabText(0) == "● Datasets"
assert win.left_panel_tabs.tabText(1) == "● Tracks"
assert win.left_panel_tabs.tabData(0) == "expanded"
assert win.left_panel_tabs.tabData(1) == "expanded"
win.collapse_dataset_button.click()
app.processEvents()
assert win.main_splitter.sizes()[0] == 0
assert win.left_panel_tabs.tabText(0) == "○ Datasets"
assert win.left_panel_tabs.tabData(0) == "collapsed"
QTest.mouseClick(
    win.left_panel_tabs,
    Qt.LeftButton,
    pos=win.left_panel_tabs.tabRect(0).center(),
)
app.processEvents()
assert win.main_splitter.sizes()[0] >= win.dataset_panel.minimumWidth()
assert win.left_panel_tabs.tabText(0) == "● Datasets"
assert win.left_panel_tabs.tabData(0) == "expanded"
win.collapse_track_button.click()
app.processEvents()
assert win.main_splitter.sizes()[1] == 0
assert win.left_panel_tabs.tabText(1) == "○ Tracks"
assert win.left_panel_tabs.tabData(1) == "collapsed"
QTest.mouseClick(
    win.left_panel_tabs,
    Qt.LeftButton,
    pos=win.left_panel_tabs.tabRect(1).center(),
)
app.processEvents()
assert win.main_splitter.sizes()[1] >= win.track_panel.minimumWidth()
assert win.left_panel_tabs.tabText(1) == "● Tracks"
assert win.left_panel_tabs.tabData(1) == "expanded"
assert win.view_tools_box.title().replace("&&", "&") == "View & Basic"
assert win.track_internal_tools_box.title() == "Track-internal Level"
assert win.track_level_tools_box.title() == "Track Level"
tools_layout = win.track_level_tools_box.parentWidget().layout()
assert tools_layout.indexOf(win.track_level_tools_box) < tools_layout.indexOf(
    win.track_internal_tools_box
)
assert win.view_tools_box.layout().indexOf(win.mode_buttons["inspect"]) >= 0
assert win.track_internal_tools_box.layout().indexOf(win.mode_buttons["assign"]) >= 0
assert win.track_level_tools_box.layout().indexOf(win.initialize_track_button) >= 0
internal_layout = win.track_internal_tools_box.layout()
assert internal_layout.getItemPosition(
    internal_layout.indexOf(win.mode_buttons["assign"])
) == (0, 0, 1, 1)
assert internal_layout.getItemPosition(
    internal_layout.indexOf(win.mode_buttons["remove"])
) == (1, 0, 1, 1)
assert internal_layout.getItemPosition(
    internal_layout.indexOf(win.add_point_button)
) == (2, 0, 1, 1)
assert internal_layout.getItemPosition(
    internal_layout.indexOf(win.add_point_panel)
) == (3, 0, 1, 1)
assert win.add_point_panel.content.isVisible()
track_level_layout = win.track_level_tools_box.layout()
assert track_level_layout.getItemPosition(
    track_level_layout.indexOf(win.initialize_track_button)
) == (0, 0, 1, 1)
assert track_level_layout.getItemPosition(
    track_level_layout.indexOf(win.delete_track_button)
) == (0, 1, 1, 1)
assert track_level_layout.getItemPosition(
    track_level_layout.indexOf(win.merge_button)
) == (1, 0, 1, 1)
assert track_level_layout.getItemPosition(
    track_level_layout.indexOf(win.split_button)
) == (1, 1, 1, 1)
assert win.main_splitter.sizes()[3] > 300

win.open_path(DEMO)
assert win.session is not None, "session failed to load"
assert win.current_dataset_entry() is not None
assert win.current_dataset_entry().active_tracks == win.session.n_tracks
print(f"loaded: {win.session.n_tracks} tracks, {win.session.n_frames} frames")

# --- all track-list sort fields support ascending and descending order ------
sort_options = [
    "Track ID ↑",
    "Track ID ↓",
    "Points ↑",
    "Points ↓",
    "Gaps ↑",
    "Gaps ↓",
    "Start frame ↑",
    "Start frame ↓",
]
assert [win.sort_combo.itemText(index) for index in range(win.sort_combo.count())] == (
    sort_options
)
sort_specs = {
    "Track ID ↑": ("track_id", False),
    "Track ID ↓": ("track_id", True),
    "Points ↑": ("n_points", False),
    "Points ↓": ("n_points", True),
    "Gaps ↑": ("n_gaps", False),
    "Gaps ↓": ("n_gaps", True),
    "Start frame ↑": ("start_frame", False),
    "Start frame ↓": ("start_frame", True),
}
summaries_by_id = {
    track_id: win.session.track_summary(track_id)
    for track_id in win.session.active_track_ids
}
for sort_option, (field, descending) in sort_specs.items():
    expected_ids = sorted(
        summaries_by_id,
        key=lambda track_id: (
            summaries_by_id[track_id][field] is None,
            (
                -summaries_by_id[track_id][field]
                if descending and summaries_by_id[track_id][field] is not None
                else summaries_by_id[track_id][field]
                if summaries_by_id[track_id][field] is not None
                else 0
            ),
            track_id,
        ),
    )
    win.sort_combo.setCurrentText(sort_option)
    app.processEvents()
    actual_ids = [
        int(win.track_list.item(row).data(Qt.UserRole))
        for row in range(win.track_list.count())
    ]
    assert actual_ids == expected_ids
win.sort_combo.setCurrentText("Track ID ↑")
win.sort_combo.showPopup()
app.processEvents()
sort_options_shot = OUT_DIR / "track_sort_options.png"
assert win.sort_combo.view().grab().save(str(sort_options_shot))
win.sort_combo.hidePopup()
print(
    "track-list sorting: 4 fields x ascending/descending OK,",
    sort_options_shot,
)

# --- visible keyboard map + physical layer-slot shortcuts -------------------
assert win.mode_buttons["inspect"].text() == "Inspect (Esc)"
assert win.mode_buttons["assign"].text() == "Assign (A)"
assert win.mode_buttons["remove"].text() == "Remove (R)"
assert win.add_point_button.text() == "Add point (M)"
assert win.initialize_track_button.text() == "New Track (N)"
assert win.merge_button.text() == "Merge (J)"
assert win.split_button.text() == "Split (S)"
assert win.delete_track_button.text() == "Delete Track (D)"
assert win.undo_button.text().startswith("Undo (")
assert win.redo_button.text().startswith("Redo (")
assert win.play_button.text() == "Play (Space)"
assert win.fps_spin.value() == 10
assert win.display_panel.interpolate_button.text() == "Interpolate Dataset (I)"
for scope in (SCOPE_FULL_STACK, SCOPE_TRACK_RANGE, SCOPE_TRACK_PADDED):
    native_shortcut = QKeySequence(SCOPE_SHORTCUTS[scope]).toString(
        QKeySequence.NativeText
    )
    assert win.scope_buttons[scope].text().endswith(f"({native_shortcut})")

win.activateWindow()
win.viewer.setFocus()
app.processEvents()
mode_button_heights = {
    mode: button.height() for mode, button in win.mode_buttons.items()
}
mode_button_hint_heights = {
    mode: button.sizeHint().height() for mode, button in win.mode_buttons.items()
}
assert len(set(mode_button_heights.values())) == 1
assert len(set(mode_button_hint_heights.values())) == 1
QTest.keyClick(win.viewer, Qt.Key_I)
app.processEvents()
assert win.display_panel.interpolate_button.isChecked()
assert win.display_panel.interpolate_button.text() == "Interpolated (I)"
QTest.keyClick(win.viewer, Qt.Key_I)
assert not win.display_panel.interpolate_button.isChecked()
win.mode_buttons["assign"].setChecked(True)
QTest.keyClick(win.viewer, Qt.Key_Escape)
assert win.mode_buttons["inspect"].isChecked()
for key, mode in (
    (Qt.Key_A, "assign"),
    (Qt.Key_R, "remove"),
    (Qt.Key_M, "add_point"),
):
    QTest.keyClick(win.viewer, key)
    app.processEvents()
    assert win.mode_buttons[mode].isChecked()
    assert {
        name: button.height() for name, button in win.mode_buttons.items()
    } == mode_button_heights
    assert {
        name: button.sizeHint().height()
        for name, button in win.mode_buttons.items()
    } == mode_button_hint_heights
    QTest.keyClick(win.viewer, key)
    app.processEvents()
    assert win.mode_buttons["inspect"].isChecked()
    assert {
        name: button.height() for name, button in win.mode_buttons.items()
    } == mode_button_heights

inspect_button = win.mode_buttons["inspect"]
assert inspect_button.contentsRect().width() >= inspect_button.fontMetrics().horizontalAdvance(
    inspect_button.text()
)
assert inspect_button.contentsRect().height() >= inspect_button.fontMetrics().height()
print("correction mode buttons: stable checked-state geometry + readable Inspect text OK")

slot_four = win.display_panel.slot(4)
assert win.display_panel.slot_layer_key(4) == "points"
points_group = win.display_panel._layer_groups["points"]
points_visible = points_group.isChecked()
QTest.keyClick(win.viewer, Qt.Key_4)
assert points_group.isChecked() is not points_visible
QTest.keyClick(win.viewer, Qt.Key_4)
assert points_group.isChecked() is points_visible
win.display_panel.move_layer("track_path", 3)
assert win.display_panel.slot(4) is slot_four
assert win.display_panel.slot_layer_key(4) == "track_path"
track_path_group = win.display_panel._layer_groups["track_path"]
track_path_visible = track_path_group.isChecked()
QTest.keyClick(win.viewer, Qt.Key_4)
assert track_path_group.isChecked() is not track_path_visible
QTest.keyClick(win.viewer, Qt.Key_4)
assert track_path_group.isChecked() is track_path_visible
win.display_panel.move_layer("track_path", 1)
assert win.display_panel.layer_order() == DEFAULT_LAYER_ORDER

slot_four_visible = win.display_panel.slot(4).group.isChecked()
win.display_panel.interp_factor_spin.setFocus()
win.display_panel.interp_factor_spin.selectAll()
QTest.keyClick(win.display_panel.interp_factor_spin, Qt.Key_1)
assert win.display_panel.slot(4).group.isChecked() is slot_four_visible
win.display_panel.interp_factor_spin.setValue(5.0)
print("visible shortcut labels + fixed 1-8 physical slot routing: OK")

# --- reviewed visual defaults ----------------------------------------------
assert (win.viewer.db_min, win.viewer.db_max) == (-30.0, 0.0)
assert (
    win.display_panel.db_min_spin.value(),
    win.display_panel.db_max_spin.value(),
) == (-30.0, 0.0)
assert not win.display_panel.db_compression_radio.isChecked()
assert win.display_panel.power_compression_radio.isChecked()
assert win.display_panel.compression_options_stack.currentIndex() == 1
assert win.display_panel.power_gamma_spin.value() == 1.0
assert win.display_panel.power_clim_control.values() == (0.1, 0.7)
assert win.viewer.display_settings.mode == DISPLAY_MODE_POWER

image_group = win.display_panel._layer_groups["image"]
image_group.setExpanded(True)
app.processEvents()
movie_defaults_shot = OUT_DIR / "ultrasound_movie_defaults.png"
assert image_group.grab().save(str(movie_defaults_shot))
print("Ultrasound Movie defaults screenshot:", movie_defaults_shot)
image_group.setExpanded(False)

multiple_dataset_group = win.display_panel._layer_groups["multiple_dataset_plot"]
multiple_dataset_group.setExpanded(True)
app.processEvents()
assert win.display_panel.multiple_dataset_plot_gamma_slider.value() == 100
assert win.display_panel.multiple_dataset_plot_gamma_spin.value() == 1.0
assert win.display_panel.multiple_dataset_plot_density_clim_min_spin.value() == 0.0
assert win.display_panel.multiple_dataset_plot_density_clim_max_spin.value() == 1.0
multiple_dataset_defaults_shot = OUT_DIR / "multi_dataset_defaults.png"
assert multiple_dataset_group.grab().save(str(multiple_dataset_defaults_shot))
print("Multi Dataset defaults screenshot:", multiple_dataset_defaults_shot)
multiple_dataset_group.setExpanded(False)

power_pixmap_key = win.viewer.image_item.pixmap().cacheKey()
win.display_panel.db_compression_radio.click()
app.processEvents()
assert win.display_panel.db_compression_radio.isChecked()
assert not win.display_panel.power_compression_radio.isChecked()
assert win.display_panel.compression_options_stack.currentIndex() == 0
assert win.viewer.display_settings.mode == DISPLAY_MODE_DB
assert win.viewer.image_item.pixmap().cacheKey() != power_pixmap_key

win.display_panel.power_compression_radio.click()
win.display_panel.power_gamma_spin.setValue(0.75)
win.display_panel.power_clim_control.minimum_spin.setValue(0.142)
app.processEvents()
assert not win.display_panel.db_compression_radio.isChecked()
assert win.display_panel.power_compression_radio.isChecked()
assert win.display_panel.compression_options_stack.currentIndex() == 1
assert win.viewer.display_settings.mode == DISPLAY_MODE_POWER
assert win.viewer.display_settings.gamma == 0.75
assert (
    win.viewer.display_settings.clim_min,
    win.viewer.display_settings.clim_max,
) == (0.142, 0.7)

win.display_panel.power_gamma_spin.setValue(1.0)
win.display_panel.power_clim_control.slider.setValues(0.1, 0.7)
expected_visual_defaults = {
    "image": (0.68, 1.0),
    "dataset_plot": (1.0, 1.0),
    "multiple_dataset_plot": (0.5, 1.0),
    "track_plot": (1.0, 1.0),
    "all_lines": (0.43, 0.5),
    "points": (0.51, 0.5),
    "track_path": (0.58, 0.5),
    "track_point": (0.20, 2.0),
}
for layer_key, (opacity, size) in expected_visual_defaults.items():
    style = win.viewer.styles[layer_key]
    assert (style.opacity, style.size) == (opacity, size)
assert win.viewer.styles["all_lines"].visible is False
assert win.viewer.styles["dataset_plot"].visible is False
assert win.viewer.styles["points_current"].color == "#f050dc"
assert win.viewer.styles["points_other"].color == "#3cf078"
assert win.viewer.styles["points_verified"].visible is True
assert win.viewer.styles["points_verified"].color == "#969696"
assert win.viewer.styles["points_flagged"].visible is True
assert win.viewer.styles["points_flagged"].color == "#e5484d"
assert win.viewer.styles["points"].color == "#ff9632"
assert win.viewer.styles["track_plot"].color == "#00d8ff"
assert win.viewer.styles["dataset_plot"].color == "#00d8ff"
assert win.display_panel.dataset_plot_color_button._color_hex == "#00d8ff"
assert win.display_panel.track_plot_color_button._color_hex == "#00d8ff"
localization_swatches = [
    button
    for button in win.display_panel._layer_groups["points"].findChildren(QPushButton)
    if button.toolTip()
]
assert [button.toolTip() for button in localization_swatches] == [
    "Points of the selected track",
    "Points of other tracks",
    "Unassigned detections",
    "Override verified points from other tracks with this color",
    "Override flagged points from other tracks with this color",
]
status_point_checkboxes = win.display_panel._layer_groups["points"].findChildren(
    QCheckBox
)
assert [checkbox.text() for checkbox in status_point_checkboxes] == [
    "Enable verified points",
    "Enable flagged points",
]
assert all(checkbox.isChecked() for checkbox in status_point_checkboxes)
win.display_panel.points_verified_checkbox.click()
assert win.viewer.styles["points_verified"].visible is False
win.display_panel.points_verified_checkbox.click()
win.display_panel.points_flagged_checkbox.click()
assert win.viewer.styles["points_flagged"].visible is False
win.display_panel.points_flagged_checkbox.click()
assert win.viewer.styles["points_verified"].visible is True
assert win.viewer.styles["points_flagged"].visible is True
print("verified/flagged localization-point subset toggles: OK")
fixed_slots = tuple(win.display_panel.slot(number) for number in range(1, 9))
panel_group_titles = [slot.group.title() for slot in fixed_slots]
assert win.display_panel.layer_order() == DEFAULT_LAYER_ORDER
assert OVERLAY_LAYERS == tuple(
    key for key in DEFAULT_LAYER_ORDER if key != "image"
)
assert panel_group_titles == [
    "Selected Track (sub-pixel plotting)",
    "Selected Track",
    "Selected Track Point",
    "Localization Points",
    "All Tracks",
    "Ultrasound Movie",
    "Single Dataset Accumulation",
    "Multi Dataset Accumulation (ULM)",
]
assert [slot.shortcut_badge.text() for slot in fixed_slots] == [
    f"({number})" for number in range(1, 9)
]
assert all(slot.layout().spacing() == 5 for slot in fixed_slots)
assert all(
    not slot.slot_control.isAncestorOf(slot.group) for slot in fixed_slots
)
assert all(slot.group.parentWidget() is slot for slot in fixed_slots)
assert all(slot.slot_control.width() == 30 for slot in fixed_slots)
assert all(slot.visibility_button is slot.shortcut_badge for slot in fixed_slots)
assert all(slot.shortcut_badge.icon().isNull() for slot in fixed_slots)
display_sections = list(win.display_panel._layer_groups.values())
assert all(not section.isExpanded() for section in display_sections)
assert all(
    section.header_button.property("layerHeader") for section in display_sections
)
image_slot = win.display_panel.slot(
    win.display_panel.layer_order().index("image") + 1
)
assert not image_slot.visibility_button.isHidden()
assert win.display_panel.image_group.isChecked()
image_slot.visibility_button.click()
app.processEvents()
assert not win.viewer.styles["image"].visible
assert not win.viewer.image_item.isVisible()
image_slot.visibility_button.click()
app.processEvents()
assert win.viewer.styles["image"].visible
assert win.viewer.image_item.isVisible()
selected_point_group = win.display_panel._layer_groups["track_point"]
assert selected_point_group.isChecked()
selected_point_group.header_button.click()
assert selected_point_group.isExpanded()
selected_point_group.header_button.click()
assert not selected_point_group.isExpanded()
track_lines_group = win.display_panel._layer_groups["all_lines"]
assert not track_lines_group.isChecked()
track_lines_group.header_button.click()
assert not track_lines_group.isChecked()
assert track_lines_group.isExpanded()
track_lines_slot = win.display_panel.slot(
    win.display_panel.layer_order().index("all_lines") + 1
)
track_lines_slot.visibility_button.click()
assert track_lines_group.isChecked()
assert track_lines_group.isExpanded()
track_lines_slot.visibility_button.click()
assert not track_lines_group.isChecked()
assert track_lines_group.isExpanded()
for layer_key in ("track_path", "points"):
    compact_group = win.display_panel._layer_groups[layer_key]
    size_spin = compact_group.findChild(
        QDoubleSpinBox,
        f"{layer_key}SizeSpin",
    )
    compact_grid = compact_group.content_widget.layout()
    assert compact_grid.getItemPosition(compact_grid.indexOf(size_spin)) == (
        1,
        3,
        1,
        1,
    )
multiple_group = win.display_panel.multiple_dataset_plot_group
assert multiple_group.isAncestorOf(win.display_panel.import_multiple_dataset_button)
assert win.display_panel.import_multiple_dataset_button.isEnabled()
map_groups = tuple(
    win.display_panel._layer_groups[layer_key]
    for layer_key in ("track_plot", "multiple_dataset_plot", "dataset_plot")
)
for map_group in map_groups:
    map_group.setExpanded(True)
app.processEvents()
for status_label in (
    win.display_panel.track_plot_status_label,
    win.display_panel.multiple_dataset_plot_status_label,
    win.display_panel.dataset_plot_status_label,
):
    assert status_label.height() >= status_label.heightForWidth(status_label.width())
for map_group in map_groups:
    map_group.setExpanded(False)
app.processEvents()
win.display_panel.move_layer("dataset_plot", 0)
app.processEvents()
assert win.display_panel.layer_order()[0] == "dataset_plot"
assert tuple(win.display_panel._slots) == fixed_slots
assert win.display_panel.slot_layer_key(1) == "dataset_plot"
assert win.display_panel.slot(1).shortcut_badge.text() == "(1)"
assert win.viewer.layer_order() == win.display_panel.layer_order()
assert win.viewer.dataset_plot_item.zValue() > win.viewer.track_plot_item.zValue()
win.display_panel.move_layer("dataset_plot", DEFAULT_LAYER_ORDER.index("dataset_plot"))
app.processEvents()
assert win.display_panel.layer_order() == DEFAULT_LAYER_ORDER
assert win.viewer.layer_order() == DEFAULT_LAYER_ORDER
print(
    "reviewed visual defaults + fixed numbered slots + draggable layer cards + "
    "mutually exclusive dB/Power Law display: OK"
)

# --- select a track, step frames -----------------------------------------
win.track_list.setCurrentRow(0)
assert win.selected_track_id is not None
tid = win.selected_track_id
print("selected track:", tid)
win.step_frame(3)
win.step_frame(-1)
win.step_track(1)
win.step_track(-1)
assert win.selected_track_id == tid

# --- selected-track Batch checkpoint overlay -------------------------------
assert win.accumulation_checkpoint is not None
assert win.accumulation_checkpoint.schema_version == 2
assert win.accumulation_checkpoint.mode == "full"
assert win.accumulation_checkpoint.supported_track_maps == (
    "hard_density",
    "aa_density",
    "hard_local_velocity",
    "hard_track_mean_velocity",
)
track_plot_model = win.display_panel.track_plot_map_combo.model()
assert win.display_panel.track_plot_map_combo.count() == 4
for map_name in win.accumulation_checkpoint.supported_track_maps:
    map_index = win.display_panel.track_plot_map_combo.findData(map_name)
    assert track_plot_model.item(map_index).isEnabled()
for map_name in ("aa_local_velocity", "aa_track_mean_velocity"):
    assert win.display_panel.track_plot_map_combo.findData(map_name) == -1
for label in ("AA Local Velocity", "AA Track-Mean Velocity"):
    assert win.display_panel.track_plot_map_combo.findText(label) == -1
win.display_panel.track_plot_map_combo.setCurrentIndex(
    win.display_panel.track_plot_map_combo.findData("aa_density")
)
app.processEvents()
assert not win.display_panel.track_plot_color_label.isHidden()
assert not win.display_panel.track_plot_color_button.isHidden()
assert win.display_panel.track_plot_velocity_clim_widget.isHidden()
assert not win.viewer.track_plot_item.pixmap().isNull()
assert win.viewer.track_plot_item.isVisible()
assert win.display_panel.track_plot_opacity_slider.value() == 100
assert win.viewer.track_plot_item.opacity() == 1.0
cyan_pixmap_key = win.viewer.track_plot_item.pixmap().cacheKey()
win.display_panel.track_plot_color_button.set_color("#ff4080")
win.display_panel.track_plot_color_button.color_changed.emit("#ff4080")
app.processEvents()
assert win.viewer.styles["track_plot"].color == "#ff4080"
assert win.viewer.track_plot_item.pixmap().cacheKey() != cyan_pixmap_key
win.display_panel.track_plot_color_button.set_color("#00d8ff")
win.display_panel.track_plot_color_button.color_changed.emit("#00d8ff")
app.processEvents()
assert win.viewer.styles["track_plot"].color == "#00d8ff"
density_image = win.viewer.track_plot_item.pixmap().toImage()
density_pixels = [
    density_image.pixelColor(x, y)
    for y in range(density_image.height())
    for x in range(density_image.width())
    if density_image.pixelColor(x, y).alpha() > 0
]
assert density_pixels
assert all(
    pixel.red() == 0
    and abs(pixel.green() - 216) <= 2
    and pixel.blue() == 255
    for pixel in density_pixels
)
assert all(pixel.alpha() > 0 for pixel in density_pixels)
assert any(pixel.alpha() < 255 for pixel in density_pixels)
assert win.grab().save(str(OUT_DIR / "checkpoint_density_overlay.png"))
saved_view_transform = win.viewer.transform()
density_rect = win.viewer.track_plot_item.sceneBoundingRect().adjusted(-2.0, -2.0, 2.0, 2.0)
win.viewer.fitInView(density_rect, Qt.KeepAspectRatio)
app.processEvents()
assert win.viewer.grab().save(str(OUT_DIR / "checkpoint_density_overlay_zoom.png"))
win.viewer.setTransform(saved_view_transform)
win.display_panel.track_plot_map_combo.setCurrentIndex(
    win.display_panel.track_plot_map_combo.findData("hard_density")
)
app.processEvents()
hard_density_image = win.viewer.track_plot_item.pixmap().toImage()
hard_density_pixels = [
    hard_density_image.pixelColor(x, y)
    for y in range(hard_density_image.height())
    for x in range(hard_density_image.width())
    if hard_density_image.pixelColor(x, y).alpha() > 0
]
assert hard_density_pixels
assert all(pixel.alpha() == 255 for pixel in hard_density_pixels)
hard_density_rect = win.viewer.track_plot_item.sceneBoundingRect().adjusted(
    -2.0,
    -2.0,
    2.0,
    2.0,
)
win.viewer.fitInView(hard_density_rect, Qt.KeepAspectRatio)
app.processEvents()
assert win.viewer.grab().save(str(OUT_DIR / "checkpoint_hard_density_overlay_zoom.png"))
win.viewer.setTransform(saved_view_transform)
win.display_panel.track_plot_opacity_slider.setValue(45)
app.processEvents()
assert abs(win.viewer.track_plot_item.opacity() - 0.45) < 1e-9
win.display_panel.track_plot_opacity_slider.setValue(100)
app.processEvents()
assert win.viewer.track_plot_item.opacity() == 1.0
win.display_panel.track_plot_map_combo.setCurrentIndex(
    win.display_panel.track_plot_map_combo.findData("hard_local_velocity")
)
app.processEvents()
assert win.viewer.track_plot_map_name == "hard_local_velocity"
assert win.display_panel.track_plot_color_label.isHidden()
assert win.display_panel.track_plot_color_button.isHidden()
assert not win.display_panel.track_plot_velocity_clim_widget.isHidden()
assert win.display_panel.track_plot_clim_min_spin.value() == 0.0
assert win.display_panel.track_plot_clim_max_spin.value() == 15.0
overlay_pixmap_key = win.viewer.track_plot_item.pixmap().cacheKey()
win.display_panel.track_plot_clim_min_spin.setValue(2.0)
win.display_panel.track_plot_clim_max_spin.setValue(6.0)
app.processEvents()
assert win.viewer.track_plot_velocity_clim == (2.0, 6.0)
assert win.viewer.track_plot_item.pixmap().cacheKey() != overlay_pixmap_key
overlay_pixmap_key = win.viewer.track_plot_item.pixmap().cacheKey()
win.step_frame(1)
assert win.viewer.track_plot_item.pixmap().cacheKey() == overlay_pixmap_key
assert win.grab().save(str(OUT_DIR / "checkpoint_overlay.png"))
print(
    "selected-track schema-v2 overlay: conditional color/clim controls, "
    "density/hard velocity/opacity/frame persistence OK"
)

# --- dynamic current-dataset accumulation overlay --------------------------
assert win.display_panel.dataset_plot_group.isEnabled()
assert win.viewer.dataset_accumulator is None
assert not win.viewer.styles["dataset_plot"].visible
win.display_panel.dataset_plot_group.setChecked(True)
app.processEvents()
assert win.viewer.dataset_accumulator is not None
assert win.viewer.dataset_plot_item.isVisible()
assert not win.viewer.dataset_plot_item.pixmap().isNull()
assert win.viewer.dataset_plot_item.zValue() == DATASET_PLOT_Z
assert MULTIPLE_DATASET_PLOT_Z < DATASET_PLOT_Z < IMAGE_Z < TRACK_PLOT_Z
assert win.display_panel.dataset_plot_map_combo.currentData() == "aa_density"
assert not win.display_panel.dataset_plot_density_clim_widget.isHidden()
assert win.display_panel.dataset_plot_velocity_clim_widget.isHidden()
assert win.display_panel.dataset_plot_density_clim_min_spin.value() == 0.0
assert win.display_panel.dataset_plot_density_clim_max_spin.value() == 1.0
assert win.viewer.dataset_plot_density_clims["aa_density"] == (0.0, 1.0)
assert win.viewer.dataset_plot_density_clims["hard_density"] == (0.0, 1.0)
assert win.display_panel.dataset_plot_gamma_slider.value() == 100
assert win.display_panel.dataset_plot_gamma_spin.value() == 1.0
assert win.viewer.dataset_plot_gamma == 1.0
dataset_pixmap_key = win.viewer.dataset_plot_item.pixmap().cacheKey()
win.display_panel.dataset_plot_gamma_slider.setValue(50)
app.processEvents()
assert win.display_panel.dataset_plot_gamma_spin.value() == 0.5
assert win.viewer.dataset_plot_gamma == 0.5
assert win.viewer.dataset_plot_item.pixmap().cacheKey() != dataset_pixmap_key
dataset_pixmap_key = win.viewer.dataset_plot_item.pixmap().cacheKey()
win.display_panel.dataset_plot_density_clim_max_spin.setValue(2.0)
app.processEvents()
assert win.viewer.dataset_plot_density_clims["aa_density"] == (0.0, 2.0)
assert win.viewer.dataset_plot_item.pixmap().cacheKey() != dataset_pixmap_key
dataset_pixmap_key = win.viewer.dataset_plot_item.pixmap().cacheKey()
win.display_panel.dataset_plot_color_button.set_color("#ff8000")
win.display_panel.dataset_plot_color_button.color_changed.emit("#ff8000")
app.processEvents()
assert win.viewer.styles["dataset_plot"].color == "#ff8000"
assert win.viewer.dataset_plot_item.pixmap().cacheKey() != dataset_pixmap_key
win.display_panel.dataset_plot_opacity_slider.setValue(55)
assert abs(win.viewer.dataset_plot_item.opacity() - 0.55) < 1e-9
win.display_panel.dataset_plot_map_combo.setCurrentIndex(
    win.display_panel.dataset_plot_map_combo.findData("hard_density")
)
app.processEvents()
assert win.display_panel.dataset_plot_density_clim_min_spin.value() == 0.0
assert win.display_panel.dataset_plot_density_clim_max_spin.value() == 1.0
win.display_panel.dataset_plot_density_clim_min_spin.setValue(0.2)
win.display_panel.dataset_plot_density_clim_max_spin.setValue(1.5)
app.processEvents()
assert win.viewer.dataset_plot_density_clims["hard_density"] == (0.2, 1.5)
win.display_panel.dataset_plot_map_combo.setCurrentIndex(
    win.display_panel.dataset_plot_map_combo.findData("aa_density")
)
app.processEvents()
assert win.display_panel.dataset_plot_density_clim_min_spin.value() == 0.0
assert win.display_panel.dataset_plot_density_clim_max_spin.value() == 2.0
win.display_panel.dataset_plot_map_combo.setCurrentIndex(
    win.display_panel.dataset_plot_map_combo.findData("hard_local_velocity")
)
app.processEvents()
assert win.viewer.dataset_plot_map_name == "hard_local_velocity"
assert win.display_panel.dataset_plot_density_clim_widget.isHidden()
assert not win.display_panel.dataset_plot_velocity_clim_widget.isHidden()
win.display_panel.dataset_plot_map_combo.setCurrentIndex(
    win.display_panel.dataset_plot_map_combo.findData("aa_density")
)
app.processEvents()

track_frames = list(win.session.point_refs_for_track(tid))
assert track_frames
edited_frame = track_frames[0].frame_idx
revision = win.viewer.dataset_accumulator.revision
win.session.remove_assignment(tid, edited_frame)
win._refresh_after_edit([tid])
assert win.viewer.dataset_accumulator.revision == revision + 1
win.undo()
assert win.viewer.dataset_accumulator.revision == revision + 2
display_scroll = next(
    scroll
    for scroll in win.findChildren(QScrollArea)
    if scroll.widget() is win.display_panel
)
display_scroll.ensureWidgetVisible(win.display_panel.dataset_plot_group, 0, 0)
app.processEvents()
assert win.grab().save(str(OUT_DIR / "dynamic_dataset_accumulation_overlay.png"))
print(
    "dynamic current-dataset accumulation: lazy build/incremental edit/map/"
    "0-1 normalization/Gamma/unbounded Clim/color/opacity OK"
)

# --- playback range logic --------------------------------------------------
summary = win.session.track_summary(win.selected_track_id)
track_start = summary["start_frame"] + 1
track_end = summary["end_frame"] + 1
assert win.scope_sliders[SCOPE_TRACK_RANGE].minimum() == track_start
assert win.scope_sliders[SCOPE_TRACK_RANGE].maximum() == track_end
assert win.scope_sliders[SCOPE_FULL_STACK].isEnabled()
assert not win.scope_sliders[SCOPE_TRACK_RANGE].isEnabled()
assert win.scope_stack.currentWidget() is win.scope_sliders[SCOPE_FULL_STACK]
assert win.padding_controls.isHidden()

slot_two = win.display_panel.slot(2)
slot_two_visible = slot_two.group.isChecked()
QTest.keyClick(win.viewer, Qt.Key_W)
app.processEvents()
assert win.playback_scope == SCOPE_TRACK_RANGE
assert slot_two.group.isChecked() is slot_two_visible
QTest.keyClick(win.viewer, Qt.Key_E)
app.processEvents()
assert win.playback_scope == SCOPE_TRACK_PADDED
QTest.keyClick(win.viewer, Qt.Key_Q)
app.processEvents()
assert win.playback_scope == SCOPE_FULL_STACK

win.scope_buttons[SCOPE_TRACK_RANGE].click()
assert win.frame_slider is win.scope_sliders[SCOPE_TRACK_RANGE]
assert win.scope_stack.currentWidget() is win.frame_slider
assert win._playback_range() == (track_start, track_end)
assert win.scope_sliders[SCOPE_TRACK_RANGE].isEnabled()
assert not win.scope_sliders[SCOPE_FULL_STACK].isEnabled()
assert win.padding_controls.isHidden()

# Export exactly the active scope from the current viewer. This uses the real
# H.264 encoder; the focused pytest coverage separately inspects every
# watermark line and cancellation cleanup.
win.fps_spin.setValue(17)
win.watermark_font_size_spin.setValue(18)
win.watermark_preview_checkbox.setChecked(True)
win.set_frame_1based(track_end)
app.processEvents()
assert win.viewer.movie_watermark_preview.isVisible()
assert win.viewer.movie_watermark_preview.font_size_px == 18
assert win.viewer.movie_watermark_preview.frame_number == track_end
watermark_preview_shot = OUT_DIR / "movie_watermark_preview.png"
assert win.viewer.viewport().grab().save(str(watermark_preview_shot))
movie_frame_before_export = win.frame_spin.value()
movie_path = win._export_movie_to_path(OUT_DIR / "track_range_preview.mp4")
assert movie_path.exists() and movie_path.stat().st_size > 100
assert b"ftyp" in movie_path.read_bytes()[:64]
assert win.frame_spin.value() == movie_frame_before_export
assert win.viewer.frame_idx == movie_frame_before_export - 1
assert win.viewer.movie_watermark_preview.isVisible()
assert win.export_movie_button.text() == "Export Movie"
assert win.export_movie_button.isEnabled()
print(
    "MP4 movie export: shared live preview/18 px font/active scope/current "
    "viewer/17 fps/watermark + frame restoration OK,",
    movie_path,
    watermark_preview_shot,
)
win.watermark_preview_checkbox.setChecked(False)

win.scope_buttons[SCOPE_TRACK_PADDED].click()
start, end = win._playback_range()
assert win.frame_slider is win.scope_sliders[SCOPE_TRACK_PADDED]
assert win.scope_stack.currentWidget() is win.frame_slider
assert not win.padding_controls.isHidden()
assert win.scope_sliders[SCOPE_TRACK_PADDED].segment_frame_counts() == (
    track_start - start,
    track_end - track_start + 1,
    end - track_end,
)
print("playback range (track ±N):", start, end)
win.set_frame_1based(end)
win.play_button.setChecked(True)
win._on_play_tick()
win.play_button.setChecked(False)
assert win.frame_spin.value() == start
win.scope_buttons[SCOPE_FULL_STACK].click()
assert win._playback_range() == (1, win.session.n_frames)
assert win.padding_controls.isHidden()
win.set_frame_1based(track_start)
print("stacked proportional frame timelines + Q/W/E: range/switch/loop OK")

# --- layer style panel -----------------------------------------------------
win.display_panel.style_changed.emit("points", "size", 3.0)
win.display_panel.style_changed.emit("points", "opacity", 0.9)
win.display_panel.style_changed.emit("track_path", "color", "#00ffff")
win.display_panel.toggle_all_lines()
assert win.viewer.styles["all_lines"].visible is True
assert win.viewer.styles["points"].size == 3.0
localization_z_values = {
    item.zValue()
    for item in win.viewer.overlay_items
    if item.data(0) == "localization"
}
assert localization_z_values == {LOCALIZATION_Z}
assert any(item.zValue() == SELECTED_POINT_Z for item in win.viewer.overlay_items)
assert any(item.zValue() == TRACK_PATH_Z for item in win.viewer.overlay_items)
assert any(item.zValue() == ALL_LINES_Z for item in win.viewer.overlay_items)
assert win.viewer.dataset_plot_item.zValue() == DATASET_PLOT_Z
assert win.viewer.multiple_dataset_plot_item.zValue() == MULTIPLE_DATASET_PLOT_Z
assert win.viewer.track_plot_item.zValue() == TRACK_PLOT_Z
assert (
    MULTIPLE_DATASET_PLOT_Z
    < DATASET_PLOT_Z
    < IMAGE_Z
    < ALL_LINES_Z
    < LOCALIZATION_Z
    < SELECTED_POINT_Z
    < TRACK_PATH_Z
    < TRACK_PLOT_Z
)
print("overlay stacking and foreground-to-background Display group order: OK")

# --- display-only interpolation ---------------------------------------------
assert win.viewer.image_item.transformationMode() == Qt.FastTransformation
assert win.display_panel.interp_factor_spin.value() == 5.0
assert win.display_panel.aspect_ratio_spin.value() == 1.0
assert win.display_panel.interpolate_button.property("stateButton") is True
assert win.display_panel.toggle_button.property("stateButton") is True
assert win.display_panel.toggle_button.isCheckable()
assert (
    win.display_panel.interpolate_button.sizeHint().height()
    == win.display_panel.toggle_button.sizeHint().height()
)
view_scale_before = (
    win.viewer.transform().m11(),
    win.viewer.transform().m22(),
)
win.display_panel.interp_factor_spin.setValue(2.0)
win.display_panel.aspect_ratio_spin.setValue(1.5)
interpolate_button_height = win.display_panel.interpolate_button.sizeHint().height()
win.display_panel.interpolate_button.click()
app.processEvents()
assert win.viewer.interpolation_enabled
assert win.display_panel.interpolate_button.sizeHint().height() == interpolate_button_height
assert win.viewer.image_item.transformationMode() == Qt.FastTransformation
nz, nx = win.session.image_stack.shape[:2]
assert win.viewer.image_item.pixmap().width() == int(nx * 2.0)
assert win.viewer.image_item.pixmap().height() == int(nz * 2.0 * 1.5)
assert abs(win.viewer.scene.sceneRect().width() - nx) < 1e-6
assert abs(win.viewer.scene.sceneRect().height() - nz * 1.5) < 1e-6
assert abs(win.viewer.transform().m11() - view_scale_before[0]) < 1e-6
assert abs(win.viewer.transform().m22() - view_scale_before[1]) < 1e-6
interp_pixmap_key = win.viewer.image_item.pixmap().cacheKey()
win.viewer.render()
assert win.viewer.image_item.pixmap().cacheKey() == interp_pixmap_key
win.display_panel.interpolate_button.click()
assert not win.viewer.interpolation_enabled
assert win.viewer.image_item.pixmap().width() == nx
assert win.viewer.image_item.pixmap().height() == nz
win.display_panel.interpolate_button.click()
assert win.viewer.image_item.pixmap().cacheKey() == interp_pixmap_key
win.display_panel.interpolate_button.click()
win.display_panel.interp_factor_spin.setValue(5.0)
win.display_panel.aspect_ratio_spin.setValue(1.0)
print("display interpolation: factor/aspect/toggle/cache/zoom preservation OK")

# --- panel fits without horizontal scrolling --------------------------------
app.processEvents()
hint = win.display_panel.sizeHint().width()
print("panel sizeHint width:", hint)
# The fixed number dock stays clear of the disclosure arrow; the longest layer
# title and compact parameter rows still fit without horizontal scrolling.
assert hint <= 300, f"display panel too wide: {hint}px"

# --- master overlay toggle ---------------------------------------------------
win.display_panel._layer_groups["track_point"].setChecked(False)  # custom mix
visible_before = {k for k in OVERLAY_LAYERS if win.viewer.styles[k].visible}
assert visible_before == {
    "dataset_plot",
    "track_plot",
    "all_lines",
    "points",
    "track_path",
}
win.display_panel.toggle_overlays()  # hide all
assert all(not win.viewer.styles[k].visible for k in OVERLAY_LAYERS)
assert win.viewer.styles["image"].opacity == 0.68  # image untouched
assert win.display_panel.toggle_button.isChecked()
assert win.display_panel.toggle_button.text() == "Restore layers (T)"
win.display_panel.toggle_overlays()  # restore
visible_after = {k for k in OVERLAY_LAYERS if win.viewer.styles[k].visible}
assert visible_after == visible_before, (visible_before, visible_after)
assert not win.display_panel.toggle_button.isChecked()
assert win.display_panel.toggle_button.text() == "Keep images only (T)"
assert win.viewer.track_plot_item.isVisible()
assert win.viewer.dataset_plot_item.isVisible()
assert win.viewer.styles["points"].size == 3.0  # attributes preserved
assert win.viewer.styles["track_path"].color == "#00ffff"
win.display_panel._layer_groups["track_point"].setChecked(True)
print("overlay master toggle: OK (restored", sorted(visible_after), ")")

# --- go to a frame with detections and click-assign ------------------------
summary = win.session.track_summary(tid)
frame_idx = summary["start_frame"]
win.set_frame_1based(frame_idx + 1)

# find an unassigned detection in this frame to exercise assign
points = win.session.localized_by_frame.get(frame_idx)
target = None
for li in range(points.shape[0]):
    if win.session.track_id_for_detection(frame_idx, li) is None:
        target = li
        break

win.mode_buttons["assign"].setChecked(True)
assert win.viewer.interaction_mode == "assign"
if target is not None:
    # pick through the same path a real click uses
    z, x = points[target, 1], points[target, 2]
    picked = win.viewer._find_detection_near(x - 0.5, z - 0.5)
    assert picked == target, f"picker found {picked}, expected {target}"
    win.on_detection_clicked(frame_idx, target)
    assert win.session.track_id_for_detection(frame_idx, target) == tid
    assert win.correction_mode == "assign"
    print("assign via click path: OK (detection", target, ")")
win.mode_buttons["assign"].click()
assert win.correction_mode == "inspect"
print("active Assign button toggles back to Inspect: OK")

# --- remove mode ------------------------------------------------------------
win.mode_buttons["remove"].setChecked(True)
own_local = win.session.local_idx_for_track_frame(tid, frame_idx)
win.on_detection_clicked(frame_idx, own_local)
assert win.session.local_idx_for_track_frame(tid, frame_idx) is None
assert win.correction_mode == "remove"
print("remove via click path: OK")
win.mode_buttons["remove"].click()
assert win.correction_mode == "inspect"
print("active Remove button toggles back to Inspect: OK")

# --- inline Add Point mode (candidate localization in the main viewer) -------
from ulm_track_correction_gui.reference_processing.wrappers import (  # noqa: E402
    FS_GRADIENT_NCC_METHOD,
    RADIAL_NCC_METHOD,
    RADIAL_NON_NCC_METHOD,
)
from ulm_track_correction_gui.gui.add_point_panel import (  # noqa: E402
    MANUAL_PLACEMENT_METHOD,
)

panel = win.add_point_panel
assert [panel.method_combo.itemText(index) for index in range(4)] == [
    "Radial Symmetry (non-NCC)",
    "FS_Gradient (NCC)",
    "Radial Symmetry (NCC)",
    "Manual Placement",
]
assert panel.header_button.isChecked()
assert panel.current_params() == {
    "method": RADIAL_NCC_METHOD,
    "psfSizeZAxis": 3.0,
    "psfSizeXAxis": 4.5,
    "psfWindowSize": 5,
    "ampThreshold": 0.10,
    "corrThreshold": 0.40,
}
assert panel.candidate_style() == {
    "marker": "target",
    "color": "#00dcff",
    "radius": 1.0,
    "opacity": 0.75,
}
win.add_point_button.click()
app.processEvents()
assert win.correction_mode == "add_point"
assert win.viewer.candidate_rows.shape[0] > 0, "no candidates from radial NCC localization"
assert any(item.data(0) == "candidate" for item in win.viewer.scene.items())
assert all(
    item.zValue() == CANDIDATE_Z
    for item in win.viewer.scene.items()
    if item.data(0) == "candidate"
)
print("main-view candidates (radial NCC default):", win.viewer.candidate_rows.shape[0])

QTest.keyClick(win.viewer, Qt.Key_Comma)
assert panel.current_method() == FS_GRADIENT_NCC_METHOD
win._add_point_refresh_timer.stop()
QTest.keyClick(win.viewer, Qt.Key_Period)
assert panel.current_method() == RADIAL_NCC_METHOD
win._add_point_refresh_timer.stop()
print("Add Point method shortcuts (, / .): OK")

panel.header_button.setChecked(True)
panel.method_combo.setCurrentIndex(
    panel.method_combo.findData(FS_GRADIENT_NCC_METHOD)
)
win._add_point_refresh_timer.stop()
win.refresh_add_point_candidates()
print("main-view candidates (FS_Gradient NCC):", win.viewer.candidate_rows.shape[0])
panel.method_combo.setCurrentIndex(panel.method_combo.findData(RADIAL_NCC_METHOD))
win._add_point_refresh_timer.stop()
win.refresh_add_point_candidates()
assert win.viewer.candidate_rows.shape[0] > 0, "no candidates from radial NCC localization"
assert "filterSize" not in win._add_point_used_params
print("main-view candidates (radial NCC):", win.viewer.candidate_rows.shape[0])
radial_ncc_shot = OUT_DIR / "add_point_inline_radial_ncc.png"
assert win.grab().save(str(radial_ncc_shot))
print("Inline Radial NCC Add Point screenshot:", radial_ncc_shot)
panel.method_combo.setCurrentIndex(
    panel.method_combo.findData(RADIAL_NON_NCC_METHOD)
)
win._add_point_refresh_timer.stop()
win.refresh_add_point_candidates()

# pick the candidate farthest from existing detections (forces an append)
import numpy as _np  # noqa: E402

existing = win.session.localized_by_frame[frame_idx]
candidates = win.viewer.candidate_rows
dists = _np.min(
    _np.hypot(
        candidates[:, None, 1] - existing[None, :, 1],
        candidates[:, None, 2] - existing[None, :, 2],
    ),
    axis=1,
)
far_idx = int(_np.argmax(dists))
n_before = win.session.localized_by_frame[frame_idx].shape[0]
_, z_cand, x_cand, _ = candidates[far_idx]
candidate_scene = win.viewer._pala_to_display_xy(z_cand, x_cand)
candidate_view = win.viewer.mapFromScene(*candidate_scene)
QTest.mouseClick(
    win.viewer.viewport(),
    Qt.LeftButton,
    pos=candidate_view,
)
app.processEvents()
n_after = win.session.localized_by_frame[frame_idx].shape[0]
assert n_after == n_before + 1, "candidate not appended"
new_local = n_after - 1
assert win.session.is_manual_detection(frame_idx, new_local)
assert win.session.local_idx_for_track_frame(tid, frame_idx) == new_local
edit = next(
    entry
    for entry in reversed(win.session.edit_log)
    if entry.action == "append_manual_detection"
)
assert edit.action == "append_manual_detection"
assert edit.payload.get("source") == "localization_candidates"
assert edit.payload.get("params", {}).get("method") == "radial"
assert win.mode_buttons["add_point"].isChecked()
assert win.correction_mode == "add_point"
assert win.viewer.candidate_rows.shape[0] > 0
print("add-a-point stays active: OK (appended detection", new_local, ")")

next_frame = frame_idx + 1 if frame_idx + 1 < win.session.n_frames else frame_idx - 1
if next_frame >= 0:
    win.set_frame_1based(next_frame + 1)
    assert win._add_point_refresh_timer.isActive()
    assert win.viewer.candidate_rows.shape == (0, 4)
    win._add_point_refresh_timer.stop()
    win.refresh_add_point_candidates()
    assert win._add_point_candidate_frame == next_frame
    print("Add Point auto-localizes after frame change: OK")
    win.set_frame_1based(frame_idx + 1)
    win._add_point_refresh_timer.stop()
    win.refresh_add_point_candidates()

win.add_point_button.click()
assert win.correction_mode == "inspect"
assert win.viewer.candidate_rows.shape == (0, 4)
print("active Add Point button toggles back to Inspect: OK")

# picking a candidate that matches an existing detection must NOT append
# (clear selection so the match path cannot open a blocking conflict dialog)
row = win.session.localized_by_frame[frame_idx][0]
n_before2 = win.session.localized_by_frame[frame_idx].shape[0]
saved_tid = win.selected_track_id
win.selected_track_id = None
win._apply_candidate(frame_idx, float(row[0]), float(row[1]), float(row[2]), {})
win.selected_track_id = saved_tid
assert win.session.localized_by_frame[frame_idx].shape[0] == n_before2
print("candidate dedup against existing detection: OK")
n_before = n_before2 - 1  # baseline before dialog append, for undo check

# human-defined mode must stage, nudge, and wait for Enter before appending
panel.method_combo.setCurrentIndex(
    panel.method_combo.findData(MANUAL_PLACEMENT_METHOD)
)
assert panel.manual_hint_label.isVisible()
assert panel.parameter_stack.isHidden()
assert panel.content.layout().indexOf(panel.eraser_button) == (
    panel.content.layout().count() - 1
)
assert panel.manual_hint_label.height() >= panel.manual_hint_label.heightForWidth(
    panel.manual_hint_label.width()
)
win.add_point_button.click()
app.processEvents()
assert win.correction_mode == "add_point"
assert win.viewer._manual_candidate_placement_enabled
assert not win._add_point_refresh_timer.isActive()
assert win.viewer.candidate_rows.shape == (0, 4)
assert win.viewer.manual_mode_badge.isVisible()
assert "MANUAL PLACEMENT ACTIVE" in win.viewer.manual_mode_badge.text()
manual_active_shot = OUT_DIR / "add_point_manual_active.png"
assert win.grab().save(str(manual_active_shot))
print("Manual Placement active screenshot:", manual_active_shot)

existing = win.session.localized_by_frame[frame_idx]
manual_z = manual_x = None
for z_try in _np.linspace(2.0, win.session.image_stack.shape[0] - 1.0, 12):
    for x_try in _np.linspace(2.0, win.session.image_stack.shape[1] - 1.0, 12):
        if _np.min(_np.hypot(existing[:, 1] - z_try, existing[:, 2] - x_try)) > 1.0:
            manual_z, manual_x = float(z_try), float(x_try)
            break
    if manual_z is not None:
        break
assert manual_z is not None and manual_x is not None
manual_scene = win.viewer._pala_to_display_xy(manual_z, manual_x)
manual_view = win.viewer.mapFromScene(*manual_scene)
n_manual_before = win.session.localized_by_frame[frame_idx].shape[0]
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=manual_view)
app.processEvents()
assert win.session.localized_by_frame[frame_idx].shape[0] == n_manual_before
draft_before = win.viewer.candidate_rows[0].copy()
assert "MANUAL DRAFT PENDING" in win.viewer.manual_mode_badge.text()
QTest.keyClick(win.viewer, Qt.Key_D, Qt.ShiftModifier)
QTest.keyClick(win.viewer, Qt.Key_W, Qt.ShiftModifier)
app.processEvents()
draft_after = win.viewer.candidate_rows[0].copy()
assert _np.isclose(draft_after[1], draft_before[1] - 0.1)
assert _np.isclose(draft_after[2], draft_before[2] + 0.1)
manual_draft_shot = OUT_DIR / "add_point_manual_draft.png"
assert win.grab().save(str(manual_draft_shot))
print("Human-defined draft screenshot:", manual_draft_shot)
QTest.keyClick(win.viewer, Qt.Key_Return)
app.processEvents()
assert win.session.localized_by_frame[frame_idx].shape[0] == n_manual_before + 1
manual_edit = next(
    entry
    for entry in reversed(win.session.edit_log)
    if entry.action == "append_manual_detection"
)
assert manual_edit.payload.get("source") == "manual_placement"
assert manual_edit.payload.get("params", {}).get("method") == MANUAL_PLACEMENT_METHOD
assert win.viewer.candidate_rows.shape == (0, 4)
assert win.correction_mode == "add_point"
assert "MANUAL PLACEMENT ACTIVE" in win.viewer.manual_mode_badge.text()
manual_local_idx = n_manual_before
manual_row = win.session.localized_by_frame[frame_idx][manual_local_idx]
manual_point_scene = win.viewer._pala_to_display_xy(manual_row[1], manual_row[2])

manual_view = win.viewer.mapFromScene(*manual_scene)
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=manual_view)
assert win.viewer.candidate_rows.shape == (1, 4)
QTest.keyClick(win.viewer, Qt.Key_Backspace)
assert win.viewer.candidate_rows.shape == (0, 4)
assert win.session.localized_by_frame[frame_idx].shape[0] == n_manual_before + 1

# The eraser protects assigned points, then discards the same Add Point row
# after Remove makes it unassigned. The underlying row stays in place until
# export preparation, so all frame-local indices remain stable in the session.
panel.eraser_button.click()
app.processEvents()
assert panel.eraser_active()
assert win.viewer._manual_detection_eraser_enabled
assert win.viewer.manual_mode_badge.isHidden()
assert panel.manual_hint_label.isVisible()
assert not panel.manual_hint_label.isEnabled()
assert not panel.method_combo.isEnabled()
manual_point_view = win.viewer.mapFromScene(*manual_point_scene)
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=manual_point_view)
app.processEvents()
assert not win.session.is_discarded_manual_detection(frame_idx, manual_local_idx)
assert "assigned to track" in win.status_label.text()

win.mode_buttons["remove"].setChecked(True)
manual_point_view = win.viewer.mapFromScene(*manual_point_scene)
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=manual_point_view)
app.processEvents()
assert win.session.track_id_for_detection(frame_idx, manual_local_idx) is None

panel.eraser_button.click()
app.processEvents()
assert win.correction_mode == "add_point"
assert panel.eraser_active()
manual_point_view = win.viewer.mapFromScene(*manual_point_scene)
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=manual_point_view)
app.processEvents()
assert win.session.is_discarded_manual_detection(frame_idx, manual_local_idx)
assert win.session.localized_by_frame[frame_idx].shape[0] == n_manual_before + 1
assert "omitted from saved data" in win.status_label.text()
eraser_shot = OUT_DIR / "add_point_eraser.png"
assert win.grab().save(str(eraser_shot))
print("Add Point eraser screenshot:", eraser_shot)

panel.eraser_button.click()
app.processEvents()
win.add_point_button.click()
assert win.correction_mode == "inspect"
print("human-defined Add Point: place/nudge/Enter/Backspace/state toggle OK")
print("Add Point eraser: assigned protection/unassigned discard/export tombstone OK")

# --- undo everything --------------------------------------------------------
n_undone = 0
while win.undo_manager.can_undo:
    win.undo()
    n_undone += 1
print("undo groups rolled back:", n_undone)
assert win.session.localized_by_frame[frame_idx].shape[0] == n_before

# --- verify status marking / list update ------------------------------------
verified_hex = win.viewer.styles["track_path_verified"].color


def _path_pens():
    return {
        it.pen().color().name()
        for it in win.viewer.overlay_items
        if isinstance(it, QGraphicsPathItem)
        and it.data(0) == "selected_track_path"
    }


selected_path_items = [
    item
    for item in win.viewer.overlay_items
    if isinstance(item, QGraphicsPathItem)
    and item.data(0) == "selected_track_path"
]
assert len(selected_path_items) == 1, selected_path_items
assert selected_path_items[0].path().elementCount() >= 2
win.mark_selected_verified()
assert win.session.track_status[tid] == "verified"
assert _path_pens() == {verified_hex}, _path_pens()
app.processEvents()
win.grab().save(str(OUT_DIR / "verified_path.png"))
win.display_panel.style_changed.emit("track_path_verified", "color", "#123456")
assert _path_pens() == {"#123456"}, _path_pens()
win.display_panel.style_changed.emit("track_path_verified", "color", verified_hex)
win.undo()
assert win.session.track_status[tid] == "unreviewed"
assert _path_pens() == {win.viewer.styles["track_path"].color}, _path_pens()
print("continuous selected-track path + verified color: OK")

# the all-lines layer also singles out verified tracks (same swatch)
frame_now = win.frame_slider.value() - 1
other_geom = next(
    g
    for g in win.viewer._ensure_track_geometry()
    if g["track_id"] != tid and g["start"] <= frame_now <= g["end"]
)
win.session.set_track_status(other_geom["track_id"], "verified")
win.viewer.render()
line_pens = {
    it.pen().color().name()
    for it in win.viewer.overlay_items
    if isinstance(it, QGraphicsPathItem) and it.zValue() == ALL_LINES_Z
}
assert verified_hex in line_pens, line_pens
win.session.set_track_status(other_geom["track_id"], "unreviewed")
win.viewer.render()
print("all-lines verified color split: OK")

# --- Unverified filter -------------------------------------------------------
win.filter_combo.setCurrentText("Unverified")
assert win.track_list.count() == win.session.n_tracks  # nothing verified yet
assert win.track_count_label.text() == f"{win.track_list.count():,} shown"
win.track_list.setCurrentRow(0)
tid_a = win.selected_track_id
win.scope_buttons[SCOPE_TRACK_PADDED].click()
assert win.playback_scope == SCOPE_TRACK_PADDED
win.mark_selected_verified()
app.processEvents()
assert win.track_list.count() == win.session.n_tracks - 1
assert win.track_count_label.text() == f"{win.track_list.count():,} shown"
assert tid_a not in win._row_by_track_id
assert win.selected_track_id is not None
assert win.selected_track_id != tid_a
assert int(win.track_list.currentItem().data(Qt.UserRole)) == win.selected_track_id
assert win.playback_scope == SCOPE_TRACK_PADDED
print(
    "Unverified filter hides verified tracks, selects the first remaining track, "
    "and preserves Track ±N scope: OK (track",
    tid_a,
    "vanished)",
)

# --- steal from a verified track demotes it to flagged -----------------------
refs = win.session.point_refs_for_track(tid_a)
ref = refs[len(refs) // 2]
win.filter_combo.setCurrentText("All")
other = next(t for t in range(win.session.n_tracks) if t != tid_a)
win.track_list.setCurrentRow(win._row_by_track_id[other])
assert win.selected_track_id == other
undo_len = len(win.undo_manager.undo_stack)
win._steal_detection(other, ref.frame_idx, ref.local_idx, tid_a)
assert win.session.track_id_for_detection(ref.frame_idx, ref.local_idx) == other
assert win.session.track_status[tid_a] == "flagged"
assert "Auto-flagged" in win.session.track_notes[tid_a]
assert len(win.undo_manager.undo_stack) == undo_len + 1  # one atomic group
win.undo()
assert win.session.track_status[tid_a] == "verified"
assert win.session.track_id_for_detection(ref.frame_idx, ref.local_idx) == tid_a
print("steal-from-verified: auto-flag + atomic undo back to verified OK")
win.session.set_track_status(tid_a, "unreviewed")

# --- Inspect click selects a track; empty image space clears it --------------
# Frame spin is the single absolute source of truth; a scope slider may retain
# a clamped value after this smoke has deliberately stepped out and back.
frame_now = win.frame_spin.value()
local_idx, owner = next(
    (local_index, track_id)
    for (frame_index, local_index), track_id in win.session.assignment_index.items()
    if frame_index == frame_now - 1
)
localization_row = win.session.localized_by_frame[frame_now - 1][local_idx]
point_scene = win.viewer._pala_to_display_xy(
    localization_row[1], localization_row[2]
)
point_view = win.viewer.mapFromScene(QPointF(*point_scene))
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=point_view)
app.processEvents()
assert win.selected_track_id == owner
assert win.frame_slider.value() == frame_now  # selection must not jump frames

blank_scene = next(
    QPointF(float(x), float(z))
    for z in range(win.session.image_stack.shape[0])
    for x in range(win.session.image_stack.shape[1])
    if win.viewer._find_detection_near(float(x), float(z)) is None
)
blank_view = win.viewer.mapFromScene(blank_scene)
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=blank_view)
app.processEvents()
assert win.selected_track_id is None
assert win.track_list.currentItem() is None
assert not win.viewer.track_plot_item.isVisible()
assert win.viewer.track_plot_item.pixmap().isNull()
assert not win.mark_verified_button.isEnabled()
assert "no track selected" in win.status_label.text()

# Clearing the selected-track overlay can change the offscreen view's scene
# extent, so remap the source-space point before the second synthetic click.
point_view = win.viewer.mapFromScene(QPointF(*point_scene))
QTest.mouseClick(win.viewer.viewport(), Qt.LeftButton, pos=point_view)
app.processEvents()
assert win.selected_track_id == owner
assert win.frame_slider.value() == frame_now
print("Inspect click select + empty-space clear without frame jump: OK")

# --- merge tracks -------------------------------------------------------------
from ulm_track_correction_gui.gui.merge_dialog import MergeTracksDialog  # noqa: E402

win.filter_combo.setCurrentText("All")
n_visible = win.track_list.count()
t_target = win.selected_track_id
t_source = next(
    t
    for t in range(win.session.n_tracks)
    if t != t_target and not _np.all(_np.isnan(win.session.tracks[t]))
)
dlg2 = MergeTracksDialog(win.session, t_target)
dlg2.source_spin.setValue(t_source)
assert dlg2.button_box.button(dlg2.button_box.StandardButton.Ok).isEnabled()
overlap = dlg2.overlap_count()
result = win.session.merge_tracks(dlg2.target_id, dlg2.source_id)
win.populate_track_list()
win._refresh_after_edit([t_target, t_source])
assert result["dropped"] == overlap
assert _np.all(_np.isnan(win.session.tracks[t_source]))
assert win.track_list.count() == n_visible - 1  # husk hidden from the list
assert win.track_count_label.text() == f"{win.track_list.count():,} shown"
assert "Merged into track" in win.session.track_notes[t_source]
print(
    f"merge track {t_source} -> {t_target}: moved {result['moved']}, "
    f"dropped {result['dropped']}"
)

win.undo()  # merge is one atomic group
assert not _np.all(_np.isnan(win.session.tracks[t_source]))
assert win.track_list.count() == n_visible
assert win.track_count_label.text() == f"{win.track_list.count():,} shown"
print("merge single-undo restores both tracks and the list row: OK")

# Stable IDs on save: re-merge, save, reload, and check slot preservation.
win.session.merge_tracks(t_target, t_source)
win.session.set_track_status(t_target, "verified")
from ulm_track_correction_gui.io.mat_io import load_correction_session as _load  # noqa: E402
from ulm_track_correction_gui.io.mat_io import save_correction_session as _save  # noqa: E402

merged_out = OUT_DIR / "merged_smoke.mat"
_save(win.session, merged_out)
reloaded_merged = _load(merged_out)
assert reloaded_merged.n_tracks == win.session.n_tracks
assert t_source in reloaded_merged.retired_track_ids
assert _np.all(_np.isnan(reloaded_merged.tracks[t_source]))
assert reloaded_merged.metadata["exported_verified_track_ids"] == [t_target]
_np.testing.assert_allclose(
    reloaded_merged.tracks[t_target],
    win.session.tracks[t_target],
    equal_nan=True,
)
assert reloaded_merged.metadata["session_track_id_to_export_id"] == {
    str(track_id): track_id for track_id in range(win.session.n_tracks)
}
print(
    "save-time Track ID preservation after merge: OK "
    f"({reloaded_merged.n_tracks} stable slots, retired source {t_source})"
)
win.undo()  # undo verification
win.undo()  # undo merge and restore for the remaining checks

# --- per-track edit history dialog -------------------------------------------
from ulm_track_correction_gui.gui.history_dialog import JUMP_FRAME_ROLE  # noqa: E402

win.open_track_history(t_target)
hist = win._history_dialog
assert hist is not None and hist.isVisible()
n_ops = hist.tree.topLevelItemCount()
assert n_ops > 0, "history dialog empty for an edited track"
summaries = [hist.tree.topLevelItem(i).text(2) for i in range(n_ops)]
assert any(s.startswith("Merge: received") for s in summaries), summaries
assert any(s.startswith("Undo:") for s in summaries), summaries
jump_item = next(
    (
        hist.tree.topLevelItem(i)
        for i in range(n_ops)
        if hist.tree.topLevelItem(i).data(0, JUMP_FRAME_ROLE) is not None
    ),
    None,
)
assert jump_item is not None, "no history entry carries a frame to jump to"
target_frame = int(jump_item.data(0, JUMP_FRAME_ROLE))
hist._on_item_double_clicked(jump_item, 0)
assert win.frame_slider.value() == target_frame
print(f"history dialog: {n_ops} operation(s), double-click jumped to frame {target_frame}")
print("history summaries (newest first):", summaries[:4])
app.processEvents()
hist.tree.expandAll()
hist.grab().save(str(OUT_DIR / "history_dialog.png"))
win._close_history_dialog()
assert win._history_dialog is None

# --- track-level initialize / split / delete --------------------------------
n_slots = win.session.n_tracks
win.initialize_new_track()
new_track_id = win.selected_track_id
assert new_track_id == n_slots
assert win.session.track_summary(new_track_id)["n_points"] == 0
assert win.track_list.count() == n_visible + 1
win.undo()
assert win.session.n_tracks == n_slots
print("initialize new track + undo stable-ID lifecycle: OK")

split_parent = next(
    track_id
    for track_id in win.session.active_nonempty_track_ids
    if win.session.track_summary(track_id)["n_points"] >= 2
)
split_frames = _np.flatnonzero(~_np.isnan(win.session.tracks[split_parent]))
split_result = win.session.split_track(split_parent, int(split_frames[0]))
split_ids = [split_result["first_child_id"], split_result["second_child_id"]]
win._refresh_after_structural_edit([split_parent, *split_ids])
assert not win.session.is_track_active(split_parent)
assert all(win.session.track_status[track_id] == "flagged" for track_id in split_ids)
assert all(win.session.source_track_id_for_track(track_id) is None for track_id in split_ids)
win.undo()
assert win.session.n_tracks == n_slots
assert win.session.is_track_active(split_parent)
print("split -> two flagged IDs + single undo: OK")

delete_target = t_source
delete_rows = {
    frame_idx: rows.copy() for frame_idx, rows in win.session.localized_by_frame.items()
}
delete_point_count = win.session.track_summary(delete_target)["n_points"]
win.filter_combo.setCurrentText("All")
win.track_list.setCurrentRow(win._row_by_track_id[delete_target])
win.scope_buttons[SCOPE_TRACK_PADDED].click()
assert win.playback_scope == SCOPE_TRACK_PADDED
original_confirm = win._confirm
win._confirm = lambda *_args: True
try:
    win.show()
    win.activateWindow()
    win.raise_()
    win.viewer.setFocus()
    app.processEvents()
    QTest.keyClick(win.viewer, Qt.Key_D)
    app.processEvents()
finally:
    win._confirm = original_confirm
assert not win.session.is_track_active(delete_target)
assert win.selected_track_id == int(win.track_list.item(0).data(Qt.UserRole))
assert win.playback_scope == SCOPE_TRACK_PADDED
for frame_idx, rows in delete_rows.items():
    _np.testing.assert_array_equal(win.session.localized_by_frame[frame_idx], rows)
win.undo()
assert win.session.is_track_active(delete_target)
assert win.selected_track_id == delete_target
assert win.playback_scope == SCOPE_TRACK_PADDED
assert delete_point_count > 0
print(
    "delete track preserves localization, selects the first remaining track, "
    "preserves Track ±N scope, and restores selection on undo: OK"
)

# --- editlog autosave written ------------------------------------------------
log_path = Path(str(DEMO) + ".editlog.jsonl")
assert log_path.exists() and log_path.stat().st_size > 0, "editlog missing"
print("editlog lines:", sum(1 for _ in log_path.open()))

# --- save corrected result ---------------------------------------------------
out = OUT_DIR / "corrected_smoke.mat"
from ulm_track_correction_gui.io.mat_io import (  # noqa: E402
    load_correction_session,
    save_correction_session,
)
from scipy.io import loadmat as _loadmat  # noqa: E402


def _assert_corrected_track_lines_schema(path, expected_track_ids):
    raw = _loadmat(path)
    assert raw["track_lines"].ndim == 2
    assert raw["track_lines"].shape[1] == 5
    actual_track_ids = (
        sorted(set(raw["track_lines"][:, 0].astype(int).tolist()))
        if raw["track_lines"].size
        else []
    )
    assert actual_track_ids == expected_track_ids
    assert not {
        "MatTracking",
        "corrected_tracks_matrix",
        "corrected_track_points",
        "corrected_track_paths",
        "mapCounter",
        "mapCounter_AA",
        "VelocityDisplacement",
        "VelocityTrackMean",
    } & set(raw)

save_correction_session(win.session, out)
reloaded = load_correction_session(out)
assert reloaded.n_tracks == win.session.n_tracks
assert reloaded.n_frames == win.session.n_frames
expected_verified_ids = [
    track_id
    for track_id in win.session.active_nonempty_track_ids
    if win.session.track_status.get(track_id) == "verified"
]
assert reloaded.active_nonempty_track_ids == expected_verified_ids
_assert_corrected_track_lines_schema(out, expected_verified_ids)
print("save/reload: OK,", out.stat().st_size, "bytes")

# The queue workflow writes the same basename into an isolated corrected/
# subdirectory and records the user's explicit dataset-level decision.
from ulm_track_correction_gui.core.dataset_queue import (  # noqa: E402
    DATASET_STATUS_COMPLETED,
)

# Keep the dataset intentionally incomplete: the user may confirm completed,
# but only verified tracks may appear in the corrected tracking arrays.
source_active_ids = list(win.session.active_nonempty_track_ids)
for final_track_id in source_active_ids:
    win.session.set_track_status(final_track_id, "flagged")
verified_export_ids = source_active_ids[:2]
for final_track_id in verified_export_ids:
    win.session.set_track_status(final_track_id, "verified")
win.populate_track_list()
win._refresh_current_dataset_progress()
win._choose_dataset_status_after_save = lambda path: DATASET_STATUS_COMPLETED
completion_prompts = []
win._confirm_incomplete_dataset_completion = lambda: completion_prompts.append(
    True
) or True
assert win.save_dataset_progress()
queue_entry = win.current_dataset_entry()
assert queue_entry.status == DATASET_STATUS_COMPLETED
assert queue_entry.verified_tracks == len(verified_export_ids)
assert queue_entry.active_tracks == len(source_active_ids)
assert queue_entry.verified_tracks < queue_entry.active_tracks
assert completion_prompts == [True]
assert queue_entry.corrected_path == (
    demo_path.parent / "corrected" / demo_path.name
).resolve(strict=False)
assert queue_entry.corrected_path.exists()
queue_reloaded = load_correction_session(queue_entry.corrected_path)
assert queue_reloaded.n_frames == win.session.n_frames
assert queue_reloaded.metadata["dataset_review_state"] == DATASET_STATUS_COMPLETED
assert queue_reloaded.metadata["verification_complete"] is False
assert queue_reloaded.metadata["completion_override"] is True
assert queue_reloaded.metadata["exported_verified_track_ids"] == verified_export_ids
assert queue_reloaded.active_nonempty_track_ids == verified_export_ids
assert all(
    queue_reloaded.track_status[track_id] == "verified"
    for track_id in queue_reloaded.active_nonempty_track_ids
)
_assert_corrected_track_lines_schema(
    queue_entry.corrected_path,
    verified_export_ids,
)
print(
    "dataset queue verified-only forced completion: OK,",
    queue_entry.corrected_path,
)

# --- screenshot --------------------------------------------------------------
win.mode_buttons["inspect"].setChecked(True)
win.display_panel.power_compression_radio.click()
win.display_panel.power_gamma_spin.setValue(0.75)
win.display_panel.power_clim_control.minimum_spin.setValue(0.142)
if win.selected_track_id is not None:
    win.scope_buttons[SCOPE_TRACK_PADDED].click()
win.display_panel.toggle_all_lines()  # back off for a clean shot? keep ON:
win.display_panel.toggle_all_lines()
win.viewer.render()
app.processEvents()
shot = OUT_DIR / "gui_smoke.png"
win.grab().save(str(shot))
print("screenshot:", shot)
navigation_shot = OUT_DIR / "frame_navigation.png"
assert win.navigation_box.grab().save(str(navigation_shot))
print("frame navigation screenshot:", navigation_shot)

# Keep a focused visual artifact for the localization status-subset controls,
# the three map selectors, and their status text. The ordinary full-window
# screenshot above captures the normal state.
for display_group in win.display_panel._layer_groups.values():
    display_group.setExpanded(False)
for layer_key in (
    "track_plot",
    "points",
    "multiple_dataset_plot",
    "dataset_plot",
):
    win.display_panel._layer_groups[layer_key].setExpanded(True)
app.processEvents()
app.processEvents()
display_shot = OUT_DIR / "display_panel_expanded.png"
assert all(
    slot.slot_control.height() == slot.group.height() for slot in fixed_slots
), [
    (slot.slot_number, slot.slot_control.height(), slot.group.height())
    for slot in fixed_slots
]
assert win.display_panel.grab().save(str(display_shot))
print("expanded Display screenshot:", display_shot)

# Capture the interaction state itself: the movable card and fixed destination
# dock must highlight independently while the other numbered docks stay put.
dragged_group = win.display_panel._layer_groups["dataset_plot"]
drag_target_slot = win.display_panel.slot(2)
win.display_panel._start_layer_drag(dragged_group)
win.display_panel._move_layer_drag(
    drag_target_slot.mapToGlobal(drag_target_slot.rect().center())
)
app.processEvents()
assert win.display_panel._drag_target_index == 1
assert drag_target_slot.slot_control.property("dropTarget") is True
assert dragged_group.property("dragging") is True
drag_shot = OUT_DIR / "display_panel_drag_target.png"
assert win.display_panel.grab().save(str(drag_shot))
win.display_panel._finish_layer_drag()
win.display_panel.move_layer("dataset_plot", DEFAULT_LAYER_ORDER.index("dataset_plot"))
assert win.display_panel.layer_order() == DEFAULT_LAYER_ORDER
print("Display drag-target screenshot:", drag_shot)

print("GUI SMOKE TEST PASSED")
