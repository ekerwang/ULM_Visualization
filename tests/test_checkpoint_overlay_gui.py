import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QGraphicsLineItem, QGraphicsPathItem

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    HARD_DENSITY,
    HARD_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    AccumulationCheckpoint,
    SparseTrackContributions,
)
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.dataset_accumulation import DatasetAccumulator
from ulm_track_correction_gui.gui.display_panel import DisplayPanel
from ulm_track_correction_gui.gui.main_window import MainWindow
from ulm_track_correction_gui.gui import main_window_session as session_module
from ulm_track_correction_gui.gui.viewer import (
    ALL_LINES_Z,
    DATASET_PLOT_Z,
    IMAGE_Z,
    LOCALIZATION_Z,
    MULTIPLE_DATASET_PLOT_Z,
    SELECTED_POINT_Z,
    TRACK_PATH_Z,
    TRACK_PLOT_Z,
    PalaSceneViewer,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session():
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray([[1.0, 1.0, 1.0, 1.0]]),
            1: np.asarray([[1.0, 2.0, 2.0, 2.0]]),
        },
        tracks=[np.asarray([0.0, 0.0])],
        image_stack=np.ones((4, 5, 2), dtype=np.float32),
    )


def make_layered_session():
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 3.0, 1.0],
                ]
            ),
            1: np.asarray(
                [
                    [1.0, 2.0, 2.0, 2.0],
                    [1.0, 2.0, 4.0, 2.0],
                ]
            ),
        },
        tracks=[np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])],
        image_stack=np.ones((4, 5, 2), dtype=np.float32),
    )


def make_checkpoint(*, mode="full", track_id=42, empty=False):
    hard_rows = np.asarray([], dtype=np.uint32) if empty else np.asarray([0, 2])
    hard_cols = np.asarray([], dtype=np.uint32) if empty else np.asarray([0, 3])
    aa_rows = np.asarray([], dtype=np.uint32) if empty else np.asarray([0, 1, 2])
    aa_cols = np.asarray([], dtype=np.uint32) if empty else np.asarray([0, 1, 3])
    aa_weight = np.asarray([], dtype=np.float32) if empty else np.asarray([0.5, 1, 2])
    hard_count = len(hard_rows)
    aa_count = len(aa_rows)
    contributions = SparseTrackContributions(
        mode=mode,
        track_ids=np.asarray([track_id]),
        hard_offsets=np.asarray([0, hard_count]),
        hard_rows=hard_rows,
        hard_cols=hard_cols,
        aa_offsets=np.asarray([0, aa_count]),
        aa_rows=aa_rows,
        aa_cols=aa_cols,
        aa_density_weight=aa_weight,
        hard_local_velocity_numerator=(np.full(hard_count, 10.0) if mode == "full" else None),
        hard_track_mean_velocity_numerator=(
            np.full(hard_count, 12.0) if mode == "full" else None
        ),
    )
    return AccumulationCheckpoint(
        schema_version=2,
        mode=mode,
        map_shape=(7, 13),
        input_shape=(4, 5),
        ulm_scale_z=2.0,
        ulm_scale_x=3.0,
        params={
            "movingAverageSpan": 1,
            "maxSamplingStepUlmpx": 0.8,
            "prfHz": 1000.0,
            "antiAliasSize": 0.5,
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 2.0,
            "velocityDisplayMax": 20.0,
        },
        metadata={
            "inputPixelSizeZUm": 20.0,
            "inputPixelSizeXUm": 30.0,
            "ulmPixelSizeZUm": 10.0,
            "ulmPixelSizeXUm": 10.0,
        },
        contributions=contributions,
        source_path="checkpoint.mat",
    )


def test_selecting_track_updates_overlay_and_frame_step_keeps_it(app):
    viewer = PalaSceneViewer()
    viewer.set_session(make_session())
    viewer.set_accumulation_checkpoint(make_checkpoint(), {0: 42})
    assert viewer.track_plot_item.pixmap().isNull()

    viewer.set_selected_track(0)
    assert not viewer.track_plot_item.pixmap().isNull()
    assert viewer.track_plot_item.isVisible()
    assert viewer.track_plot_item.zValue() == TRACK_PLOT_Z
    cache_key = viewer._track_plot_cache_key
    pixmap_key = viewer.track_plot_item.pixmap().cacheKey()

    viewer.set_frame(1)
    assert viewer._track_plot_cache_key == cache_key
    assert viewer.track_plot_item.pixmap().cacheKey() == pixmap_key
    assert viewer.track_plot_item.isVisible()


def test_overlay_items_follow_bottom_to_top_layer_order(app):
    viewer = PalaSceneViewer()
    viewer.styles["all_lines"].visible = True
    viewer.set_session(make_layered_session())
    viewer.set_accumulation_checkpoint(make_checkpoint(), {0: 42})
    viewer.set_selected_track(0)

    localization_items = [
        item for item in viewer.overlay_items if item.data(0) == "localization"
    ]
    track_tier_items = [
        item
        for item in viewer.overlay_items
        if isinstance(item, (QGraphicsLineItem, QGraphicsPathItem))
    ]

    assert localization_items
    assert {item.zValue() for item in localization_items} == {LOCALIZATION_Z}
    assert any(item.zValue() == SELECTED_POINT_Z for item in viewer.overlay_items)
    assert track_tier_items
    assert {item.zValue() for item in track_tier_items} == {
        ALL_LINES_Z,
        TRACK_PATH_Z,
    }
    assert viewer.track_plot_item.zValue() == TRACK_PLOT_Z
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


def test_dynamic_dataset_layer_is_lazy_and_updates_only_after_track_edits(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    panel.layer_order_changed.connect(viewer.set_layer_order)
    panel.dataset_plot_map_changed.connect(viewer.set_dataset_plot_map)
    panel.dataset_plot_density_clim_changed.connect(
        viewer.set_dataset_plot_density_clim
    )
    panel.dataset_plot_gamma_changed.connect(viewer.set_dataset_plot_gamma)
    panel.dataset_plot_clim_changed.connect(viewer.set_dataset_plot_velocity_clim)
    viewer.dataset_plot_status_changed.connect(panel.set_dataset_plot_status)
    checkpoint = make_checkpoint()
    viewer.set_session(make_layered_session())
    panel.configure_dataset_plotting(
        True,
        "ready",
    )
    viewer.set_accumulation_checkpoint(checkpoint, {0: 42})

    assert panel.dataset_plot_density_clim_min_spin.value() == 0.0
    assert panel.dataset_plot_density_clim_max_spin.value() == 1.0
    assert viewer.dataset_plot_density_clims[AA_DENSITY] == (0.0, 1.0)
    assert viewer.dataset_plot_density_clims[HARD_DENSITY] == (0.0, 1.0)
    assert panel.dataset_plot_gamma_slider.value() == 100
    assert panel.dataset_plot_gamma_spin.value() == 1.0
    assert viewer.dataset_plot_gamma == 1.0

    assert viewer.dataset_accumulator is None
    assert viewer.dataset_plot_item.pixmap().isNull()
    assert viewer.styles["dataset_plot"].visible is False

    panel.dataset_plot_group.setChecked(True)
    app.processEvents()
    assert viewer.dataset_accumulator is not None
    assert viewer.dataset_plot_item.isVisible()
    assert not viewer.dataset_plot_item.pixmap().isNull()
    assert viewer.dataset_plot_item.zValue() == DATASET_PLOT_Z
    assert viewer.dataset_plot_item.zValue() < viewer.track_plot_item.zValue()
    assert "current tracks" in panel.dataset_plot_status_label.text()
    assert not panel.dataset_plot_density_clim_widget.isHidden()
    assert panel.dataset_plot_velocity_clim_widget.isHidden()

    accumulator = viewer.dataset_accumulator
    revision = accumulator.revision
    cached_pixmap = viewer.dataset_plot_item.pixmap().cacheKey()
    panel.move_layer("dataset_plot", 0)
    app.processEvents()
    assert viewer.dataset_accumulator is accumulator
    assert viewer.dataset_accumulator.revision == revision
    assert viewer.dataset_plot_item.pixmap().cacheKey() == cached_pixmap
    assert viewer.dataset_plot_item.zValue() > viewer.track_plot_item.zValue()

    original_pixmap = viewer.dataset_plot_item.pixmap().cacheKey()
    panel.dataset_plot_gamma_slider.setValue(50)
    app.processEvents()
    assert panel.dataset_plot_gamma_spin.value() == 0.5
    assert viewer.dataset_plot_gamma == 0.5
    assert viewer.dataset_plot_item.pixmap().cacheKey() != original_pixmap

    original_pixmap = viewer.dataset_plot_item.pixmap().cacheKey()
    panel.dataset_plot_density_clim_max_spin.setValue(2.0)
    app.processEvents()
    assert viewer.dataset_plot_density_clims[AA_DENSITY] == (0.0, 2.0)
    assert viewer.dataset_plot_item.pixmap().cacheKey() != original_pixmap

    panel.dataset_plot_opacity_slider.setValue(40)
    app.processEvents()
    assert viewer.dataset_plot_item.opacity() == pytest.approx(0.4)

    original_pixmap = viewer.dataset_plot_item.pixmap().cacheKey()
    panel.dataset_plot_color_button.set_color("#ff4080")
    panel.dataset_plot_color_button.color_changed.emit("#ff4080")
    app.processEvents()
    assert viewer.styles["dataset_plot"].color == "#ff4080"
    assert viewer.dataset_plot_item.pixmap().cacheKey() != original_pixmap

    revision = viewer.dataset_accumulator.revision
    viewer.session.remove_assignment(0, 1)
    viewer.invalidate_track_geometry([0])
    assert viewer.dataset_accumulator.revision == revision + 1

    panel.dataset_plot_map_combo.setCurrentIndex(
        panel.dataset_plot_map_combo.findData(HARD_DENSITY)
    )
    app.processEvents()
    assert panel.dataset_plot_density_clim_min_spin.value() == 0.0
    assert panel.dataset_plot_density_clim_max_spin.value() == 1.0
    panel.dataset_plot_density_clim_min_spin.setValue(0.25)
    panel.dataset_plot_density_clim_max_spin.setValue(1.25)
    app.processEvents()
    assert viewer.dataset_plot_density_clims[HARD_DENSITY] == (0.25, 1.25)

    panel.dataset_plot_map_combo.setCurrentIndex(
        panel.dataset_plot_map_combo.findData(AA_DENSITY)
    )
    app.processEvents()
    assert panel.dataset_plot_density_clim_min_spin.value() == 0.0
    assert panel.dataset_plot_density_clim_max_spin.value() == 2.0

    panel.dataset_plot_map_combo.setCurrentIndex(
        panel.dataset_plot_map_combo.findData(HARD_LOCAL_VELOCITY)
    )
    app.processEvents()
    assert viewer.dataset_plot_map_name == HARD_LOCAL_VELOCITY
    assert panel.dataset_plot_density_clim_widget.isHidden()
    assert not panel.dataset_plot_velocity_clim_widget.isHidden()
    assert panel.dataset_plot_clim_min_spin.value() == 0.0
    assert panel.dataset_plot_clim_max_spin.value() == 1.0


def test_multiple_dataset_snapshot_survives_current_session_and_checkpoint_changes(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    panel.multiple_dataset_plot_map_changed.connect(
        viewer.set_multiple_dataset_plot_map
    )
    panel.multiple_dataset_plot_density_clim_changed.connect(
        viewer.set_multiple_dataset_plot_density_clim
    )
    panel.multiple_dataset_plot_gamma_changed.connect(
        viewer.set_multiple_dataset_plot_gamma
    )
    panel.multiple_dataset_plot_clim_changed.connect(
        viewer.set_multiple_dataset_plot_velocity_clim
    )
    viewer.multiple_dataset_plot_status_changed.connect(
        panel.set_multiple_dataset_plot_status
    )

    checkpoint = make_checkpoint()
    accumulator = DatasetAccumulator.build_multiple(
        (
            (make_layered_session(), checkpoint),
            (make_layered_session(), checkpoint),
        )
    )
    viewer.set_session(make_layered_session())
    viewer.set_multiple_dataset_accumulation(
        accumulator,
        ("first.mat", "second.mat"),
    )
    panel.configure_multiple_dataset_plotting(
        True,
        "ready",
    )
    panel.multiple_dataset_plot_group.setChecked(True)
    app.processEvents()

    assert viewer.multiple_dataset_accumulator is accumulator
    assert viewer.multiple_dataset_source_paths == ("first.mat", "second.mat")
    assert viewer.multiple_dataset_plot_item.isVisible()
    assert not viewer.multiple_dataset_plot_item.pixmap().isNull()
    assert viewer.multiple_dataset_plot_item.zValue() == MULTIPLE_DATASET_PLOT_Z
    assert "across 2 imported datasets" in panel.multiple_dataset_plot_status_label.text()
    assert panel.multiple_dataset_plot_density_clim_min_spin.value() == 0.0
    assert panel.multiple_dataset_plot_density_clim_max_spin.value() == 1.0
    assert panel.multiple_dataset_plot_gamma_spin.value() == 1.0
    snapshot_revision = accumulator.revision
    snapshot_density = accumulator.density.copy()

    original_pixmap = viewer.multiple_dataset_plot_item.pixmap().cacheKey()
    panel.multiple_dataset_plot_gamma_spin.setValue(0.25)
    app.processEvents()
    assert panel.multiple_dataset_plot_gamma_slider.value() == 25
    assert viewer.multiple_dataset_plot_gamma == 0.25
    assert viewer.multiple_dataset_plot_item.pixmap().cacheKey() != original_pixmap

    viewer.set_session(make_layered_session())
    viewer.set_accumulation_checkpoint(checkpoint, {0: 42})
    viewer.session.remove_assignment(0, 1)
    viewer.invalidate_track_geometry([0])
    app.processEvents()

    assert viewer.multiple_dataset_accumulator is accumulator
    assert accumulator.revision == snapshot_revision
    np.testing.assert_array_equal(accumulator.density, snapshot_density)
    assert viewer.multiple_dataset_plot_item.isVisible()
    assert not viewer.multiple_dataset_plot_item.pixmap().isNull()

    panel.multiple_dataset_plot_opacity_slider.setValue(35)
    app.processEvents()
    assert viewer.multiple_dataset_plot_item.opacity() == pytest.approx(0.35)


def test_multiple_dataset_import_button_stays_available_without_snapshot(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    requests = []
    panel.multiple_dataset_import_requested.connect(lambda: requests.append(True))

    group = panel.multiple_dataset_plot_group
    assert group.isEnabled()
    assert group.isAncestorOf(panel.import_multiple_dataset_button)
    assert not group.isExpanded()
    assert not panel.multiple_dataset_plot_opacity_slider.isEnabled()

    group.header_button.click()
    assert not group.isChecked()
    assert group.isExpanded()
    assert panel.import_multiple_dataset_button.isEnabled()
    panel.import_multiple_dataset_button.click()
    assert requests == [True]


def test_multiple_dataset_import_is_atomic_and_replaces_only_after_success(
    app,
    tmp_path,
    monkeypatch,
):
    settings = QSettings(str(tmp_path / "multiple.ini"), QSettings.IniFormat)
    window = MainWindow(dataset_settings=settings)
    first = tmp_path / "first_results.mat"
    second = tmp_path / "second_results.mat"
    checkpoint = make_checkpoint()
    warnings = []
    selected_paths = [str(first), str(second)]
    dialog_requests = []

    class FakeDatasetImportDialog:
        def __init__(self, parent, *, start_dir, title):
            dialog_requests.append((parent, start_dir, title))

        def exec(self):
            return session_module.QFileDialog.Accepted

        def selectedFiles(self):
            return list(selected_paths)

    monkeypatch.setattr(
        session_module,
        "DatasetImportDialog",
        FakeDatasetImportDialog,
    )
    monkeypatch.setattr(
        session_module,
        "load_correction_session",
        lambda path: make_layered_session(),
    )
    monkeypatch.setattr(
        session_module,
        "discover_accumulation_checkpoint",
        lambda path, metadata: SimpleNamespace(
            path=tmp_path / "shared_accumulation_checkpoint.mat",
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        session_module,
        "load_accumulation_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        session_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )
    try:
        window.import_multiple_dataset_accumulation()
        imported = window.viewer.multiple_dataset_accumulator
        assert imported is not None
        assert imported.dataset_count == 2
        assert window.display_panel.multiple_dataset_plot_group.isChecked()
        assert warnings == []
        assert dialog_requests == [
            (
                window,
                "",
                "Import datasets for Multi Dataset Accumulation (ULM)",
            )
        ]

        selected_paths[:] = [str(tmp_path / "broken.mat")]
        monkeypatch.setattr(
            session_module,
            "load_correction_session",
            lambda path: (_ for _ in ()).throw(ValueError("broken input")),
        )
        window.import_multiple_dataset_accumulation()

        assert window.viewer.multiple_dataset_accumulator is imported
        assert len(warnings) == 1
        assert "kept unchanged" in warnings[0][2]
    finally:
        window.close()
        settings.clear()


def test_opacity_slider_and_master_toggle_control_persistent_item(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    panel.track_plot_map_changed.connect(viewer.set_track_plot_map)
    viewer.track_plot_status_changed.connect(panel.set_track_plot_status)
    checkpoint = make_checkpoint()
    panel.configure_track_plotting(checkpoint.supported_track_maps, "ready")
    viewer.set_session(make_session())
    viewer.set_accumulation_checkpoint(checkpoint, {0: 42})
    viewer.set_selected_track(0)

    panel.track_plot_opacity_slider.setValue(35)
    app.processEvents()
    assert viewer.track_plot_item.opacity() == pytest.approx(0.35)

    panel.toggle_overlays()
    app.processEvents()
    assert viewer.styles["track_plot"].visible is False
    assert viewer.track_plot_item.isVisible() is False
    panel.toggle_overlays()
    app.processEvents()
    assert viewer.styles["track_plot"].visible is True
    assert viewer.track_plot_item.isVisible() is True
    assert viewer.track_plot_item.opacity() == pytest.approx(0.35)


def test_density_color_swatch_rebuilds_the_track_plot_pixmap(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    viewer.set_session(make_session())
    viewer.set_accumulation_checkpoint(make_checkpoint(), {0: 42})
    viewer.set_selected_track(0)

    original_pixmap_key = viewer.track_plot_item.pixmap().cacheKey()
    panel.track_plot_color_button.set_color("#ff4080")
    panel.track_plot_color_button.color_changed.emit("#ff4080")
    app.processEvents()

    assert viewer.styles["track_plot"].color == "#ff4080"
    assert viewer.track_plot_item.pixmap().cacheKey() != original_pixmap_key
    image = viewer.track_plot_item.pixmap().toImage()
    occupied = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    ]
    assert occupied
    assert all(
        (pixel.red(), pixel.green(), pixel.blue()) == (255, 64, 128)
        for pixel in occupied
    )


def test_map_switch_replaces_density_swatch_with_velocity_clim(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.track_plot_map_changed.connect(viewer.set_track_plot_map)
    panel.track_plot_clim_changed.connect(viewer.set_track_plot_velocity_clim)
    checkpoint = make_checkpoint()
    viewer.set_session(make_session())
    viewer.set_accumulation_checkpoint(checkpoint, {0: 42})
    viewer.set_selected_track(0)
    panel.configure_track_plotting(
        checkpoint.supported_track_maps,
        "ready",
        velocity_display_max=20.0,
    )

    panel.track_plot_map_combo.setCurrentIndex(
        panel.track_plot_map_combo.findData(AA_DENSITY)
    )
    app.processEvents()
    assert not panel.track_plot_color_label.isHidden()
    assert not panel.track_plot_color_button.isHidden()
    assert panel.track_plot_velocity_clim_widget.isHidden()

    panel.track_plot_map_combo.setCurrentIndex(
        panel.track_plot_map_combo.findData(HARD_LOCAL_VELOCITY)
    )
    app.processEvents()
    assert panel.track_plot_color_label.isHidden()
    assert panel.track_plot_color_button.isHidden()
    assert not panel.track_plot_velocity_clim_widget.isHidden()
    assert panel.track_plot_clim_min_spin.value() == 0.0
    assert panel.track_plot_clim_max_spin.value() == 20.0

    original_pixmap_key = viewer.track_plot_item.pixmap().cacheKey()
    panel.track_plot_clim_min_spin.setValue(5.0)
    panel.track_plot_clim_max_spin.setValue(10.0)
    app.processEvents()

    assert viewer.track_plot_velocity_clim == (5.0, 10.0)
    assert viewer.track_plot_item.pixmap().cacheKey() != original_pixmap_key
    image = viewer.track_plot_item.pixmap().toImage()
    occupied = [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    ]
    assert occupied
    assert all(
        (pixel.red(), pixel.green(), pixel.blue(), pixel.alpha())
        == (128, 0, 0, 255)
        for pixel in occupied
    )


def test_density_only_disables_velocity_choices(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    checkpoint = make_checkpoint(mode="density_only")
    panel.configure_track_plotting(checkpoint.supported_track_maps, "density only")

    density_item = panel.track_plot_map_combo.model().item(
        panel.track_plot_map_combo.findData(AA_DENSITY)
    )
    velocity_item = panel.track_plot_map_combo.model().item(
        panel.track_plot_map_combo.findData(HARD_LOCAL_VELOCITY)
    )
    assert density_item.isEnabled()
    assert not velocity_item.isEnabled()
    assert panel.track_plot_group.isEnabled()

    panel.configure_track_plotting((), "unavailable")
    assert panel.track_plot_group.isEnabled()
    assert panel.track_plot_group.header_button.isEnabled()
    track_plot_slot = panel.slot(panel.layer_order().index("track_plot") + 1)
    assert not track_plot_slot.visibility_button.isEnabled()
    assert not panel.track_plot_map_combo.isEnabled()


def test_schema_v2_omits_aa_velocity_choices(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    checkpoint = make_checkpoint()

    panel.configure_track_plotting(checkpoint.supported_track_maps, "schema v2 ready")

    assert checkpoint.supported_track_maps == (
        HARD_DENSITY,
        AA_DENSITY,
        HARD_LOCAL_VELOCITY,
        HARD_TRACK_MEAN_VELOCITY,
    )
    assert panel.track_plot_map_combo.count() == 4
    for map_name in checkpoint.supported_track_maps:
        index = panel.track_plot_map_combo.findData(map_name)
        assert index >= 0
        assert panel.track_plot_map_combo.model().item(index).isEnabled()
    assert panel.track_plot_map_combo.findData("aa_local_velocity") == -1
    assert panel.track_plot_map_combo.findData("aa_track_mean_velocity") == -1
    assert panel.track_plot_map_combo.findText("AA Local Velocity") == -1
    assert panel.track_plot_map_combo.findText("AA Track-Mean Velocity") == -1
    assert panel.track_plot_group.isEnabled()


def test_unknown_track_and_empty_contribution_clear_without_crashing(app):
    viewer = PalaSceneViewer()
    statuses = []
    viewer.track_plot_status_changed.connect(statuses.append)
    viewer.set_session(make_session())

    viewer.set_accumulation_checkpoint(make_checkpoint(track_id=99), {0: 42})
    viewer.set_selected_track(0)
    assert viewer.track_plot_item.pixmap().isNull()
    assert "unavailable" in statuses[-1]

    viewer.set_accumulation_checkpoint(make_checkpoint(empty=True), {0: 42})
    assert viewer.track_plot_item.pixmap().isNull()
    assert "empty contribution" in statuses[-1]


def test_interpolation_aspect_ratio_keeps_checkpoint_overlay_aligned(app):
    viewer = PalaSceneViewer()
    viewer.set_session(make_session())
    viewer.set_accumulation_checkpoint(make_checkpoint(), {0: 42})
    viewer.set_selected_track(0)

    base_y = viewer.track_plot_item.pos().y()
    base_scale_y = viewer.track_plot_item.transform().m22()
    pixmap_key = viewer.track_plot_item.pixmap().cacheKey()

    viewer.set_interpolation(True, factor=5.0, aspect_ratio=2.0)

    assert viewer.track_plot_item.pos().y() == pytest.approx(base_y * 2.0)
    assert viewer.track_plot_item.transform().m22() == pytest.approx(base_scale_y * 2.0)
    assert viewer.track_plot_item.pixmap().cacheKey() == pixmap_key
