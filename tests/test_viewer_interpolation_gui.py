import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QScrollArea,
    QSlider,
)

from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_FLAGGED,
    TRACK_STATUS_VERIFIED,
    CorrectionSession,
)
from ulm_track_correction_gui.gui.display_panel import (
    OVERLAY_LAYERS,
    ColorButton,
    DisplayPanel,
)
from ulm_track_correction_gui.gui.display_compression import (
    DEFAULT_POWER_CLIM_MAX,
    DEFAULT_POWER_CLIM_MIN,
    DEFAULT_POWER_GAMMA,
    DISPLAY_MODE_DB,
    DISPLAY_MODE_POWER,
    normalize_power_frame,
)
from ulm_track_correction_gui.gui.layer_styles import (
    DEFAULT_DB_MAX,
    DEFAULT_DB_MIN,
    DEFAULT_IMAGE_OPACITY,
    DEFAULT_MULTIPLE_DATASET_OPACITY,
)
from ulm_track_correction_gui.gui.layer_order import DEFAULT_LAYER_ORDER, LAYER_LABELS
from ulm_track_correction_gui.gui.viewer import (
    TRACK_PATH_Z,
    PalaSceneViewer,
    interpolate_frame_for_display,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def make_session():
    return CorrectionSession(
        localized_by_frame={0: np.asarray([[1.0, 2.0, 2.0, 1.0]])},
        tracks=[np.asarray([0.0])],
        image_stack=np.arange(20, dtype=np.float32).reshape(4, 5, 1),
    )


def make_selection_session():
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 3.0, 4.0, 1.0],
                ]
            )
        },
        tracks=[np.asarray([0.0])],
        image_stack=np.arange(20, dtype=np.float32).reshape(4, 5, 1),
    )


def make_power_session():
    stack = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 1.0],
                [1.0, 0.5, 4.0],
                [4.0, 1.0, 9.0],
            ]
        ],
        dtype=np.float32,
    )
    return CorrectionSession(
        localized_by_frame={0: np.asarray([[1.0, 1.0, 1.0, 1.0]])},
        tracks=[np.asarray([0.0, np.nan, np.nan])],
        image_stack=stack,
    )


def make_track_path_session():
    return CorrectionSession(
        localized_by_frame={
            frame_idx: np.asarray([[1.0, z, x, frame_idx + 1.0]])
            for frame_idx, (z, x) in enumerate(
                ((1.0, 2.0), (2.0, 3.0), (3.0, 5.0))
            )
        },
        tracks=[np.asarray([0.0, 0.0, 0.0])],
        image_stack=np.ones((4, 6, 3), dtype=np.float32),
    )


def make_status_points_session():
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 3.0, 1.0],
                    [1.0, 1.0, 4.0, 1.0],
                    [1.0, 1.0, 5.0, 1.0],
                ]
            )
        },
        tracks=[np.asarray([float(local_idx)]) for local_idx in range(4)],
        image_stack=np.ones((2, 5, 1), dtype=np.float32),
        track_status={
            0: TRACK_STATUS_VERIFIED,
            1: TRACK_STATUS_VERIFIED,
            2: TRACK_STATUS_FLAGGED,
        },
    )


def test_display_panel_interpolation_defaults_and_state_signal(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    changes = []
    panel.interpolation_changed.connect(lambda *args: changes.append(args))

    assert panel.interp_factor_spin.value() == 5.0
    assert panel.aspect_ratio_spin.value() == 1.0
    assert panel.interpolate_button.isCheckable()
    assert not panel.interpolate_button.isChecked()
    assert panel.interpolate_button.property("stateButton") is True
    interpolate_height = panel.interpolate_button.sizeHint().height()

    panel.interpolate_button.click()
    assert changes[-1] == (True, 5.0, 1.0)
    assert panel.interpolate_button.text() == "Interpolated (I)"
    assert panel.interpolate_button.sizeHint().height() == interpolate_height

    panel.interp_factor_spin.setValue(3.0)
    assert changes[-1] == (True, 3.0, 1.0)

    panel.interpolate_button.click()
    assert changes[-1] == (False, 3.0, 1.0)
    assert panel.interpolate_button.text() == "Interpolate Dataset (I)"
    assert panel.interpolate_button.sizeHint().height() == interpolate_height


def test_dynamic_range_controls_drive_mutually_exclusive_viewer_modes(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.display_settings_changed.connect(viewer.set_display_settings)
    session = make_power_session()
    viewer.set_session(session)

    assert not panel.db_compression_radio.isChecked()
    assert panel.power_compression_radio.isChecked()
    assert panel.compression_options_stack.currentIndex() == 1
    assert panel.power_gamma_spin.value() == DEFAULT_POWER_GAMMA
    assert panel.power_clim_control.values() == (
        DEFAULT_POWER_CLIM_MIN,
        DEFAULT_POWER_CLIM_MAX,
    )
    assert viewer.display_settings.mode == DISPLAY_MODE_POWER
    expected = normalize_power_frame(
        session.image_stack[:, :, 0],
        reference=4.0,
        gamma=DEFAULT_POWER_GAMMA,
        clim_min=DEFAULT_POWER_CLIM_MIN,
        clim_max=DEFAULT_POWER_CLIM_MAX,
    )
    image = viewer.image_item.pixmap().toImage()
    rendered = np.asarray(
        [
            [image.pixelColor(x, y).red() for x in range(image.width())]
            for y in range(image.height())
        ]
    )
    np.testing.assert_array_equal(rendered, expected)

    panel.power_gamma_spin.setValue(0.75)
    panel.power_clim_control.minimum_spin.setValue(0.142)
    assert viewer.display_settings.gamma == 0.75
    assert (viewer.display_settings.clim_min, viewer.display_settings.clim_max) == (
        0.142,
        DEFAULT_POWER_CLIM_MAX,
    )

    panel.power_clim_control.slider.setValues(0.2, 0.8)
    assert panel.power_clim_control.minimum_spin.value() == 0.2
    assert panel.power_clim_control.maximum_spin.value() == 0.8
    assert (viewer.display_settings.clim_min, viewer.display_settings.clim_max) == (
        0.2,
        0.8,
    )

    panel.db_compression_radio.click()
    assert panel.compression_options_stack.currentIndex() == 0
    assert panel.db_compression_radio.isChecked()
    assert not panel.power_compression_radio.isChecked()
    assert viewer.display_settings.mode == DISPLAY_MODE_DB


def test_selected_track_uses_one_continuous_path_item(app):
    viewer = PalaSceneViewer()
    viewer.set_session(make_track_path_session())
    viewer.set_selected_track(0)

    track_items = [
        item
        for item in viewer.overlay_items
        if item.zValue() == TRACK_PATH_Z
        and isinstance(item, (QGraphicsLineItem, QGraphicsPathItem))
    ]

    assert len(track_items) == 1
    item = track_items[0]
    assert isinstance(item, QGraphicsPathItem)
    assert item.data(0) == "selected_track_path"
    assert item.data(1) == 0
    assert item.path().elementCount() == 3
    assert [
        (item.path().elementAt(index).x, item.path().elementAt(index).y)
        for index in range(item.path().elementCount())
    ] == pytest.approx([(1.5, 0.5), (2.5, 1.5), (4.5, 2.5)])
    assert item.pen().color().alpha() == round(
        255 * viewer.styles["track_path"].opacity
    )
    assert item.pen().capStyle() == Qt.RoundCap
    assert item.pen().joinStyle() == Qt.RoundJoin


def test_display_panel_state_buttons_share_style_and_report_state(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)

    assert panel.toggle_button.isCheckable()
    assert panel.toggle_button.property("stateButton") is True
    assert (
        panel.toggle_button.sizeHint().height()
        == panel.interpolate_button.sizeHint().height()
    )
    assert not panel.toggle_button.isChecked()
    assert panel.toggle_button.text() == "Keep images only (T)"

    panel.toggle_overlays()
    assert panel.toggle_button.isChecked()
    assert panel.toggle_button.text() == "Restore layers (T)"

    panel.toggle_overlays()
    assert not panel.toggle_button.isChecked()
    assert panel.toggle_button.text() == "Keep images only (T)"


def test_inspect_click_selects_track_clears_background_and_ignores_drag(app):
    viewer = PalaSceneViewer()
    viewer.resize(500, 400)
    viewer.show()
    viewer.set_session(make_selection_session())
    app.processEvents()

    picked = []
    cleared = []
    edit_clicks = []
    viewer.track_picked.connect(picked.append)
    viewer.track_selection_cleared.connect(lambda: cleared.append(True))
    viewer.detection_clicked.connect(lambda *args: edit_clicks.append(args))

    assigned_pos = viewer.mapFromScene(QPointF(1.5, 1.5))
    unassigned_pos = viewer.mapFromScene(QPointF(3.5, 2.5))
    blank_pos = viewer.mapFromScene(QPointF(0.5, 3.5))

    QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=assigned_pos)
    assert picked == [0]

    viewer.set_selected_track(0)
    QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=unassigned_pos)
    assert cleared == []

    QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=blank_pos)
    assert cleared == [True]

    cleared.clear()
    viewer.set_selected_track(0)
    drag_distance = QApplication.startDragDistance() + 10
    QTest.mousePress(viewer.viewport(), Qt.LeftButton, pos=blank_pos)
    QTest.mouseMove(viewer.viewport(), blank_pos + QPoint(drag_distance, 0))
    QTest.mouseRelease(
        viewer.viewport(),
        Qt.LeftButton,
        pos=blank_pos + QPoint(drag_distance, 0),
    )
    assert cleared == []

    viewer.set_interaction_mode("assign")
    QTest.mouseClick(viewer.viewport(), Qt.LeftButton, pos=assigned_pos)
    assert edit_clicks == [(0, 0)]
    assert picked == [0]
    assert cleared == []


def test_reviewed_visual_defaults_match_display_controls(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)

    assert (viewer.db_min, viewer.db_max) == (DEFAULT_DB_MIN, DEFAULT_DB_MAX)
    assert (panel.db_min_spin.value(), panel.db_max_spin.value()) == (
        DEFAULT_DB_MIN,
        DEFAULT_DB_MAX,
    )
    expected = {
        "image": (DEFAULT_IMAGE_OPACITY, None),
        "dataset_plot": (1.0, None),
        "multiple_dataset_plot": (DEFAULT_MULTIPLE_DATASET_OPACITY, None),
        "track_plot": (1.0, None),
        "all_lines": (0.43, 0.5),
        "points": (0.51, 0.5),
        "track_path": (0.58, 0.5),
        "track_point": (0.20, 2.0),
    }
    for layer_key, (opacity, size) in expected.items():
        style = viewer.styles[layer_key]
        assert style.opacity == opacity
        group = panel._layer_groups[layer_key]
        assert group.findChild(QSlider).value() == round(opacity * 100)
        if size is not None:
            assert style.size == size
            assert group.findChild(QDoubleSpinBox).value() == size

    assert viewer.styles["all_lines"].visible is False
    assert viewer.styles["dataset_plot"].visible is False
    assert viewer.styles["points_current"].color == "#f050dc"
    assert viewer.styles["points_other"].color == "#3cf078"
    assert viewer.styles["points_verified"].visible is True
    assert viewer.styles["points_verified"].color == "#969696"
    assert viewer.styles["points_flagged"].visible is True
    assert viewer.styles["points_flagged"].color == "#e5484d"
    assert viewer.styles["points"].color == "#ff9632"
    assert viewer.styles["track_plot"].color == "#00d8ff"
    assert viewer.styles["dataset_plot"].color == "#00d8ff"
    assert panel.dataset_plot_color_button._color_hex == "#00d8ff"
    assert panel.track_plot_color_button._color_hex == "#00d8ff"
    assert "Hard/AA Density" in panel.track_plot_color_button.toolTip()
    localization_swatches = panel._layer_groups["points"].findChildren(ColorButton)
    assert [button.toolTip() for button in localization_swatches] == [
        "Points of the selected track",
        "Points of other tracks",
        "Unassigned detections",
        "Override verified points from other tracks with this color",
        "Override flagged points from other tracks with this color",
    ]
    assert [button._color_hex for button in localization_swatches] == [
        "#f050dc",
        "#3cf078",
        "#ff9632",
        "#969696",
        "#e5484d",
    ]
    status_checkboxes = panel._layer_groups["points"].findChildren(QCheckBox)
    assert [checkbox.text() for checkbox in status_checkboxes] == [
        "Enable verified points",
        "Enable flagged points",
    ]
    assert all(checkbox.isChecked() for checkbox in status_checkboxes)


def test_localization_status_colors_override_only_other_track_points(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    viewer.set_session(make_status_points_session())
    viewer.set_selected_track(0)

    def rendered_colors():
        return {
            int(item.data(2)): item.pen().color().name()
            for item in viewer.overlay_items
            if item.data(0) == "localization"
        }

    assert rendered_colors() == {
        0: "#f050dc",  # selected stays selected even though its track is verified
        1: "#969696",
        2: "#e5484d",
        3: "#3cf078",
        4: "#ff9632",
    }

    panel.style_changed.emit("points_flagged", "color", "#b00020")
    assert rendered_colors()[2] == "#b00020"

    panel.points_verified_checkbox.click()
    assert rendered_colors()[1] == "#3cf078"
    assert rendered_colors()[2] == "#b00020"

    panel.points_flagged_checkbox.click()
    assert rendered_colors()[2] == "#3cf078"

    panel.style_changed.emit("points", "size", 2.0)
    localization_items = [
        item for item in viewer.overlay_items if item.data(0) == "localization"
    ]
    assert {item.rect().width() for item in localization_items} == {4.0}


def test_display_panel_orders_cards_foreground_to_background(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    slots = [panel.slot(slot_number) for slot_number in range(1, 9)]
    group_titles = [slot.group.title() for slot in slots]

    assert panel.layer_order() == DEFAULT_LAYER_ORDER
    assert OVERLAY_LAYERS == tuple(
        key for key in DEFAULT_LAYER_ORDER if key != "image"
    )
    assert group_titles == [
        "Selected Track (sub-pixel plotting)",
        "Selected Track",
        "Selected Track Point",
        "Localization Points",
        "All Tracks",
        "Ultrasound Movie",
        "Single Dataset Accumulation",
        "Multi Dataset Accumulation (ULM)",
    ]
    assert group_titles == [LAYER_LABELS[key] for key in DEFAULT_LAYER_ORDER]
    assert [slot.shortcut_badge.text() for slot in slots] == [
        f"({slot_number})" for slot_number in range(1, 9)
    ]
    assert all(slot.slot_control.width() == 30 for slot in slots)
    assert all(slot.layout().spacing() == 5 for slot in slots)
    assert all(not slot.slot_control.isAncestorOf(slot.group) for slot in slots)
    assert all(slot.group.parentWidget() is slot for slot in slots)
    assert all(slot.visibility_button is slot.shortcut_badge for slot in slots)
    assert all(slot.shortcut_badge.icon().isNull() for slot in slots)


def test_slot_base_grows_with_its_card_and_uses_number_as_toggle(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.resize(300, 900)
    panel.show()
    app.processEvents()
    try:
        group = panel._layer_groups["points"]
        slot = panel.slot(panel.layer_order().index("points") + 1)
        collapsed_height = slot.slot_control.height()
        assert collapsed_height == group.height()
        assert slot.shortcut_badge.text() == "(4)"

        group.header_button.click()
        app.processEvents()
        assert slot.slot_control.height() == group.height()
        assert slot.slot_control.height() > collapsed_height

        visible_before = group.isChecked()
        slot.shortcut_badge.click()
        assert group.isChecked() is not visible_before
    finally:
        panel.close()


def test_expanded_map_status_text_uses_scroll_area_without_clipping(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    scroll = QScrollArea()
    scroll.setWidget(panel)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.resize(330, 700)

    panel.configure_track_plotting(
        ("hard_density", "aa_density"),
        "Batch checkpoint overlay: Hard Density map for selected track.",
    )
    panel.configure_multiple_dataset_plotting(
        False,
        "Import multiple dataset MAT files to create a retained ULM accumulation.",
    )
    panel.configure_dataset_plotting(
        True,
        "Ready to accumulate every current track on the original PALA grid. "
        "Enable this layer to build it.",
    )
    for layer_key in ("track_plot", "multiple_dataset_plot", "dataset_plot"):
        panel._layer_groups[layer_key].setExpanded(True)

    scroll.show()
    app.processEvents()
    try:
        assert scroll.verticalScrollBar().maximum() > 0
        for label in (
            panel.track_plot_status_label,
            panel.multiple_dataset_plot_status_label,
            panel.dataset_plot_status_label,
        ):
            assert label.width() > 205
            assert label.height() >= label.heightForWidth(label.width())
        for layer_key in ("track_plot", "multiple_dataset_plot", "dataset_plot"):
            group = panel._layer_groups[layer_key]
            slot = panel.slot(panel.layer_order().index(layer_key) + 1)
            assert slot.slot_control.height() == group.height()

        panel.set_track_plot_status(
            "Batch checkpoint overlay: this longer map status changed after the "
            "card was already expanded and must remain fully readable."
        )
        app.processEvents()
        label = panel.track_plot_status_label
        assert label.height() >= label.heightForWidth(label.width())
    finally:
        scroll.close()


def test_image_visibility_and_layer_reordering_drive_the_viewer(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.style_changed.connect(viewer.update_style)
    panel.layer_order_changed.connect(viewer.set_layer_order)
    viewer.set_session(make_session())

    image_slot_number = panel.layer_order().index("image") + 1
    image_slot = panel.slot(image_slot_number)
    assert not image_slot.visibility_button.isHidden()
    assert panel.image_group.isChecked()
    image_slot.visibility_button.click()
    app.processEvents()
    assert viewer.styles["image"].visible is False
    assert viewer.image_item.isVisible() is False
    image_slot.visibility_button.click()
    app.processEvents()
    assert viewer.styles["image"].visible is True
    assert viewer.image_item.isVisible() is True

    changes = []
    panel.layer_order_changed.connect(changes.append)
    fixed_slots = tuple(panel._slots)
    panel.move_layer("dataset_plot", 0)
    app.processEvents()
    assert panel.layer_order()[0] == "dataset_plot"
    assert viewer.layer_order() == panel.layer_order()
    assert viewer.layer_z("dataset_plot") > viewer.layer_z("track_plot")
    assert changes == [panel.layer_order()]
    assert tuple(panel._slots) == fixed_slots
    assert panel.slot_layer_key(1) == "dataset_plot"
    assert panel.slot(1).shortcut_badge.text() == "(1)"


def test_dragging_a_layer_title_reorders_the_stack(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    panel.resize(280, 720)
    panel.show()
    app.processEvents()
    try:
        dragged = panel._layer_groups["dataset_plot"]
        foreground = panel._layer_groups["track_plot"]
        start = dragged.header_button.rect().center()
        target_global = foreground.mapToGlobal(foreground.rect().topLeft())
        target = dragged.header_button.mapFromGlobal(target_global)
        changes = []
        panel.layer_order_changed.connect(changes.append)

        QTest.mousePress(dragged.header_button, Qt.LeftButton, pos=start)
        QTest.mouseMove(dragged.header_button, target, delay=10)
        app.processEvents()
        assert panel._drag_target_index == 0
        assert panel.slot(1).slot_control.property("dropTarget") is True
        QTest.mouseRelease(dragged.header_button, Qt.LeftButton, pos=target)
        app.processEvents()

        assert panel.layer_order()[0] == "dataset_plot"
        assert changes == [panel.layer_order()]
        assert not dragged.isExpanded()
        assert panel._drag_target_index is None
        assert all(not slot.slot_control.property("dropTarget") for slot in panel._slots)
    finally:
        panel.close()


def test_display_sections_keep_expansion_and_slot_visibility_independent(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)
    sections = list(panel._layer_groups.values())

    assert all(not section.isExpanded() for section in sections)
    assert all(section.header_button.property("layerHeader") for section in sections)

    selected_point = panel._layer_groups["track_point"]
    assert selected_point.isChecked()
    selected_point.header_button.click()
    assert selected_point.isChecked()
    assert selected_point.isExpanded()
    panel.toggle_overlays()
    assert not selected_point.isChecked()
    assert selected_point.isExpanded()
    panel.toggle_overlays()
    assert selected_point.isChecked()
    assert selected_point.isExpanded()

    visibility_changes = []
    panel.style_changed.connect(
        lambda layer, field, value: visibility_changes.append((layer, field, value))
    )
    track_lines = panel._layer_groups["all_lines"]
    assert not track_lines.isChecked()
    track_lines.header_button.click()
    assert not track_lines.isChecked()
    assert track_lines.isExpanded()
    assert not visibility_changes
    track_lines_slot = panel.slot(panel.layer_order().index("all_lines") + 1)
    track_lines_slot.visibility_button.click()
    assert track_lines.isChecked()
    assert track_lines.isExpanded()
    assert visibility_changes[-1] == ("all_lines", "visible", True)
    track_lines_slot.visibility_button.click()
    assert not track_lines.isChecked()
    assert track_lines.isExpanded()


def test_multiswatch_size_controls_share_one_compact_row(app):
    viewer = PalaSceneViewer()
    panel = DisplayPanel(viewer.styles)

    for layer_key in ("track_path", "points"):
        group = panel._layer_groups[layer_key]
        spin = group.findChild(QDoubleSpinBox, f"{layer_key}SizeSpin")
        grid = group.content_widget.layout()
        row, column, row_span, column_span = grid.getItemPosition(grid.indexOf(spin))
        assert (row, column, row_span, column_span) == (1, 3, 1, 1)


def test_cubic_display_interpolation_uses_factor_and_z_aspect_ratio():
    frame = np.arange(20, dtype=np.float32).reshape(4, 5)
    resized = interpolate_frame_for_display(frame, factor=3.0, aspect_ratio=2.0)
    assert resized.shape == (24, 15)
    assert resized.dtype == np.float32


def test_interpolation_aligns_image_overlays_and_picking_without_refitting(app):
    viewer = PalaSceneViewer()
    viewer.resize(500, 400)
    viewer.set_session(make_session())
    transform_before = viewer.transform()

    assert viewer.image_item.transformationMode() == Qt.FastTransformation

    viewer.set_interpolation(True, factor=5.0, aspect_ratio=2.0)

    assert viewer.image_item.pixmap().width() == 25
    assert viewer.image_item.pixmap().height() == 40
    assert viewer.image_item.transform().m11() == pytest.approx(0.2)
    assert viewer.image_item.transform().m22() == pytest.approx(0.2)
    assert viewer.image_item.transformationMode() == Qt.FastTransformation
    assert viewer.scene.sceneRect().width() == pytest.approx(5.0)
    assert viewer.scene.sceneRect().height() == pytest.approx(8.0)
    assert viewer._pala_to_display_xy(2.0, 2.0) == pytest.approx((1.5, 3.0))
    assert viewer._find_detection_near(1.5, 3.0) == 0
    assert viewer.transform().m11() == pytest.approx(transform_before.m11())
    assert viewer.transform().m22() == pytest.approx(transform_before.m22())

    pixmap_key = viewer.image_item.pixmap().cacheKey()
    viewer.set_selected_track(0)
    assert viewer.image_item.pixmap().cacheKey() == pixmap_key

    localization = next(
        item for item in viewer.overlay_items if item.data(0) == "localization"
    )
    center = localization.mapToScene(localization.boundingRect().center())
    assert (center.x(), center.y()) == pytest.approx((1.5, 3.0))

    viewer.set_interpolation(False, factor=5.0, aspect_ratio=2.0)
    assert viewer.image_item.pixmap().width() == 5
    assert viewer.image_item.pixmap().height() == 4
    assert viewer.scene.sceneRect().height() == pytest.approx(4.0)
    assert viewer._pala_to_display_xy(2.0, 2.0) == pytest.approx((1.5, 1.5))

    viewer.set_interpolation(True, factor=5.0, aspect_ratio=2.0)
    assert viewer.image_item.pixmap().cacheKey() == pixmap_key
