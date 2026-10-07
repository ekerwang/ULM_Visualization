"""Widget construction, layout, and menus of the main window.

Pure UI assembly: every handler these widgets connect to lives in the
behavior mixins (editing / session / tracklist / playback) or on
``MainWindow`` itself.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QStylePainter,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ulm_track_correction_gui.gui.display_panel import DisplayPanel
from ulm_track_correction_gui.gui.dataset_list_delegate import DatasetItemDelegate
from ulm_track_correction_gui.gui.add_point_panel import AddPointPanel
from ulm_track_correction_gui.gui.main_window_tracklist import (
    FILTER_OPTIONS,
    SORT_OPTIONS,
)
from ulm_track_correction_gui.gui.movie_export import (
    DEFAULT_WATERMARK_FONT_SIZE_PX,
    MAX_WATERMARK_FONT_SIZE_PX,
    MIN_WATERMARK_FONT_SIZE_PX,
)
from ulm_track_correction_gui.gui.track_list_delegate import TrackItemDelegate
from ulm_track_correction_gui.gui.viewer import PalaSceneViewer


SCOPE_FULL_STACK = "full_stack"
SCOPE_TRACK_RANGE = "track_range"
SCOPE_TRACK_PADDED = "track_padded"
SCOPE_ORDER = (SCOPE_FULL_STACK, SCOPE_TRACK_RANGE, SCOPE_TRACK_PADDED)
SCOPE_SHORTCUTS = {
    SCOPE_FULL_STACK: "Q",
    SCOPE_TRACK_RANGE: "W",
    SCOPE_TRACK_PADDED: "E",
}
LEFT_PANEL_TAB_LABELS = ("Datasets", "Tracks")

_CORRECTION_MODE_BUTTON_STYLESHEET = """
QPushButton[correctionModeButton="true"] {
    padding: 2px 6px;
    border: 1px solid palette(mid);
    border-radius: 5px;
    background-color: palette(button);
    color: palette(button-text);
}
QPushButton[correctionModeButton="true"]:hover,
QPushButton[correctionModeButton="true"]:focus {
    border-color: palette(highlight);
}
QPushButton[correctionModeButton="true"]:pressed {
    background-color: palette(midlight);
}
QPushButton[correctionModeButton="true"]:checked {
    background-color: palette(highlight);
    color: palette(highlighted-text);
    border-color: palette(highlight);
}
"""


def _native_shortcut(
    shortcut: QKeySequence.StandardKey | str,
) -> str:
    """Return the platform-native label used on visible command buttons."""

    return QKeySequence(shortcut).toString(QKeySequence.NativeText)


class FrameScopeSlider(QSlider):
    """Frame slider that can show proportional padding around a track.

    The slider value always remains an absolute 1-based frame number.  Only
    the painted rail changes: frames before/after the selected track use a
    dashed line while the track's inclusive start--end span stays solid.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self._track_start: int | None = None
        self._track_end: int | None = None
        self.setMinimumHeight(30)
        self.setStyleSheet(
            "QSlider::groove:horizontal {"
            " height: 4px;"
            " background: palette(mid);"
            " border-radius: 2px;"
            "}"
            "QSlider::sub-page:horizontal {"
            " background: palette(highlight);"
            " border-radius: 2px;"
            "}"
            "QSlider::add-page:horizontal {"
            " background: palette(mid);"
            " border-radius: 2px;"
            "}"
            "QSlider::handle:horizontal {"
            " width: 18px;"
            " margin: -7px 0;"
            " background: palette(highlight);"
            " border: 1px solid palette(highlighted-text);"
            " border-radius: 9px;"
            "}"
        )

    @property
    def track_span(self) -> tuple[int, int] | None:
        if self._track_start is None or self._track_end is None:
            return None
        return self._track_start, self._track_end

    def set_track_span(self, start: int | None, end: int | None) -> None:
        span = None if start is None or end is None else (int(start), int(end))
        current = self.track_span
        if span == current:
            return
        if span is None:
            self._track_start = None
            self._track_end = None
        else:
            self._track_start, self._track_end = span
        self.update()

    def segment_frame_counts(self) -> tuple[int, int, int]:
        """Return before/track/after frame counts for proportional painting."""

        if self.track_span is None:
            return 0, max(1, self.maximum() - self.minimum() + 1), 0
        start, end = self.track_span
        start = max(self.minimum(), min(start, self.maximum()))
        end = max(self.minimum(), min(end, self.maximum()))
        if end < start:
            return 0, max(1, self.maximum() - self.minimum() + 1), 0
        return start - self.minimum(), end - start + 1, self.maximum() - end

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.isEnabled() or self.track_span is None:
            return

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.CC_Slider,
            option,
            QStyle.SC_SliderGroove,
            self,
        )
        before, track, after = self.segment_frame_counts()
        total = before + track + after
        if total <= 0 or groove.width() <= 0:
            return

        left = float(groove.left())
        right = float(groove.right())
        span = max(1.0, right - left)
        track_left = left + span * before / total
        track_right = left + span * (before + track) / total
        y = float(groove.center().y())

        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Mask the native continuous rail, then redraw the three semantic
        # regions. Colors come from QPalette so dark/light desktop themes work.
        mask_pen = QPen(self.palette().window().color())
        mask_pen.setWidthF(max(5.0, float(groove.height())))
        mask_pen.setCapStyle(Qt.FlatCap)
        painter.setPen(mask_pen)
        painter.drawLine(round(left), round(y), round(right), round(y))

        padding_pen = QPen(self.palette().mid().color())
        padding_pen.setWidthF(3.0)
        padding_pen.setStyle(Qt.DashLine)
        padding_pen.setCapStyle(Qt.FlatCap)
        painter.setPen(padding_pen)
        if before:
            painter.drawLine(round(left), round(y), round(track_left), round(y))
        if after:
            painter.drawLine(round(track_right), round(y), round(right), round(y))

        track_pen = QPen(self.palette().highlight().color())
        track_pen.setWidthF(4.0)
        track_pen.setCapStyle(Qt.FlatCap)
        painter.setPen(track_pen)
        painter.drawLine(
            round(track_left),
            round(y),
            round(track_right),
            round(y),
        )

        # The custom rail was painted over the native control; redraw only the
        # handle so it remains crisp and follows the platform style.
        option.subControls = QStyle.SC_SliderHandle
        painter.drawComplexControl(QStyle.CC_Slider, option)


class WindowUiMixin:
    """Creates all widgets, the four-pane layout, and the menu bar."""

    def _build_widgets(self) -> None:
        self.dataset_header = QLabel()
        self.dataset_list = QListWidget()
        self.dataset_list.setUniformItemSizes(True)
        self.dataset_list.setItemDelegate(DatasetItemDelegate(self.dataset_list))
        self.dataset_list.currentItemChanged.connect(self.on_dataset_selected)
        self.dataset_list.itemDoubleClicked.connect(self.on_dataset_double_clicked)

        self.import_dataset_button = QPushButton("Add dataset")
        self.import_dataset_button.setToolTip(
            "Add one or more source MAT files to the dataset queue."
        )
        self.import_dataset_button.clicked.connect(self.import_dataset_list)
        self.remove_dataset_button = QPushButton("Remove")
        self.remove_dataset_button.clicked.connect(self.remove_selected_dataset)
        self.remove_all_datasets_button = QPushButton("Remove all")
        self.remove_all_datasets_button.setToolTip(
            "Clear the dataset queue without deleting any files."
        )
        self.remove_all_datasets_button.clicked.connect(self.remove_all_datasets)
        self.save_progress_button = QPushButton("Save progress")
        self.save_progress_button.setToolTip(
            "Save verified tracks to a corrected subfolder using the source filename."
        )
        self.save_progress_button.clicked.connect(self.save_dataset_progress)
        self.next_dataset_button = QPushButton("Next dataset")
        self.next_dataset_button.setToolTip(
            "Load the literal next row in the dataset list."
        )
        self.next_dataset_button.clicked.connect(self.load_next_dataset)

        self.viewer = PalaSceneViewer()
        self.viewer.detection_clicked.connect(self.on_detection_clicked)
        self.viewer.candidate_clicked.connect(self.on_candidate_clicked)
        self.viewer.manual_candidate_placed.connect(self.on_manual_candidate_placed)
        self.viewer.manual_detection_erase_requested.connect(
            self.on_manual_detection_erase_requested
        )
        self.viewer.track_picked.connect(self.on_track_picked)
        self.viewer.track_selection_cleared.connect(self.clear_track_selection)

        self.display_panel = DisplayPanel(self.viewer.styles)
        self.display_panel.style_changed.connect(self.viewer.update_style)
        self.display_panel.layer_order_changed.connect(self.viewer.set_layer_order)
        self.viewer.set_layer_order(self.display_panel.layer_order())
        self.display_panel.display_settings_changed.connect(
            self.viewer.set_display_settings
        )
        self.display_panel.interpolation_changed.connect(
            self.viewer.set_interpolation
        )
        self.display_panel.dataset_plot_map_changed.connect(
            self.viewer.set_dataset_plot_map
        )
        self.display_panel.dataset_plot_density_clim_changed.connect(
            self.viewer.set_dataset_plot_density_clim
        )
        self.display_panel.dataset_plot_gamma_changed.connect(
            self.viewer.set_dataset_plot_gamma
        )
        self.display_panel.dataset_plot_clim_changed.connect(
            self.viewer.set_dataset_plot_velocity_clim
        )
        self.viewer.dataset_plot_status_changed.connect(
            self.display_panel.set_dataset_plot_status
        )
        self.display_panel.multiple_dataset_import_requested.connect(
            self.import_multiple_dataset_accumulation
        )
        self.display_panel.multiple_dataset_plot_map_changed.connect(
            self.viewer.set_multiple_dataset_plot_map
        )
        self.display_panel.multiple_dataset_plot_density_clim_changed.connect(
            self.viewer.set_multiple_dataset_plot_density_clim
        )
        self.display_panel.multiple_dataset_plot_gamma_changed.connect(
            self.viewer.set_multiple_dataset_plot_gamma
        )
        self.display_panel.multiple_dataset_plot_clim_changed.connect(
            self.viewer.set_multiple_dataset_plot_velocity_clim
        )
        self.viewer.multiple_dataset_plot_status_changed.connect(
            self.display_panel.set_multiple_dataset_plot_status
        )
        self.display_panel.track_plot_map_changed.connect(
            self.viewer.set_track_plot_map
        )
        self.display_panel.track_plot_clim_changed.connect(
            self.viewer.set_track_plot_velocity_clim
        )
        self.viewer.track_plot_status_changed.connect(
            self.display_panel.set_track_plot_status
        )

        # Inspect is the resting mode. Assign, Remove, and Add Point remain
        # active until the user selects another mode or toggles them off.
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, label, tooltip in (
            (
                "inspect",
                "Inspect (Esc)",
                "Click an assigned point to select its track; click empty image "
                "space to clear; drag to pan (Esc)",
            ),
            (
                "assign",
                "Assign (A)",
                "Click detections to assign them to the selected track; click "
                "again or press A/Esc to return to Inspect",
            ),
            (
                "remove",
                "Remove (R)",
                "Click detections to remove their assignments; click again or "
                "press R/Esc to return to Inspect",
            ),
            (
                "add_point",
                "Add point (M)",
                "Automatically localize each current frame, then click candidates; "
                "the inline eraser removes unassigned added points; click again "
                "or press M/Esc to return to Inspect",
            ),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("correctionModeButton", True)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setToolTip(tooltip)
            button.setStyleSheet(_CORRECTION_MODE_BUTTON_STYLESHEET)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            button.pressed.connect(
                lambda m=mode: self.remember_correction_mode_button_press(m)
            )
            button.toggled.connect(
                lambda checked, m=mode: checked and self.set_correction_mode(m)
            )
            button.clicked.connect(
                lambda checked=False, m=mode: self.finish_correction_mode_button_click(m)
            )
        self.mode_buttons["inspect"].setChecked(True)
        self.add_point_button = self.mode_buttons["add_point"]
        self.add_point_panel = AddPointPanel()
        self.add_point_panel.parameters_changed.connect(
            self.schedule_add_point_localization
        )
        self.add_point_panel.style_changed.connect(
            self.on_candidate_style_changed
        )
        self.add_point_panel.eraser_toggled.connect(
            self.on_add_point_eraser_toggled
        )

        self.merge_button = QPushButton("Merge (J)")
        self.merge_button.setToolTip(
            "Merge two tracks by ID; the absorbed track empties while every "
            "Track ID stays unchanged in saved results (J)."
        )
        self.merge_button.clicked.connect(self.open_merge_dialog)

        self.initialize_track_button = QPushButton("New Track (N)")
        self.initialize_track_button.setToolTip(
            "Create and select an active empty track, then use Assign or Add Point (N)."
        )
        self.initialize_track_button.clicked.connect(self.initialize_new_track)

        self.split_button = QPushButton("Split (S)")
        self.split_button.setToolTip(
            "Replace the selected track with two new flagged tracks (S)."
        )
        self.split_button.clicked.connect(self.open_split_dialog)

        self.delete_track_button = QPushButton("Delete Track (D)")
        self.delete_track_button.setToolTip(
            "Retire the selected track; localization detections are preserved (D)."
        )
        self.delete_track_button.clicked.connect(self.delete_selected_track)

        self.undo_button = QPushButton(
            f"Undo ({_native_shortcut(QKeySequence.Undo)})"
        )
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton(
            f"Redo ({_native_shortcut(QKeySequence.Redo)})"
        )
        self.redo_button.clicked.connect(self.redo)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(SORT_OPTIONS)
        self.sort_combo.currentIndexChanged.connect(lambda _: self.populate_track_list())
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTER_OPTIONS)
        self.filter_combo.currentIndexChanged.connect(lambda _: self.populate_track_list())

        self.track_list = QListWidget()
        self.track_list.setUniformItemSizes(True)
        self.track_list.setItemDelegate(TrackItemDelegate(self.track_list))
        self.track_list.currentItemChanged.connect(self.on_track_selected)
        self.track_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.track_list.customContextMenuRequested.connect(
            self._show_track_context_menu
        )

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setMinimumSize(84, 30)
        self.frame_spin.setAccessibleName("Current frame")
        self.frame_spin.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.frame_spin.valueChanged.connect(self.on_frame_spin_changed)

        self.frame_caption_label = QLabel("Frame")
        self.frame_caption_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.scope_context_label = QLabel("No stack loaded")
        self.scope_context_label.setAccessibleName("Current frame context")
        self.scope_context_label.setStyleSheet(
            "font-size: 13px; color: palette(window-text);"
        )

        self.playback_scope = SCOPE_FULL_STACK
        self.scope_group = QButtonGroup(self)
        self.scope_buttons: dict[str, QPushButton] = {}
        self.scope_sliders: dict[str, FrameScopeSlider] = {}
        self.scope_labels: dict[str, str] = {}
        self.scope_stack = QStackedWidget()
        self.scope_stack.setObjectName("frameScopeStack")
        self.scope_stack.setMinimumHeight(32)
        scope_specs = (
            (
                SCOPE_FULL_STACK,
                "Full stack",
                "Navigate every frame in the loaded stack.",
            ),
            (
                SCOPE_TRACK_RANGE,
                "Track range",
                "Expand the selected track's start-to-end frame span across "
                "the full slider width; gap frames remain navigable.",
            ),
            (
                SCOPE_TRACK_PADDED,
                "Track ±N",
                "Navigate the selected track plus N surrounding frames; "
                "dashed rail segments are outside the track range.",
            ),
        )
        for scope, label, tooltip in scope_specs:
            shortcut = SCOPE_SHORTCUTS[scope]
            native_shortcut = _native_shortcut(shortcut)
            button = QPushButton(f"{label} ({native_shortcut})")
            button.setCheckable(True)
            button.setProperty("frameScope", True)
            button.setMinimumHeight(30)
            button.setAccessibleName(f"{label} playback scope")
            button.setToolTip(f"{tooltip} ({native_shortcut})")
            button.setStyleSheet(
                "QPushButton[frameScope=\"true\"] {"
                " padding: 4px 10px;"
                " border: 1px solid palette(mid);"
                " border-radius: 5px;"
                " background: palette(button);"
                "}"
                "QPushButton[frameScope=\"true\"]:checked {"
                " background: palette(highlight);"
                " color: palette(highlighted-text);"
                " border-color: palette(highlight);"
                " font-weight: 600;"
                "}"
                "QPushButton[frameScope=\"true\"]:focus {"
                " border: 2px solid palette(highlight);"
                "}"
                "QPushButton[frameScope=\"true\"]:disabled {"
                " background: palette(window);"
                " color: palette(mid);"
                " border-color: palette(mid);"
                "}"
            )
            self.scope_group.addButton(button)
            self.scope_buttons[scope] = button
            self.scope_labels[scope] = label

            slider = FrameScopeSlider()
            slider.setRange(1, 1)
            slider.setToolTip(tooltip)
            slider.setAccessibleName(f"{label} frame timeline")
            slider.valueChanged.connect(
                lambda value, s=scope: self.on_scope_slider_changed(s, value)
            )
            self.scope_sliders[scope] = slider
            self.scope_stack.addWidget(slider)

            button.toggled.connect(
                lambda checked, s=scope: checked and self.set_playback_scope(s)
            )
        full_scope_button = self.scope_buttons[SCOPE_FULL_STACK]
        full_scope_button.blockSignals(True)
        full_scope_button.setChecked(True)
        full_scope_button.blockSignals(False)
        # Compatibility alias for code that asks for the currently active
        # frame slider. set_playback_scope() keeps it up to date.
        self.frame_slider = self.scope_sliders[SCOPE_FULL_STACK]
        self.scope_stack.setCurrentWidget(self.frame_slider)

        self.play_button = QPushButton("Play (Space)")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self.on_play_toggled)
        self.export_movie_button = QPushButton("Export Movie")
        self.export_movie_button.setToolTip(
            "Export the current playback scope and visible viewer layers as MP4."
        )
        self.export_movie_button.clicked.connect(self.export_movie)
        self.watermark_preview_checkbox = QCheckBox("Preview watermark")
        self.watermark_preview_checkbox.setToolTip(
            "Show the export watermark at its video position in the viewer."
        )
        self.watermark_preview_checkbox.toggled.connect(
            self.refresh_movie_watermark_preview
        )
        self.watermark_font_size_label = QLabel("Font")
        self.watermark_font_size_spin = QSpinBox()
        self.watermark_font_size_spin.setRange(
            MIN_WATERMARK_FONT_SIZE_PX,
            MAX_WATERMARK_FONT_SIZE_PX,
        )
        self.watermark_font_size_spin.setValue(DEFAULT_WATERMARK_FONT_SIZE_PX)
        self.watermark_font_size_spin.setSuffix(" px")
        self.watermark_font_size_spin.setAccessibleName("Watermark font size")
        self.watermark_font_size_spin.setToolTip(
            "Font size used by both the viewer preview and exported MP4."
        )
        self.watermark_font_size_spin.valueChanged.connect(
            self.refresh_movie_watermark_preview
        )
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 100)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._apply_fps)
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(1, 500)
        self.pad_spin.setValue(10)
        self.pad_spin.setPrefix("±")
        self.pad_spin.setAccessibleName("Track padding frames")
        self.pad_spin.setToolTip("Frames shown before and after the selected track")
        self.pad_spin.valueChanged.connect(self.on_track_padding_changed)
        self.padding_controls = QWidget()
        padding_layout = QHBoxLayout(self.padding_controls)
        padding_layout.setContentsMargins(0, 0, 0, 0)
        padding_layout.setSpacing(5)
        padding_label = QLabel("Padding")
        padding_label.setStyleSheet("font-size: 12px;")
        padding_unit_label = QLabel("frames")
        padding_unit_label.setStyleSheet("font-size: 12px;")
        padding_layout.addWidget(padding_label)
        padding_layout.addWidget(self.pad_spin)
        padding_layout.addWidget(padding_unit_label)
        self.padding_controls.setVisible(False)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        self.mark_verified_button = QPushButton("Mark Verified (V)")
        self.mark_verified_button.clicked.connect(self.mark_selected_verified)
        self.flag_button = QPushButton("Flag (X)")
        self.flag_button.clicked.connect(self.flag_selected)
        self.history_button = QPushButton("Edit History (H)")
        self.history_button.setToolTip(
            "Show every edit recorded for the selected track; double-click an "
            "entry to jump to its frame."
        )
        self.history_button.clicked.connect(
            lambda checked=False: self.open_track_history()
        )
        self.status_label = QLabel("Open a MAT file to begin.")

    def _build_layout(self) -> None:
        dataset_actions = QGridLayout()
        dataset_actions.setContentsMargins(0, 0, 0, 0)
        dataset_actions.setHorizontalSpacing(4)
        dataset_actions.setVerticalSpacing(4)
        dataset_actions.addWidget(self.import_dataset_button, 0, 0, 1, 2)
        dataset_actions.addWidget(self.remove_dataset_button, 1, 0)
        dataset_actions.addWidget(self.remove_all_datasets_button, 1, 1)
        dataset_actions.addWidget(self.save_progress_button, 2, 0)
        dataset_actions.addWidget(self.next_dataset_button, 2, 1)

        self.collapse_dataset_button = QToolButton()
        self.collapse_dataset_button.setAutoRaise(True)
        self.collapse_dataset_button.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowLeft)
        )
        self.collapse_dataset_button.setToolTip(
            "Collapse the Dataset panel; reopen it from the left tab rail."
        )
        self.collapse_dataset_button.setAccessibleName("Collapse Dataset panel")
        self.collapse_dataset_button.clicked.connect(
            lambda checked=False: self._collapse_left_panel(0)
        )

        dataset_header_row = QHBoxLayout()
        dataset_header_row.setContentsMargins(0, 0, 0, 0)
        dataset_header_row.addWidget(self.dataset_header, stretch=1)
        dataset_header_row.addWidget(self.collapse_dataset_button)

        datasets = QVBoxLayout()
        datasets.addLayout(dataset_header_row)
        datasets.addWidget(self.dataset_list, stretch=1)
        datasets.addLayout(dataset_actions)

        list_controls = QHBoxLayout()
        list_controls.addWidget(self.sort_combo, stretch=1)
        list_controls.addWidget(self.filter_combo, stretch=1)

        self.track_header = QLabel("Tracks (↑/↓)")
        self.track_count_label = QLabel("0 shown")
        self.track_count_label.setAccessibleName("Filtered track count")
        self.track_count_label.setToolTip(
            "Number of active tracks matching the current filter."
        )
        self.track_count_label.setStyleSheet(
            "color: palette(window-text); font-weight: 600;"
        )
        self.collapse_track_button = QToolButton()
        self.collapse_track_button.setAutoRaise(True)
        self.collapse_track_button.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowLeft)
        )
        self.collapse_track_button.setToolTip(
            "Collapse the Tracks panel; reopen it from the left tab rail."
        )
        self.collapse_track_button.setAccessibleName("Collapse Tracks panel")
        self.collapse_track_button.clicked.connect(
            lambda checked=False: self._collapse_left_panel(1)
        )

        track_header_row = QHBoxLayout()
        track_header_row.setContentsMargins(0, 0, 0, 0)
        track_header_row.addWidget(self.track_header, stretch=1)
        track_header_row.addWidget(self.track_count_label)
        track_header_row.addWidget(self.collapse_track_button)

        tracks = QVBoxLayout()
        tracks.addLayout(track_header_row)
        tracks.addLayout(list_controls)
        tracks.addWidget(self.track_list, stretch=1)
        tracks.addWidget(self.mark_verified_button)
        tracks.addWidget(self.flag_button)
        tracks.addWidget(self.history_button)

        self.navigation_box = QGroupBox("Frame navigation (←/→)")
        navigation_layout = QVBoxLayout(self.navigation_box)
        navigation_layout.setContentsMargins(9, 7, 9, 7)
        navigation_layout.setSpacing(5)

        scope_switch_row = QHBoxLayout()
        scope_switch_row.setContentsMargins(0, 0, 0, 0)
        scope_switch_row.setSpacing(5)
        for scope in SCOPE_ORDER:
            scope_switch_row.addWidget(self.scope_buttons[scope], stretch=1)
        navigation_layout.addLayout(scope_switch_row)

        frame_readout_row = QHBoxLayout()
        frame_readout_row.setContentsMargins(0, 0, 0, 0)
        frame_readout_row.setSpacing(6)
        frame_readout_row.addWidget(self.frame_caption_label)
        frame_readout_row.addWidget(self.frame_spin)
        frame_readout_row.addWidget(self.scope_context_label)
        frame_readout_row.addStretch(1)
        frame_readout_row.addWidget(self.padding_controls)
        navigation_layout.addLayout(frame_readout_row)
        navigation_layout.addWidget(self.scope_stack)

        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(0, 0, 0, 0)
        transport_row.addWidget(self.play_button)
        transport_row.addSpacing(6)
        transport_row.addWidget(self.export_movie_button)
        transport_row.addSpacing(6)
        transport_row.addWidget(self.watermark_preview_checkbox)
        transport_row.addWidget(self.watermark_font_size_label)
        transport_row.addWidget(self.watermark_font_size_spin)
        transport_row.addSpacing(6)
        transport_row.addWidget(self.fps_spin)
        transport_row.addStretch(1)
        navigation_layout.addLayout(transport_row)

        center = QVBoxLayout()
        center.addWidget(self.viewer, stretch=1)
        center.addWidget(self.navigation_box)
        center.addWidget(self.status_label)

        mode_box = QGroupBox("Tools")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setContentsMargins(7, 5, 7, 7)
        mode_layout.setSpacing(6)

        # Escape the ampersand so Qt paints it instead of treating "B" as a
        # hidden mnemonic marker in the group title.
        self.view_tools_box = QGroupBox("View && Basic")
        view_layout = QHBoxLayout(self.view_tools_box)
        view_layout.setContentsMargins(6, 4, 6, 5)
        view_layout.setSpacing(3)
        view_layout.addWidget(self.mode_buttons["inspect"])
        view_layout.addWidget(self.undo_button)
        view_layout.addWidget(self.redo_button)
        mode_layout.addWidget(self.view_tools_box)

        self.track_level_tools_box = QGroupBox("Track Level")
        track_level_layout = QGridLayout(self.track_level_tools_box)
        track_level_layout.setContentsMargins(6, 4, 6, 5)
        track_level_layout.setHorizontalSpacing(3)
        track_level_layout.setVerticalSpacing(3)
        track_level_layout.addWidget(self.initialize_track_button, 0, 0)
        track_level_layout.addWidget(self.delete_track_button, 0, 1)
        track_level_layout.addWidget(self.merge_button, 1, 0)
        track_level_layout.addWidget(self.split_button, 1, 1)
        mode_layout.addWidget(self.track_level_tools_box)

        self.track_internal_tools_box = QGroupBox("Track-internal Level")
        internal_layout = QGridLayout(self.track_internal_tools_box)
        internal_layout.setContentsMargins(6, 4, 6, 5)
        internal_layout.setVerticalSpacing(3)
        internal_layout.addWidget(self.mode_buttons["assign"], 0, 0)
        internal_layout.addWidget(self.mode_buttons["remove"], 1, 0)
        internal_layout.addWidget(self.add_point_button, 2, 0)
        internal_layout.addWidget(self.add_point_panel, 3, 0)
        mode_layout.addWidget(self.track_internal_tools_box)

        right_scroll = QScrollArea()
        right_scroll.setWidget(self.display_panel)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QScrollArea.NoFrame)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(mode_box)
        right.addWidget(right_scroll, stretch=1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(270)

        self.dataset_panel = QWidget()
        self.dataset_panel.setLayout(datasets)
        self.dataset_panel.setMinimumWidth(230)

        self.track_panel = QWidget()
        self.track_panel.setLayout(tracks)
        self.track_panel.setMinimumWidth(230)

        center_widget = QWidget()
        center_widget.setLayout(center)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.dataset_panel)
        self.main_splitter.addWidget(self.track_panel)
        self.main_splitter.addWidget(center_widget)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setStretchFactor(2, 1)
        self.main_splitter.setStretchFactor(3, 0)
        self.main_splitter.setCollapsible(2, False)
        self.main_splitter.setSizes([260, 300, 650, 330])
        self._left_panel_restore_sizes = {0: 260, 1: 300}
        self.main_splitter.splitterMoved.connect(
            self._remember_left_panel_sizes
        )

        self.left_panel_tabs = QTabBar()
        self.left_panel_tabs.setObjectName("leftPanelTabs")
        self.left_panel_tabs.setShape(QTabBar.RoundedWest)
        self.left_panel_tabs.setDocumentMode(True)
        self.left_panel_tabs.setExpanding(False)
        self.left_panel_tabs.setUsesScrollButtons(False)
        for label in LEFT_PANEL_TAB_LABELS:
            self.left_panel_tabs.addTab(label)
        self.left_panel_tabs.setAccessibleName("Left panel tabs")
        self.left_panel_tabs.setFixedWidth(34)
        self.left_panel_tabs.setStyleSheet(
            "QTabBar::tab {"
            " min-height: 72px;"
            " padding: 5px 4px;"
            " border: none;"
            " border-right: 1px solid palette(mid);"
            " border-bottom: 1px solid palette(midlight);"
            " background-color: palette(window);"
            "}"
            "QTabBar::tab:hover { background-color: palette(midlight); }"
        )
        self.left_panel_tabs.tabBarClicked.connect(self._show_left_panel)
        self._sync_left_panel_tab_states()

        left_panel_rail = QWidget()
        left_panel_rail.setObjectName("leftPanelRail")
        left_panel_rail.setFixedWidth(34)
        left_panel_rail_layout = QVBoxLayout(left_panel_rail)
        left_panel_rail_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_rail_layout.setSpacing(0)
        left_panel_rail_layout.addWidget(self.left_panel_tabs, alignment=Qt.AlignTop)
        left_panel_rail_layout.addStretch(1)

        central_shell = QWidget()
        central_shell_layout = QHBoxLayout(central_shell)
        central_shell_layout.setContentsMargins(0, 0, 0, 0)
        central_shell_layout.setSpacing(0)
        central_shell_layout.addWidget(left_panel_rail)
        central_shell_layout.addWidget(self.main_splitter, stretch=1)
        self.setCentralWidget(central_shell)

    def _remember_left_panel_sizes(self, *_splitter_position: int) -> None:
        """Remember useful widths and sync rail state after splitter resizing."""

        sizes = self.main_splitter.sizes()
        for index in (0, 1):
            if sizes[index] >= self.main_splitter.widget(index).minimumWidth():
                self._left_panel_restore_sizes[index] = sizes[index]
        self._sync_left_panel_tab_states()

    def _sync_left_panel_tab_states(self) -> None:
        """Show both left-pane states without implying single-tab selection."""

        sizes = self.main_splitter.sizes()
        palette = self.left_panel_tabs.palette()
        for index, label in enumerate(LEFT_PANEL_TAB_LABELS):
            expanded = sizes[index] > 0
            marker = "●" if expanded else "○"
            state = "expanded" if expanded else "collapsed"
            self.left_panel_tabs.setTabText(index, f"{marker} {label}")
            self.left_panel_tabs.setTabData(index, state)
            self.left_panel_tabs.setTabTextColor(
                index,
                (
                    palette.highlight().color()
                    if expanded
                    else palette.buttonText().color()
                ),
            )
            action = (
                "is open; use its header arrow to collapse it"
                if expanded
                else "is collapsed; click to open it"
            )
            self.left_panel_tabs.setTabToolTip(index, f"{label} panel {action}.")

    def _collapse_left_panel(self, index: int) -> None:
        """Collapse one left pane while leaving its vertical restore tab visible."""

        sizes = self.main_splitter.sizes()
        if sizes[index] <= 0:
            return
        self._left_panel_restore_sizes[index] = sizes[index]
        sizes[index] = 0
        self.main_splitter.setSizes(sizes)
        self.left_panel_tabs.setCurrentIndex(index)
        self._sync_left_panel_tab_states()

    def _show_left_panel(self, index: int) -> None:
        """Restore a collapsed Dataset or Tracks pane from the vertical rail."""

        sizes = self.main_splitter.sizes()
        if sizes[index] > 0:
            return
        panel = self.main_splitter.widget(index)
        sizes[index] = max(
            panel.minimumWidth(),
            self._left_panel_restore_sizes[index],
        )
        self.main_splitter.setSizes(sizes)
        self._sync_left_panel_tab_states()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open MAT...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_mat)
        file_menu.addAction(open_action)

        import_dataset_action = QAction("Add Dataset...", self)
        import_dataset_action.triggered.connect(self.import_dataset_list)
        file_menu.addAction(import_dataset_action)

        self.load_checkpoint_action = QAction(
            "Load Accumulation Checkpoint...",
            self,
        )
        self.load_checkpoint_action.triggered.connect(
            self.load_accumulation_checkpoint_manually
        )
        file_menu.addAction(self.load_checkpoint_action)

        save_action = QAction("Save Correction As...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_mat_as)
        file_menu.addAction(save_action)

        edit_menu = self.menuBar().addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()
        for mode, key in (
            ("inspect", "Esc"),
            ("assign", "A"),
            ("remove", "R"),
        ):
            action = QAction(f"{mode.capitalize()} mode", self)
            action.setShortcut(key)
            action.setAutoRepeat(False)
            action.triggered.connect(
                lambda checked=False, m=mode: self.toggle_correction_mode(m)
            )
            edit_menu.addAction(action)

        self.add_point_action = QAction("Add Point mode", self)
        self.add_point_action.setShortcut("M")
        self.add_point_action.setAutoRepeat(False)
        self.add_point_action.triggered.connect(
            lambda: self.toggle_correction_mode("add_point")
        )
        edit_menu.addAction(self.add_point_action)

        merge_action = QAction("Merge Tracks...", self)
        merge_action.setShortcut("J")
        merge_action.triggered.connect(self.open_merge_dialog)
        edit_menu.addAction(merge_action)

        edit_menu.addSeparator()
        self.initialize_track_action = QAction("Initialize New Track...", self)
        self.initialize_track_action.setShortcut("N")
        self.initialize_track_action.triggered.connect(self.initialize_new_track)
        edit_menu.addAction(self.initialize_track_action)

        self.split_track_action = QAction("Split Track...", self)
        self.split_track_action.setShortcut("S")
        self.split_track_action.triggered.connect(self.open_split_dialog)
        edit_menu.addAction(self.split_track_action)

        self.delete_track_action = QAction("Delete Track...", self)
        self.delete_track_action.setShortcut("D")
        self.delete_track_action.triggered.connect(self.delete_selected_track)
        edit_menu.addAction(self.delete_track_action)

        view_menu = self.menuBar().addMenu("View")

        scope_menu = view_menu.addMenu("Playback Scope")
        self.scope_action_group = QActionGroup(self)
        self.scope_action_group.setExclusive(True)
        self.scope_actions: dict[str, QAction] = {}
        for scope in SCOPE_ORDER:
            action = QAction(self.scope_labels[scope], self)
            action.setCheckable(True)
            action.setShortcut(QKeySequence(SCOPE_SHORTCUTS[scope]))
            action.setShortcutContext(Qt.WindowShortcut)
            action.triggered.connect(
                lambda checked=False, s=scope: checked and self.set_playback_scope(s)
            )
            self.scope_action_group.addAction(action)
            self.scope_actions[scope] = action
            scope_menu.addAction(action)
        self.scope_actions[SCOPE_FULL_STACK].setChecked(True)
        view_menu.addSeparator()

        self.show_lines_action = QAction("Toggle All Track Lines", self)
        self.show_lines_action.triggered.connect(self.display_panel.toggle_all_lines)
        view_menu.addAction(self.show_lines_action)

        self.interpolate_action = QAction("Interpolate Dataset", self)
        self.interpolate_action.setShortcut("I")
        self.interpolate_action.triggered.connect(
            self.display_panel.interpolate_button.toggle
        )
        view_menu.addAction(self.interpolate_action)

        toggle_overlays_action = QAction("Keep Images Only / Restore Layers", self)
        toggle_overlays_action.setShortcut("T")
        toggle_overlays_action.triggered.connect(self.display_panel.toggle_overlays)
        view_menu.addAction(toggle_overlays_action)

        fit_action = QAction("Fit View", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.viewer.fit_view)
        view_menu.addAction(fit_action)

        play_action = QAction("Play/Pause", self)
        play_action.setShortcut(Qt.Key_Space)
        play_action.triggered.connect(self.play_button.toggle)
        view_menu.addAction(play_action)

        track_menu = self.menuBar().addMenu("Track")

        verify_action = QAction("Mark Verified", self)
        verify_action.setShortcut("V")
        verify_action.triggered.connect(self.mark_selected_verified)
        track_menu.addAction(verify_action)

        flag_action = QAction("Flag", self)
        flag_action.setShortcut("X")
        flag_action.triggered.connect(self.flag_selected)
        track_menu.addAction(flag_action)

        history_action = QAction("Edit History...", self)
        history_action.setShortcut("H")
        history_action.triggered.connect(
            lambda checked=False: self.open_track_history()
        )
        track_menu.addAction(history_action)
