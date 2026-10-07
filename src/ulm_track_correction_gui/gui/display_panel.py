"""Right-side panel with compact, collapsible per-layer visual controls.

The numbered rows are fixed foreground-to-background slots. Their clickable
number docks never move; only the layer card occupying a slot can be dragged.
Layer visibility and parameter expansion are independent states. A master
toggle hides every non-image layer and restores the previous visible set.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    DENSITY_TRACK_MAPS,
    FULL_TRACK_MAPS,
    HARD_DENSITY,
    TRACK_MAP_LABELS,
    VELOCITY_TRACK_MAPS,
)
from ulm_track_correction_gui.gui.layer_styles import (
    LayerStyle,
)
from ulm_track_correction_gui.gui.dynamic_range_controls import (
    create_dynamic_range_panel,
    current_display_settings,
)
from ulm_track_correction_gui.gui.layer_order import DEFAULT_LAYER_ORDER, LAYER_LABELS
from ulm_track_correction_gui.gui.track_plot_rendering import (
    DEFAULT_ACCUMULATION_CLIM,
    DEFAULT_ACCUMULATION_GAMMA,
)

# The master visibility action intentionally excludes Ultrasound Movie, preserving the
# earlier "hide overlays" behavior even though cards can now be reordered.
OVERLAY_LAYERS = tuple(key for key in DEFAULT_LAYER_ORDER if key != "image")

_STATE_BUTTON_STYLESHEET = """
QPushButton[stateButton="true"] {
    min-height: 20px;
    padding: 2px 0;
    border: 1px solid palette(mid);
    border-radius: 3px;
    background-color: palette(button);
    color: palette(button-text);
}
QPushButton[stateButton="true"]:hover {
    border-color: palette(highlight);
}
QPushButton[stateButton="true"]:pressed {
    background-color: palette(midlight);
}
QPushButton[stateButton="true"]:checked {
    border-color: #4aae43;
    background-color: #80e36f;
    color: #10210e;
}
QPushButton[stateButton="true"]:checked:hover {
    background-color: #8beb79;
}
QPushButton[stateButton="true"]:checked:pressed {
    background-color: #70d65e;
}
"""

_LAYER_SECTION_STYLESHEET = """
QGroupBox[displaySection="true"] {
    border: 1px solid palette(mid);
    border-radius: 7px;
    margin: 0;
    padding: 0;
    background-color: palette(window);
}
QGroupBox[displaySection="true"][dragging="true"] {
    border: 2px solid palette(highlight);
    background-color: palette(midlight);
}
QToolButton[layerHeader="true"] {
    min-height: 30px;
    padding: 2px 2px;
    border: 0;
    border-radius: 4px;
    background: transparent;
    font-size: 13px;
    font-weight: 650;
    text-align: left;
}
QToolButton[layerHeader="true"]:hover {
    background-color: palette(midlight);
}
"""

_LAYER_SLOT_STYLESHEET = """
QFrame[layerSlot="true"] {
    border: 0;
    background: transparent;
}
QFrame[layerSlotControl="true"] {
    border: 1px solid palette(mid);
    border-radius: 6px;
    background-color: palette(button);
}
QFrame[layerSlotControl="true"][layerVisible="true"] {
    border-color: palette(highlight);
}
QFrame[layerSlotControl="true"][dropTarget="true"] {
    border: 2px solid palette(highlight);
    background-color: palette(midlight);
}
QToolButton[layerSlotBadge="true"] {
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: palette(mid);
    font-size: 12px;
    font-weight: 700;
}
QToolButton[layerSlotBadge="true"]:hover {
    background-color: palette(midlight);
}
QToolButton[layerSlotBadge="true"][layerVisible="true"] {
    color: palette(highlight);
    font-weight: 800;
}
"""


class LayerSlot(QFrame):
    """Fixed slot control dock beside one visually independent movable card."""

    def __init__(self, slot_number: int) -> None:
        super().__init__()
        self.slot_number = int(slot_number)
        self.group: CollapsibleLayerGroup | None = None
        self.setProperty("layerSlot", True)
        self.setProperty("layerVisible", False)
        self.setStyleSheet(_LAYER_SLOT_STYLESHEET)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.slot_control = QFrame()
        self.slot_control.setProperty("layerSlotControl", True)
        self.slot_control.setProperty("layerVisible", False)
        self.slot_control.setProperty("dropTarget", False)
        self.slot_control.setFixedWidth(30)
        self.slot_control.setMinimumHeight(36)
        self.slot_control.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slot_control.setAccessibleName(f"Layer slot {self.slot_number} controls")

        self.shortcut_badge = QToolButton()
        self.shortcut_badge.setText(f"({self.slot_number})")
        self.shortcut_badge.setCheckable(True)
        self.shortcut_badge.setAutoRaise(True)
        self.shortcut_badge.setProperty("layerSlotBadge", True)
        self.shortcut_badge.setProperty("layerVisible", False)
        self.shortcut_badge.setFixedHeight(30)
        self.shortcut_badge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.shortcut_badge.setCursor(Qt.PointingHandCursor)
        self.shortcut_badge.setAccessibleName(f"Layer slot {self.slot_number}")
        self.shortcut_badge.clicked.connect(self._set_group_visibility)
        # Compatibility name for callers that toggle slot visibility directly.
        self.visibility_button = self.shortcut_badge

        control_layout = QVBoxLayout(self.slot_control)
        control_layout.setContentsMargins(3, 3, 3, 3)
        control_layout.setSpacing(0)
        control_layout.addWidget(self.shortcut_badge)
        control_layout.addStretch(1)

        self._card_layout = QVBoxLayout()
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.slot_control)
        layout.addLayout(self._card_layout, 1)

    def take_group(self) -> CollapsibleLayerGroup | None:
        group = self.group
        if group is not None:
            self._card_layout.removeWidget(group)
        self.group = None
        return group

    def set_group(self, group: CollapsibleLayerGroup) -> None:
        if self.group is group:
            self.sync()
            return
        self.take_group()
        self.group = group
        self._card_layout.addWidget(group)
        self.sync()

    def sync(self) -> None:
        group = self.group
        checked = bool(group is not None and group.isChecked())
        self.shortcut_badge.blockSignals(True)
        self.shortcut_badge.setChecked(checked)
        self.shortcut_badge.blockSignals(False)
        self.shortcut_badge.setEnabled(
            bool(group is not None and group.isVisibilityControlEnabled())
        )
        if group is not None:
            title = group.title()
            self.shortcut_badge.setAccessibleName(
                f"Toggle {title} in layer slot {self.slot_number}"
            )
            self.shortcut_badge.setToolTip(
                f"({self.slot_number}) Toggle {title} visibility"
            )
        self.setProperty("layerVisible", checked)
        self.slot_control.setProperty("layerVisible", checked)
        self.shortcut_badge.setProperty("layerVisible", checked)
        self.slot_control.style().unpolish(self.slot_control)
        self.slot_control.style().polish(self.slot_control)
        self.slot_control.update()
        self.shortcut_badge.style().unpolish(self.shortcut_badge)
        self.shortcut_badge.style().polish(self.shortcut_badge)
        self.shortcut_badge.update()
        self.update()

    def refresh_minimum_height(self) -> None:
        """Follow an expanded card's wrapping-dependent minimum height."""

        required_height = 0 if self.group is None else self.group.minimumHeight()
        if self.minimumHeight() != required_height:
            self.setMinimumHeight(required_height)
            self.updateGeometry()

    def set_drop_target(self, active: bool) -> None:
        """Highlight only the fixed dock that will receive the dragged card."""

        self.slot_control.setProperty("dropTarget", bool(active))
        self.slot_control.style().unpolish(self.slot_control)
        self.slot_control.style().polish(self.slot_control)
        self.slot_control.update()

    def _set_group_visibility(self, checked: bool) -> None:
        if self.group is not None:
            self.group.setChecked(bool(checked))
        self.sync()


class LayerHeaderButton(QToolButton):
    """Clickable disclosure title that also initiates card reordering."""

    drag_started = Signal()
    drag_moved = Signal(QPoint)
    drag_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._press_pos: QPoint | None = None
        self._dragging = False
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_pos is None or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        if not self._dragging and (
            event.position().toPoint() - self._press_pos
        ).manhattanLength() >= QApplication.startDragDistance():
            self._dragging = True
            self.setDown(False)
            self.setCursor(Qt.ClosedHandCursor)
            self.drag_started.emit()
        if self._dragging:
            self.drag_moved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self.setDown(False)
            self.setCursor(Qt.OpenHandCursor)
            self._dragging = False
            self._press_pos = None
            self.drag_finished.emit()
            event.accept()
            return
        self._press_pos = None
        super().mouseReleaseEvent(event)


class CollapsibleLayerGroup(QGroupBox):
    """Movable layer card with independent visibility and expansion states."""

    visibility_changed = Signal(bool)
    drag_started = Signal()
    drag_moved = Signal(QPoint)
    drag_finished = Signal()

    def __init__(
        self,
        layer_key: str,
        title: str,
        *,
        checked: bool = True,
        visibility_control: bool = True,
    ) -> None:
        # The visible title is the movable custom header below; the native
        # QGroupBox title stays empty to avoid platform-specific title layout.
        super().__init__("")
        self.layer_key = str(layer_key)
        self._section_title = str(title)
        self._expanded = False
        self._has_visibility_control = bool(visibility_control)
        self._visibility_control_enabled = bool(visibility_control)
        self._checked = bool(checked)
        self.setProperty("displaySection", True)
        self.setProperty("layerVisible", self._checked)
        self.setStyleSheet(_LAYER_SECTION_STYLESHEET)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.header_button = LayerHeaderButton()
        self.header_button.setProperty("layerHeader", True)
        self.header_button.setText(self._section_title)
        self.header_button.setArrowType(Qt.RightArrow)
        self.header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # Long canonical layer names should use the width assigned by the
        # Display column instead of widening the whole right sidebar.
        self.header_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.header_button.setAccessibleName(f"{self._section_title} settings")
        self.header_button.setToolTip(
            "Click to expand or collapse; drag to change the stacking order"
        )

        header = QWidget()
        header.setObjectName("displaySectionHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 2, 7, 2)
        header_layout.setSpacing(0)
        header_layout.addWidget(self.header_button, 1)

        self.content_widget = QWidget()
        self.content_widget.setObjectName("displaySectionContent")
        self.content_widget.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.content_widget)

        self.header_button.clicked.connect(self._on_header_clicked)
        self.header_button.drag_started.connect(self.drag_started.emit)
        self.header_button.drag_moved.connect(self.drag_moved.emit)
        self.header_button.drag_finished.connect(self.drag_finished.emit)

    def title(self) -> str:
        """Return the visible section title (matching ``QGroupBox.title``)."""

        return self._section_title

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        """Set layer visibility without changing parameter expansion."""

        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.setProperty("layerVisible", checked)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        self.visibility_changed.emit(checked)

    def isVisibilityControlEnabled(self) -> bool:
        return self._has_visibility_control and self._visibility_control_enabled

    def setVisibilityControlEnabled(self, enabled: bool) -> None:
        self._visibility_control_enabled = bool(enabled)

    def isExpanded(self) -> bool:
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.content_widget.setVisible(self._expanded)
        self.header_button.setArrowType(
            Qt.DownArrow if self._expanded else Qt.RightArrow
        )
        self.refresh_minimum_height()
        self.updateGeometry()

    def refresh_minimum_height(self) -> None:
        """Keep wrapped card content from being compressed below its required height."""

        if self._expanded:
            for label in self.content_widget.findChildren(QLabel):
                if label.wordWrap() and label.width() > 0:
                    label.setMinimumHeight(label.heightForWidth(label.width()))
        content_layout = self.content_widget.layout()
        if content_layout is not None:
            content_layout.invalidate()
        group_layout = self.layout()
        if group_layout is not None:
            group_layout.invalidate()
        required_height = 0
        if self._expanded and self.width() > 0 and self.hasHeightForWidth():
            required_height = max(0, self.heightForWidth(self.width()))
        if self.minimumHeight() != required_height:
            self.setMinimumHeight(required_height)
            self.updateGeometry()
        parent = self.parentWidget()
        if isinstance(parent, LayerSlot):
            parent.refresh_minimum_height()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refresh_minimum_height()

    def _on_header_clicked(self) -> None:
        self.setExpanded(not self.isExpanded())


class ColorButton(QPushButton):
    """Small swatch button opening a color dialog."""

    color_changed = Signal(str)

    def __init__(self, color_hex: str, tooltip: str = "") -> None:
        super().__init__()
        self.setFixedSize(26, 18)
        if tooltip:
            self.setToolTip(tooltip)
        self._color_hex = color_hex
        self._apply_swatch()
        self.clicked.connect(self._pick_color)

    def _apply_swatch(self) -> None:
        self.setStyleSheet(
            f"background-color: {self._color_hex}; border: 1px solid #555;"
        )

    def set_color(self, color_hex: str) -> None:
        self._color_hex = color_hex
        self._apply_swatch()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._color_hex), self, "Pick color")
        if color.isValid():
            self._color_hex = color.name()
            self._apply_swatch()
            self.color_changed.emit(self._color_hex)


class DensityColorBarButton(QPushButton):
    """Clickable alpha ramp that also serves as the density color picker."""

    color_changed = Signal(str)

    def __init__(self, color_hex: str, tooltip: str = "") -> None:
        super().__init__()
        self.setFixedHeight(18)
        self.setMinimumWidth(48)
        if tooltip:
            self.setToolTip(tooltip)
        self._color_hex = color_hex
        self.clicked.connect(self._pick_color)

    def set_color(self, color_hex: str) -> None:
        self._color_hex = color_hex
        self.update()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._color_hex), self, "Pick color")
        if color.isValid():
            self._color_hex = color.name()
            self.update()
            self.color_changed.emit(self._color_hex)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        rect = self.rect().adjusted(1, 1, -2, -2)
        square = 4
        for y in range(rect.top(), rect.bottom() + 1, square):
            for x in range(rect.left(), rect.right() + 1, square):
                parity = ((x - rect.left()) // square + (y - rect.top()) // square) % 2
                painter.fillRect(
                    x,
                    y,
                    min(square, rect.right() - x + 1),
                    min(square, rect.bottom() - y + 1),
                    QColor("#777777" if parity else "#d0d0d0"),
                )
        start = QColor(self._color_hex)
        start.setAlpha(0)
        gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
        gradient.setColorAt(0.0, start)
        gradient.setColorAt(1.0, QColor(self._color_hex))
        painter.fillRect(rect, QBrush(gradient))
        painter.setPen(QPen(QColor("#555555")))
        painter.drawRect(rect)


class AccumulationGammaControl(QWidget):
    """Synchronized slider and numeric editor for normalized-map Gamma."""

    valueChanged = Signal(float)

    def __init__(self, accessible_prefix: str, parent=None) -> None:
        super().__init__(parent)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(round(DEFAULT_ACCUMULATION_GAMMA * 100))
        self.slider.setAccessibleName(f"{accessible_prefix} Gamma slider")
        self.slider.setToolTip(
            "Exponent applied after the selected map is normalized to 0–1"
        )

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(DEFAULT_ACCUMULATION_GAMMA)
        self.spin.setKeyboardTracking(False)
        self.spin.setMaximumWidth(68)
        self.spin.setAccessibleName(f"{accessible_prefix} Gamma value")
        self.spin.setToolTip(self.slider.toolTip())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return float(self.spin.value())

    def _from_slider(self, value: int) -> None:
        gamma = float(value) / 100.0
        previous = self.spin.blockSignals(True)
        self.spin.setValue(gamma)
        self.spin.blockSignals(previous)
        self.valueChanged.emit(gamma)

    def _from_spin(self, value: float) -> None:
        previous = self.slider.blockSignals(True)
        self.slider.setValue(round(float(value) * 100.0))
        self.slider.blockSignals(previous)
        self.valueChanged.emit(float(value))


class DisplayPanel(QWidget):
    """Emits ``style_changed(layer_key, field, value)`` on any control change."""

    style_changed = Signal(str, str, object)
    db_range_changed = Signal(float, float)
    display_settings_changed = Signal(object)
    interpolation_changed = Signal(bool, float, float)
    dataset_plot_map_changed = Signal(str)
    dataset_plot_density_clim_changed = Signal(str, float, float)
    dataset_plot_clim_changed = Signal(float, float)
    dataset_plot_gamma_changed = Signal(float)
    multiple_dataset_import_requested = Signal()
    multiple_dataset_plot_map_changed = Signal(str)
    multiple_dataset_plot_density_clim_changed = Signal(str, float, float)
    multiple_dataset_plot_clim_changed = Signal(float, float)
    multiple_dataset_plot_gamma_changed = Signal(float)
    track_plot_map_changed = Signal(str)
    track_plot_clim_changed = Signal(float, float)
    layer_order_changed = Signal(object)

    def __init__(self, styles: dict[str, LayerStyle]) -> None:
        super().__init__()
        self._styles = styles
        self._layer_groups: dict[str, CollapsibleLayerGroup] = {}
        self._layer_order = list(DEFAULT_LAYER_ORDER)
        self._slots: list[LayerSlot] = []
        self._dragged_group: CollapsibleLayerGroup | None = None
        self._drag_target_index: int | None = None
        self._drag_order_changed = False
        self._overlay_memory: list[str] = []
        self._dataset_density_clims = {
            HARD_DENSITY: (0.0, 1.0),
            AA_DENSITY: (0.0, 1.0),
        }
        self._multiple_dataset_density_clims = {
            HARD_DENSITY: (0.0, 1.0),
            AA_DENSITY: (0.0, 1.0),
        }
        self._updating_dataset_density_clim = False
        self._updating_dataset_plot_clim = False
        self._updating_multiple_dataset_density_clim = False
        self._updating_multiple_dataset_plot_clim = False
        self._updating_track_plot_clim = False

        root = QVBoxLayout(self)
        self._root_layout = root
        # Expanded cards change the panel's height dynamically.  Propagate the
        # new layout bounds to the owning QScrollArea so it scrolls instead of
        # compressing word-wrapped status text below its height-for-width.
        root.setSizeConstraint(QLayout.SetMinAndMaxSize)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.toggle_button = self._make_state_button(
            "Keep images only (T)",
            "Hide every non-image layer; press again to restore the previous set. "
            "The Ultrasound Movie layer's own visibility is unchanged.",
        )
        self.toggle_button.clicked.connect(self.toggle_overlays)
        root.addWidget(self.toggle_button)

        self._build_image_group()
        self._build_layer_group(
            LAYER_LABELS["track_point"],
            "track_point",
            colors=[("Color", "track_point", "Marker color")],
            size_spec=("Radius", 1.0, 12.0, 0.5),
        )
        self._build_layer_group(
            LAYER_LABELS["all_lines"],
            "all_lines",
            colors=[("Color", "all_lines", "Line color")],
            size_spec=("Width", 0.5, 6.0, 0.5),
        )
        self._build_layer_group(
            LAYER_LABELS["track_path"],
            "track_path",
            colors=[
                ("Colors", "track_path", "Line color"),
                (
                    "",
                    "track_path_verified",
                    "Line color of verified tracks (also used by the all-tracks layer)",
                ),
            ],
            size_spec=("Width", 0.5, 6.0, 0.5),
        )
        self._build_layer_group(
            LAYER_LABELS["points"],
            "points",
            colors=[
                ("Colors", "points_current", "Points of the selected track"),
                ("", "points_other", "Points of other tracks"),
                ("", "points", "Unassigned detections"),
            ],
            size_spec=("Radius", 0.5, 8.0, 0.5),
            status_subsets=[
                (
                    "Enable verified points",
                    "points_verified",
                    "Override verified points from other tracks with this color",
                ),
                (
                    "Enable flagged points",
                    "points_flagged",
                    "Override flagged points from other tracks with this color",
                ),
            ],
        )
        self._build_dataset_plot_group()
        self._build_multiple_dataset_plot_group()
        self._build_track_plot_group()
        for layer_key in self._layer_order:
            group = self._layer_groups[layer_key]
            group.drag_started.connect(
                lambda current=group: self._start_layer_drag(current)
            )
            group.drag_moved.connect(self._move_layer_drag)
            group.drag_finished.connect(self._finish_layer_drag)
        for slot_number in range(1, len(self._layer_order) + 1):
            slot = LayerSlot(slot_number)
            self._slots.append(slot)
            root.addWidget(slot)
        self._place_groups_in_slots()
        root.addStretch(1)
        self.configure_dataset_plotting(
            False,
            "Load a Batch accumulation checkpoint to supply the PALA grid and parameters.",
        )
        self.configure_multiple_dataset_plotting(
            False,
            f"Import multiple dataset MAT files to create "
            f"{LAYER_LABELS['multiple_dataset_plot']}.",
        )
        self.configure_track_plotting(
            (),
            "No Batch accumulation checkpoint is loaded.",
        )
        self._update_toggle_button()

    # ------------------------------------------------------------- builders

    @staticmethod
    def _make_grid(group: CollapsibleLayerGroup) -> QGridLayout:
        grid = QGridLayout(group.content_widget)
        grid.setContentsMargins(9, 5, 9, 7)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        return grid

    @staticmethod
    def _make_state_button(text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setProperty("stateButton", True)
        button.setStyleSheet(_STATE_BUTTON_STYLESHEET)
        button.setToolTip(tooltip)
        return button

    def _build_image_group(self) -> CollapsibleLayerGroup:
        group = CollapsibleLayerGroup(
            "image",
            LAYER_LABELS["image"],
            checked=self._styles["image"].visible,
        )
        group.visibility_changed.connect(
            lambda checked: self._on_group_toggled("image", checked)
        )
        self._layer_groups["image"] = group
        self.image_group = group
        grid = self._make_grid(group)

        grid.addWidget(QLabel("Opacity"), 0, 0)
        grid.addWidget(self._make_opacity_slider("image"), 0, 1, 1, 3)

        self.dynamic_range_panel = create_dynamic_range_panel(
            self,
            self._emit_display_settings,
        )
        self.db_min_spin.valueChanged.connect(self._emit_db_range)
        self.db_max_spin.valueChanged.connect(self._emit_db_range)
        grid.addWidget(self.dynamic_range_panel, 1, 0, 1, 4)

        self.interp_factor_spin = QDoubleSpinBox()
        self.interp_factor_spin.setRange(1.0, 20.0)
        self.interp_factor_spin.setValue(5.0)
        self.interp_factor_spin.setSingleStep(1.0)
        self.interp_factor_spin.setDecimals(2)
        self.interp_factor_spin.setKeyboardTracking(False)
        self.interp_factor_spin.setMaximumWidth(64)
        self.interp_factor_spin.setToolTip(
            "Display sampling multiplier. The source image stack is not modified."
        )
        grid.addWidget(QLabel("Interp."), 2, 0)
        grid.addWidget(self.interp_factor_spin, 2, 1)

        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(0.01, 20.0)
        self.aspect_ratio_spin.setValue(1.0)
        self.aspect_ratio_spin.setSingleStep(0.1)
        self.aspect_ratio_spin.setDecimals(2)
        self.aspect_ratio_spin.setKeyboardTracking(False)
        self.aspect_ratio_spin.setMaximumWidth(64)
        self.aspect_ratio_spin.setToolTip(
            "Displayed z/x pixel-size ratio; 1.00 keeps square source pixels."
        )
        grid.addWidget(QLabel("Aspect"), 2, 2)
        grid.addWidget(self.aspect_ratio_spin, 2, 3)

        self.interpolate_button = self._make_state_button(
            "Interpolate Dataset (I)",
            "(I) Toggle between the original display and an interpolated display.",
        )
        self.interpolate_button.toggled.connect(self._emit_interpolation)
        self.interp_factor_spin.valueChanged.connect(
            self._emit_active_interpolation_parameters
        )
        self.aspect_ratio_spin.valueChanged.connect(
            self._emit_active_interpolation_parameters
        )
        grid.addWidget(self.interpolate_button, 3, 0, 1, 4)
        return group

    def _build_dataset_plot_group(self) -> CollapsibleLayerGroup:
        group = CollapsibleLayerGroup(
            "dataset_plot",
            LAYER_LABELS["dataset_plot"],
            checked=self._styles["dataset_plot"].visible,
        )
        group.visibility_changed.connect(
            lambda checked: self._on_group_toggled("dataset_plot", checked)
        )
        self._layer_groups["dataset_plot"] = group
        self.dataset_plot_group = group
        grid = self._make_grid(group)

        grid.addWidget(QLabel("Opacity"), 0, 0)
        self.dataset_plot_opacity_slider = self._make_opacity_slider("dataset_plot")
        grid.addWidget(self.dataset_plot_opacity_slider, 0, 1, 1, 3)

        grid.addWidget(QLabel("Gamma"), 1, 0)
        self.dataset_plot_gamma_control = AccumulationGammaControl(
            "Single Dataset Accumulation"
        )
        self.dataset_plot_gamma_slider = self.dataset_plot_gamma_control.slider
        self.dataset_plot_gamma_spin = self.dataset_plot_gamma_control.spin
        self.dataset_plot_gamma_control.valueChanged.connect(
            self.dataset_plot_gamma_changed.emit
        )
        grid.addWidget(self.dataset_plot_gamma_control, 1, 1, 1, 3)

        self.dataset_plot_clim_label = QLabel("Clim")
        self.dataset_plot_clim_label.setToolTip(
            "Display limits applied after normalization and Gamma; limits may exceed 1"
        )
        grid.addWidget(self.dataset_plot_clim_label, 2, 0)

        self.dataset_plot_density_clim_widget = QWidget()
        density_clim_row = QHBoxLayout(self.dataset_plot_density_clim_widget)
        density_clim_row.setContentsMargins(0, 0, 0, 0)
        density_clim_row.setSpacing(4)

        self.dataset_plot_density_clim_min_spin = QDoubleSpinBox()
        self.dataset_plot_density_clim_min_spin.setRange(0.0, 999_999_999.9999)
        self.dataset_plot_density_clim_min_spin.setDecimals(4)
        self.dataset_plot_density_clim_min_spin.setSingleStep(0.1)
        self.dataset_plot_density_clim_min_spin.setKeyboardTracking(False)
        self.dataset_plot_density_clim_min_spin.setMaximumWidth(68)
        self.dataset_plot_density_clim_min_spin.setToolTip(
            "Normalized-map Clim minimum; the map data itself remains in 0–1"
        )

        self.dataset_plot_color_button = DensityColorBarButton(
            self._styles["dataset_plot"].color,
            "Density alpha scale; click to choose the Hard/AA Density color",
        )
        self.dataset_plot_color_button.color_changed.connect(
            lambda color_hex: self.style_changed.emit(
                "dataset_plot",
                "color",
                color_hex,
            )
        )

        self.dataset_plot_density_clim_max_spin = QDoubleSpinBox()
        self.dataset_plot_density_clim_max_spin.setRange(0.0001, 1_000_000_000.0)
        self.dataset_plot_density_clim_max_spin.setDecimals(4)
        self.dataset_plot_density_clim_max_spin.setSingleStep(0.1)
        self.dataset_plot_density_clim_max_spin.setKeyboardTracking(False)
        self.dataset_plot_density_clim_max_spin.setMaximumWidth(68)
        self.dataset_plot_density_clim_max_spin.setToolTip(
            "Normalized-map Clim maximum; values above 1 are allowed"
        )
        self.dataset_plot_density_clim_max_spin.setValue(1.0)

        self.dataset_plot_density_clim_min_spin.valueChanged.connect(
            self._emit_dataset_density_clim
        )
        self.dataset_plot_density_clim_max_spin.valueChanged.connect(
            self._emit_dataset_density_clim
        )
        density_clim_row.addWidget(self.dataset_plot_density_clim_min_spin)
        density_clim_row.addWidget(self.dataset_plot_color_button, 1)
        density_clim_row.addWidget(self.dataset_plot_density_clim_max_spin)
        grid.addWidget(self.dataset_plot_density_clim_widget, 2, 1, 1, 3)

        self.dataset_plot_velocity_clim_widget = QWidget()
        clim_row = QHBoxLayout(self.dataset_plot_velocity_clim_widget)
        clim_row.setContentsMargins(0, 0, 0, 0)
        clim_row.setSpacing(4)

        self.dataset_plot_clim_min_spin = QDoubleSpinBox()
        self.dataset_plot_clim_min_spin.setRange(0.0, 999_999.99)
        self.dataset_plot_clim_min_spin.setDecimals(2)
        self.dataset_plot_clim_min_spin.setSingleStep(1.0)
        self.dataset_plot_clim_min_spin.setKeyboardTracking(False)
        self.dataset_plot_clim_min_spin.setMaximumWidth(62)
        self.dataset_plot_clim_min_spin.setToolTip(
            "Normalized-map Clim minimum; the map data itself remains in 0–1"
        )

        self.dataset_plot_velocity_colorbar = QFrame()
        self.dataset_plot_velocity_colorbar.setFrameShape(QFrame.NoFrame)
        self.dataset_plot_velocity_colorbar.setFixedHeight(18)
        self.dataset_plot_velocity_colorbar.setMinimumWidth(48)
        self.dataset_plot_velocity_colorbar.setToolTip(
            "Jet scale after normalization and Gamma; values outside Clim are clipped"
        )
        self.dataset_plot_velocity_colorbar.setStyleSheet(
            "QFrame {"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #000080, stop:0.125 #0000ff, stop:0.375 #00ffff, "
            "stop:0.625 #ffff00, stop:0.875 #ff0000, stop:1 #800000);"
            "border: 1px solid #555;"
            "}"
        )

        self.dataset_plot_clim_max_spin = QDoubleSpinBox()
        self.dataset_plot_clim_max_spin.setRange(0.01, 1_000_000.0)
        self.dataset_plot_clim_max_spin.setDecimals(2)
        self.dataset_plot_clim_max_spin.setSingleStep(1.0)
        self.dataset_plot_clim_max_spin.setKeyboardTracking(False)
        self.dataset_plot_clim_max_spin.setMaximumWidth(62)
        self.dataset_plot_clim_max_spin.setToolTip(
            "Normalized-map Clim maximum; values above 1 are allowed"
        )
        self.dataset_plot_clim_max_spin.setValue(DEFAULT_ACCUMULATION_CLIM[1])

        self.dataset_plot_clim_min_spin.valueChanged.connect(
            self._emit_dataset_plot_clim
        )
        self.dataset_plot_clim_max_spin.valueChanged.connect(
            self._emit_dataset_plot_clim
        )
        clim_row.addWidget(self.dataset_plot_clim_min_spin)
        clim_row.addWidget(self.dataset_plot_velocity_colorbar, 1)
        clim_row.addWidget(self.dataset_plot_clim_max_spin)
        grid.addWidget(self.dataset_plot_velocity_clim_widget, 2, 1, 1, 3)

        grid.addWidget(QLabel("Map"), 3, 0)
        self.dataset_plot_map_combo = QComboBox()
        self.dataset_plot_map_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.dataset_plot_map_combo.setMinimumContentsLength(8)
        self.dataset_plot_map_combo.setMaximumWidth(175)
        for map_name in FULL_TRACK_MAPS:
            self.dataset_plot_map_combo.addItem(TRACK_MAP_LABELS[map_name], map_name)
        self.dataset_plot_map_combo.setCurrentIndex(
            self.dataset_plot_map_combo.findData(AA_DENSITY)
        )
        self.dataset_plot_map_combo.currentIndexChanged.connect(
            self._on_dataset_plot_map_changed
        )
        grid.addWidget(self.dataset_plot_map_combo, 3, 1, 1, 3)

        self.dataset_plot_status_label = QLabel()
        self.dataset_plot_status_label.setWordWrap(True)
        self.dataset_plot_status_label.setToolTip(
            "This layer is rebuilt from every current session track. It uses the "
            "Batch checkpoint only for the original PALA grid and parameters."
        )
        grid.addWidget(self.dataset_plot_status_label, 4, 0, 1, 4)
        self._dataset_parameter_widgets = (
            self.dataset_plot_opacity_slider,
            self.dataset_plot_gamma_control,
            self.dataset_plot_clim_label,
            self.dataset_plot_density_clim_widget,
            self.dataset_plot_velocity_clim_widget,
            self.dataset_plot_map_combo,
        )
        self._update_dataset_plot_control_mode()
        return group

    def _build_multiple_dataset_import_button(self) -> QPushButton:
        """Create the retained-snapshot importer inside its owning layer card."""

        button = QPushButton("Import / replace datasets…")
        button.setAccessibleName("Import or replace multiple datasets")
        button.setToolTip(
            "Choose one or more dataset MAT files. A successful import replaces "
            f"{LAYER_LABELS['multiple_dataset_plot']}; cancelling or an error keeps it."
        )
        button.clicked.connect(
            self.multiple_dataset_import_requested.emit
        )
        self.import_multiple_dataset_button = button
        return button

    def _build_multiple_dataset_plot_group(self) -> CollapsibleLayerGroup:
        group = CollapsibleLayerGroup(
            "multiple_dataset_plot",
            LAYER_LABELS["multiple_dataset_plot"],
            checked=self._styles["multiple_dataset_plot"].visible,
        )
        group.visibility_changed.connect(
            lambda checked: self._on_group_toggled(
                "multiple_dataset_plot",
                checked,
            )
        )
        self._layer_groups["multiple_dataset_plot"] = group
        self.multiple_dataset_plot_group = group
        grid = self._make_grid(group)

        grid.addWidget(self._build_multiple_dataset_import_button(), 0, 0, 1, 4)

        grid.addWidget(QLabel("Opacity"), 1, 0)
        self.multiple_dataset_plot_opacity_slider = self._make_opacity_slider(
            "multiple_dataset_plot"
        )
        grid.addWidget(self.multiple_dataset_plot_opacity_slider, 1, 1, 1, 3)

        grid.addWidget(QLabel("Gamma"), 2, 0)
        self.multiple_dataset_plot_gamma_control = AccumulationGammaControl(
            "Multi Dataset Accumulation"
        )
        self.multiple_dataset_plot_gamma_slider = (
            self.multiple_dataset_plot_gamma_control.slider
        )
        self.multiple_dataset_plot_gamma_spin = (
            self.multiple_dataset_plot_gamma_control.spin
        )
        self.multiple_dataset_plot_gamma_control.valueChanged.connect(
            self.multiple_dataset_plot_gamma_changed.emit
        )
        grid.addWidget(self.multiple_dataset_plot_gamma_control, 2, 1, 1, 3)

        self.multiple_dataset_plot_clim_label = QLabel("Clim")
        self.multiple_dataset_plot_clim_label.setToolTip(
            "Display limits applied after normalization and Gamma; limits may exceed 1"
        )
        grid.addWidget(self.multiple_dataset_plot_clim_label, 3, 0)

        self.multiple_dataset_plot_density_clim_widget = QWidget()
        density_clim_row = QHBoxLayout(
            self.multiple_dataset_plot_density_clim_widget
        )
        density_clim_row.setContentsMargins(0, 0, 0, 0)
        density_clim_row.setSpacing(4)

        self.multiple_dataset_plot_density_clim_min_spin = QDoubleSpinBox()
        self.multiple_dataset_plot_density_clim_min_spin.setRange(
            0.0,
            999_999_999.9999,
        )
        self.multiple_dataset_plot_density_clim_min_spin.setDecimals(4)
        self.multiple_dataset_plot_density_clim_min_spin.setSingleStep(0.1)
        self.multiple_dataset_plot_density_clim_min_spin.setKeyboardTracking(False)
        self.multiple_dataset_plot_density_clim_min_spin.setMaximumWidth(68)
        self.multiple_dataset_plot_density_clim_min_spin.setToolTip(
            "Normalized-map Clim minimum; the map data itself remains in 0–1"
        )

        self.multiple_dataset_plot_color_button = DensityColorBarButton(
            self._styles["multiple_dataset_plot"].color,
            "Density alpha scale; click to choose the Hard/AA Density color",
        )
        self.multiple_dataset_plot_color_button.color_changed.connect(
            lambda color_hex: self.style_changed.emit(
                "multiple_dataset_plot",
                "color",
                color_hex,
            )
        )

        self.multiple_dataset_plot_density_clim_max_spin = QDoubleSpinBox()
        self.multiple_dataset_plot_density_clim_max_spin.setRange(
            0.0001,
            1_000_000_000.0,
        )
        self.multiple_dataset_plot_density_clim_max_spin.setDecimals(4)
        self.multiple_dataset_plot_density_clim_max_spin.setSingleStep(0.1)
        self.multiple_dataset_plot_density_clim_max_spin.setKeyboardTracking(False)
        self.multiple_dataset_plot_density_clim_max_spin.setMaximumWidth(68)
        self.multiple_dataset_plot_density_clim_max_spin.setToolTip(
            "Normalized-map Clim maximum; values above 1 are allowed"
        )
        self.multiple_dataset_plot_density_clim_max_spin.setValue(1.0)
        self.multiple_dataset_plot_density_clim_min_spin.valueChanged.connect(
            self._emit_multiple_dataset_density_clim
        )
        self.multiple_dataset_plot_density_clim_max_spin.valueChanged.connect(
            self._emit_multiple_dataset_density_clim
        )
        density_clim_row.addWidget(
            self.multiple_dataset_plot_density_clim_min_spin
        )
        density_clim_row.addWidget(self.multiple_dataset_plot_color_button, 1)
        density_clim_row.addWidget(
            self.multiple_dataset_plot_density_clim_max_spin
        )
        grid.addWidget(
            self.multiple_dataset_plot_density_clim_widget,
            3,
            1,
            1,
            3,
        )

        self.multiple_dataset_plot_velocity_clim_widget = QWidget()
        velocity_clim_row = QHBoxLayout(
            self.multiple_dataset_plot_velocity_clim_widget
        )
        velocity_clim_row.setContentsMargins(0, 0, 0, 0)
        velocity_clim_row.setSpacing(4)

        self.multiple_dataset_plot_clim_min_spin = QDoubleSpinBox()
        self.multiple_dataset_plot_clim_min_spin.setRange(0.0, 999_999.99)
        self.multiple_dataset_plot_clim_min_spin.setDecimals(2)
        self.multiple_dataset_plot_clim_min_spin.setSingleStep(1.0)
        self.multiple_dataset_plot_clim_min_spin.setKeyboardTracking(False)
        self.multiple_dataset_plot_clim_min_spin.setMaximumWidth(62)
        self.multiple_dataset_plot_clim_min_spin.setToolTip(
            "Normalized-map Clim minimum; the map data itself remains in 0–1"
        )

        self.multiple_dataset_plot_velocity_colorbar = QFrame()
        self.multiple_dataset_plot_velocity_colorbar.setFrameShape(QFrame.NoFrame)
        self.multiple_dataset_plot_velocity_colorbar.setFixedHeight(18)
        self.multiple_dataset_plot_velocity_colorbar.setMinimumWidth(48)
        self.multiple_dataset_plot_velocity_colorbar.setToolTip(
            "Jet scale after normalization and Gamma; values outside Clim are clipped"
        )
        self.multiple_dataset_plot_velocity_colorbar.setStyleSheet(
            "QFrame {"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #000080, stop:0.125 #0000ff, stop:0.375 #00ffff, "
            "stop:0.625 #ffff00, stop:0.875 #ff0000, stop:1 #800000);"
            "border: 1px solid #555;"
            "}"
        )

        self.multiple_dataset_plot_clim_max_spin = QDoubleSpinBox()
        self.multiple_dataset_plot_clim_max_spin.setRange(0.01, 1_000_000.0)
        self.multiple_dataset_plot_clim_max_spin.setDecimals(2)
        self.multiple_dataset_plot_clim_max_spin.setSingleStep(1.0)
        self.multiple_dataset_plot_clim_max_spin.setKeyboardTracking(False)
        self.multiple_dataset_plot_clim_max_spin.setMaximumWidth(62)
        self.multiple_dataset_plot_clim_max_spin.setToolTip(
            "Normalized-map Clim maximum; values above 1 are allowed"
        )
        self.multiple_dataset_plot_clim_max_spin.setValue(
            DEFAULT_ACCUMULATION_CLIM[1]
        )
        self.multiple_dataset_plot_clim_min_spin.valueChanged.connect(
            self._emit_multiple_dataset_plot_clim
        )
        self.multiple_dataset_plot_clim_max_spin.valueChanged.connect(
            self._emit_multiple_dataset_plot_clim
        )
        velocity_clim_row.addWidget(self.multiple_dataset_plot_clim_min_spin)
        velocity_clim_row.addWidget(
            self.multiple_dataset_plot_velocity_colorbar,
            1,
        )
        velocity_clim_row.addWidget(self.multiple_dataset_plot_clim_max_spin)
        grid.addWidget(
            self.multiple_dataset_plot_velocity_clim_widget,
            3,
            1,
            1,
            3,
        )

        grid.addWidget(QLabel("Map"), 4, 0)
        self.multiple_dataset_plot_map_combo = QComboBox()
        self.multiple_dataset_plot_map_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.multiple_dataset_plot_map_combo.setMinimumContentsLength(8)
        self.multiple_dataset_plot_map_combo.setMaximumWidth(175)
        for map_name in FULL_TRACK_MAPS:
            self.multiple_dataset_plot_map_combo.addItem(
                TRACK_MAP_LABELS[map_name],
                map_name,
            )
        self.multiple_dataset_plot_map_combo.setCurrentIndex(
            self.multiple_dataset_plot_map_combo.findData(AA_DENSITY)
        )
        self.multiple_dataset_plot_map_combo.currentIndexChanged.connect(
            self._on_multiple_dataset_plot_map_changed
        )
        grid.addWidget(self.multiple_dataset_plot_map_combo, 4, 1, 1, 3)

        self.multiple_dataset_plot_status_label = QLabel()
        self.multiple_dataset_plot_status_label.setWordWrap(True)
        self.multiple_dataset_plot_status_label.setToolTip(
            "This snapshot is rebuilt only by Import / replace datasets. "
            "Loading, switching, or editing the current dataset does not change it."
        )
        grid.addWidget(self.multiple_dataset_plot_status_label, 5, 0, 1, 4)
        self._multiple_dataset_parameter_widgets = (
            self.multiple_dataset_plot_opacity_slider,
            self.multiple_dataset_plot_gamma_control,
            self.multiple_dataset_plot_clim_label,
            self.multiple_dataset_plot_density_clim_widget,
            self.multiple_dataset_plot_velocity_clim_widget,
            self.multiple_dataset_plot_map_combo,
        )
        self._update_multiple_dataset_plot_control_mode()
        return group

    def _build_layer_group(
        self,
        title: str,
        layer_key: str,
        *,
        colors: list[tuple[str, str, str]],
        size_spec: tuple[str, float, float, float] | None,
        status_subsets: list[tuple[str, str, str]] | None = None,
    ) -> CollapsibleLayerGroup:
        group = CollapsibleLayerGroup(
            layer_key,
            title,
            checked=self._styles[layer_key].visible,
        )
        group.visibility_changed.connect(
            lambda checked, key=layer_key: self._on_group_toggled(key, checked)
        )
        self._layer_groups[layer_key] = group
        grid = self._make_grid(group)

        grid.addWidget(QLabel("Opacity"), 0, 0)
        grid.addWidget(self._make_opacity_slider(layer_key), 0, 1, 1, 3)

        color_label = colors[0][0]
        grid.addWidget(QLabel(color_label), 1, 0)
        color_row = QHBoxLayout()
        color_row.setSpacing(4)
        for _, color_key, tooltip in colors:
            button = ColorButton(self._styles[color_key].color, tooltip)
            button.color_changed.connect(
                lambda color_hex, key=color_key: self.style_changed.emit(
                    key, "color", color_hex
                )
            )
            color_row.addWidget(button)
        color_row.addStretch(1)

        if size_spec is not None:
            label, minimum, maximum, step = size_spec
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(1)
            spin.setMaximumWidth(58)
            spin.setObjectName(f"{layer_key}SizeSpin")
            spin.setValue(self._styles[layer_key].size)
            spin.valueChanged.connect(
                lambda value, key=layer_key: self.style_changed.emit(
                    key, "size", float(value)
                )
            )
            grid.addLayout(color_row, 1, 1)
            grid.addWidget(QLabel(label), 1, 2, Qt.AlignRight)
            grid.addWidget(spin, 1, 3, Qt.AlignRight)
        else:
            grid.addLayout(color_row, 1, 1, 1, 3)

        for row, (label, style_key, tooltip) in enumerate(
            status_subsets or (),
            start=2,
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(self._styles[style_key].visible)
            checkbox.setToolTip(tooltip)
            checkbox.setObjectName(f"{style_key}Checkbox")
            checkbox.toggled.connect(
                lambda checked, key=style_key: self.style_changed.emit(
                    key, "visible", bool(checked)
                )
            )
            color_button = ColorButton(self._styles[style_key].color, tooltip)
            color_button.setObjectName(f"{style_key}ColorButton")
            color_button.color_changed.connect(
                lambda color_hex, key=style_key: self.style_changed.emit(
                    key, "color", color_hex
                )
            )
            grid.addWidget(checkbox, row, 0, 1, 3)
            grid.addWidget(color_button, row, 3, Qt.AlignRight)
            setattr(self, f"{style_key}_checkbox", checkbox)
            setattr(self, f"{style_key}_color_button", color_button)

        return group

    def _build_track_plot_group(self) -> CollapsibleLayerGroup:
        group = CollapsibleLayerGroup(
            "track_plot",
            LAYER_LABELS["track_plot"],
            checked=self._styles["track_plot"].visible,
        )
        group.visibility_changed.connect(
            lambda checked: self._on_group_toggled("track_plot", checked)
        )
        self._layer_groups["track_plot"] = group
        self.track_plot_group = group
        grid = self._make_grid(group)

        grid.addWidget(QLabel("Opacity"), 0, 0)
        self.track_plot_opacity_slider = self._make_opacity_slider("track_plot")
        grid.addWidget(self.track_plot_opacity_slider, 0, 1, 1, 3)

        self.track_plot_color_label = QLabel("Color")
        grid.addWidget(self.track_plot_color_label, 1, 0)
        self.track_plot_color_button = ColorButton(
            self._styles["track_plot"].color,
            "Color for Hard/AA Density; velocity maps keep their Jet colors",
        )
        self.track_plot_color_button.color_changed.connect(
            lambda color_hex: self.style_changed.emit(
                "track_plot",
                "color",
                color_hex,
            )
        )
        grid.addWidget(self.track_plot_color_button, 1, 1, Qt.AlignLeft)

        self.track_plot_velocity_clim_widget = QWidget()
        clim_row = QHBoxLayout(self.track_plot_velocity_clim_widget)
        clim_row.setContentsMargins(0, 0, 0, 0)
        clim_row.setSpacing(4)

        self.track_plot_clim_min_spin = QDoubleSpinBox()
        self.track_plot_clim_min_spin.setRange(0.0, 999_999.99)
        self.track_plot_clim_min_spin.setDecimals(2)
        self.track_plot_clim_min_spin.setSingleStep(1.0)
        self.track_plot_clim_min_spin.setKeyboardTracking(False)
        self.track_plot_clim_min_spin.setMaximumWidth(62)
        self.track_plot_clim_min_spin.setToolTip("Velocity color limit minimum")

        self.track_plot_velocity_colorbar = QFrame()
        self.track_plot_velocity_colorbar.setFrameShape(QFrame.NoFrame)
        self.track_plot_velocity_colorbar.setFixedHeight(18)
        self.track_plot_velocity_colorbar.setMinimumWidth(48)
        self.track_plot_velocity_colorbar.setToolTip(
            "Jet velocity scale; values outside the limits are clipped"
        )
        self.track_plot_velocity_colorbar.setStyleSheet(
            "QFrame {"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #000080, stop:0.125 #0000ff, stop:0.375 #00ffff, "
            "stop:0.625 #ffff00, stop:0.875 #ff0000, stop:1 #800000);"
            "border: 1px solid #555;"
            "}"
        )

        self.track_plot_clim_max_spin = QDoubleSpinBox()
        self.track_plot_clim_max_spin.setRange(0.01, 1_000_000.0)
        self.track_plot_clim_max_spin.setDecimals(2)
        self.track_plot_clim_max_spin.setSingleStep(1.0)
        self.track_plot_clim_max_spin.setKeyboardTracking(False)
        self.track_plot_clim_max_spin.setMaximumWidth(62)
        self.track_plot_clim_max_spin.setToolTip("Velocity color limit maximum")
        self.track_plot_clim_max_spin.setValue(15.0)

        self.track_plot_clim_min_spin.valueChanged.connect(
            self._emit_track_plot_clim
        )
        self.track_plot_clim_max_spin.valueChanged.connect(
            self._emit_track_plot_clim
        )
        clim_row.addWidget(self.track_plot_clim_min_spin)
        clim_row.addWidget(self.track_plot_velocity_colorbar, 1)
        clim_row.addWidget(self.track_plot_clim_max_spin)
        grid.addWidget(self.track_plot_velocity_clim_widget, 1, 0, 1, 4)

        grid.addWidget(QLabel("Map"), 2, 0)
        self.track_plot_map_combo = QComboBox()
        self.track_plot_map_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.track_plot_map_combo.setMinimumContentsLength(8)
        self.track_plot_map_combo.setMaximumWidth(175)
        for map_name in FULL_TRACK_MAPS:
            self.track_plot_map_combo.addItem(TRACK_MAP_LABELS[map_name], map_name)
        self.track_plot_map_combo.currentIndexChanged.connect(
            self._on_track_plot_map_changed
        )
        grid.addWidget(self.track_plot_map_combo, 2, 1, 1, 3)

        self.track_plot_status_label = QLabel()
        self.track_plot_status_label.setWordWrap(True)
        self.track_plot_status_label.setToolTip(
            "This layer displays the original Batch checkpoint contribution; "
            "it is not recomputed after track edits."
        )
        grid.addWidget(self.track_plot_status_label, 3, 0, 1, 4)
        self._track_plot_parameter_widgets = (
            self.track_plot_opacity_slider,
            self.track_plot_color_label,
            self.track_plot_color_button,
            self.track_plot_velocity_clim_widget,
            self.track_plot_map_combo,
        )
        self._update_track_plot_control_mode()
        return group

    def _make_opacity_slider(self, layer_key: str) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setMinimumWidth(60)
        slider.setValue(round(self._styles[layer_key].opacity * 100))
        slider.valueChanged.connect(
            lambda value, key=layer_key: self.style_changed.emit(
                key, "opacity", value / 100.0
            )
        )
        return slider

    # ----------------------------------------------------------- layer order

    def layer_order(self) -> tuple[str, ...]:
        """Return the layer content in fixed slots, foreground to background."""

        return tuple(self._layer_order)

    def slot_layer_key(self, slot_number: int) -> str:
        """Return the layer currently occupying a 1-based physical slot."""

        slot_number = int(slot_number)
        if not 1 <= slot_number <= len(self._slots):
            raise ValueError("Layer slot must be between 1 and 8.")
        return self._layer_order[slot_number - 1]

    def slot(self, slot_number: int) -> LayerSlot:
        """Return a fixed 1-based slot widget for UI checks and accessibility."""

        slot_number = int(slot_number)
        if not 1 <= slot_number <= len(self._slots):
            raise ValueError("Layer slot must be between 1 and 8.")
        return self._slots[slot_number - 1]

    def toggle_slot(self, slot_number: int) -> bool:
        """Toggle the layer currently in a fixed slot; return whether it changed."""

        slot = self.slot(slot_number)
        group = slot.group
        if group is None or not group.isVisibilityControlEnabled():
            return False
        group.setChecked(not group.isChecked())
        slot.sync()
        return True

    def move_layer(self, layer_key: str, target_index: int) -> None:
        """Move one card to a final foreground-to-background list position."""

        layer_key = str(layer_key)
        if layer_key not in self._layer_order:
            raise ValueError(f"Unknown display layer: {layer_key}")
        target_index = int(target_index)
        if not 0 <= target_index < len(self._layer_order):
            raise ValueError("Layer target index is outside the display stack.")
        order = [key for key in self._layer_order if key != layer_key]
        order.insert(target_index, layer_key)
        self._apply_layer_order(order, emit=True)

    def _apply_layer_order(self, order: list[str], *, emit: bool) -> bool:
        if len(order) != len(self._layer_order) or set(order) != set(
            self._layer_order
        ):
            raise ValueError("Layer order must contain every display layer once.")
        if order == self._layer_order:
            return False

        self._layer_order = list(order)
        self._place_groups_in_slots()
        self._root_layout.invalidate()
        self.updateGeometry()
        if emit:
            self.layer_order_changed.emit(self.layer_order())
        return True

    def _place_groups_in_slots(self) -> None:
        for slot in self._slots:
            slot.take_group()
        for slot, key in zip(self._slots, self._layer_order, strict=True):
            slot.set_group(self._layer_groups[key])

    def _sync_layer_slot(self, layer_key: str) -> None:
        for slot in self._slots:
            if slot.group is self._layer_groups[layer_key]:
                slot.sync()
                return

    @staticmethod
    def _refresh_group_style(group: CollapsibleLayerGroup) -> None:
        group.style().unpolish(group)
        group.style().polish(group)
        group.update()

    def _start_layer_drag(self, group: CollapsibleLayerGroup) -> None:
        self._dragged_group = group
        self._drag_order_changed = False
        group.setProperty("dragging", True)
        self._refresh_group_style(group)
        self._set_drag_target(self._layer_order.index(group.layer_key))

    def _set_drag_target(self, target_index: int | None) -> None:
        self._drag_target_index = target_index
        for index, slot in enumerate(self._slots):
            slot.set_drop_target(index == target_index)

    def _move_layer_drag(self, global_position: QPoint) -> None:
        group = self._dragged_group
        if group is None:
            return
        pointer_y = self.mapFromGlobal(global_position).y()
        target_index = min(
            len(self._slots) - 1,
            sum(pointer_y > slot.geometry().center().y() for slot in self._slots),
        )
        self._set_drag_target(target_index)
        current_index = self._layer_order.index(group.layer_key)
        if target_index == current_index:
            return
        order = list(self._layer_order)
        order.pop(current_index)
        order.insert(target_index, group.layer_key)
        if self._apply_layer_order(order, emit=False):
            self._drag_order_changed = True

    def _finish_layer_drag(self) -> None:
        group = self._dragged_group
        if group is None:
            return
        group.setProperty("dragging", False)
        self._refresh_group_style(group)
        self._dragged_group = None
        self._set_drag_target(None)
        if self._drag_order_changed:
            self.layer_order_changed.emit(self.layer_order())
        self._drag_order_changed = False

    # -------------------------------------------------------------- signals

    def _emit_display_settings(self, *_args) -> None:
        self.display_settings_changed.emit(current_display_settings(self))

    def _emit_db_range(self) -> None:
        db_min = float(self.db_min_spin.value())
        db_max = float(self.db_max_spin.value())
        if db_min < db_max:
            self.db_range_changed.emit(db_min, db_max)

    def _emit_interpolation(self, checked: bool) -> None:
        self.interpolate_button.setText(
            "Interpolated (I)" if checked else "Interpolate Dataset (I)"
        )
        self.interpolation_changed.emit(
            bool(checked),
            float(self.interp_factor_spin.value()),
            float(self.aspect_ratio_spin.value()),
        )

    def _emit_active_interpolation_parameters(self) -> None:
        if self.interpolate_button.isChecked():
            self._emit_interpolation(True)

    def _on_group_toggled(self, layer_key: str, checked: bool) -> None:
        self._sync_layer_slot(layer_key)
        self.style_changed.emit(layer_key, "visible", bool(checked))
        self._update_toggle_button()

    def _emit_dataset_plot_map(self) -> None:
        map_name = self.dataset_plot_map_combo.currentData()
        if map_name:
            self.dataset_plot_map_changed.emit(str(map_name))

    def _on_dataset_plot_map_changed(self) -> None:
        map_name = self.dataset_plot_map_combo.currentData()
        if map_name in DENSITY_TRACK_MAPS:
            self._load_dataset_density_clim(str(map_name))
        self._update_dataset_plot_control_mode()
        self._emit_dataset_plot_map()

    def _update_dataset_plot_control_mode(self) -> None:
        map_name = self.dataset_plot_map_combo.currentData()
        is_density = map_name in DENSITY_TRACK_MAPS
        is_velocity = map_name in VELOCITY_TRACK_MAPS
        self.dataset_plot_density_clim_widget.setVisible(is_density)
        self.dataset_plot_velocity_clim_widget.setVisible(is_velocity)

    @staticmethod
    def _validate_dataset_density_clim(minimum: float, maximum: float) -> None:
        if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
            raise ValueError("Density color limits must be finite with minimum < maximum.")
        if minimum < 0.0 or maximum > 1_000_000_000.0:
            raise ValueError(
                "Density color limits must be between 0 and 1,000,000,000."
            )

    def _load_dataset_density_clim(self, map_name: str) -> None:
        minimum, maximum = self._dataset_density_clims[map_name]
        self._updating_dataset_density_clim = True
        try:
            self.dataset_plot_density_clim_min_spin.setValue(minimum)
            self.dataset_plot_density_clim_max_spin.setValue(maximum)
        finally:
            self._updating_dataset_density_clim = False
        self.dataset_plot_density_clim_changed.emit(map_name, minimum, maximum)

    def _set_dataset_density_clim(
        self,
        map_name: str,
        minimum: float,
        maximum: float,
    ) -> None:
        if map_name not in DENSITY_TRACK_MAPS:
            raise ValueError(f"Unsupported density map: {map_name}")
        minimum = float(minimum)
        maximum = float(maximum)
        self._validate_dataset_density_clim(minimum, maximum)
        self._dataset_density_clims[map_name] = (minimum, maximum)
        if self.dataset_plot_map_combo.currentData() == map_name:
            self._load_dataset_density_clim(map_name)
        else:
            self.dataset_plot_density_clim_changed.emit(map_name, minimum, maximum)

    def _emit_dataset_density_clim(self) -> None:
        if self._updating_dataset_density_clim:
            return
        map_name = self.dataset_plot_map_combo.currentData()
        if map_name not in DENSITY_TRACK_MAPS:
            return

        minimum = float(self.dataset_plot_density_clim_min_spin.value())
        maximum = float(self.dataset_plot_density_clim_max_spin.value())
        if minimum >= maximum:
            self._updating_dataset_density_clim = True
            try:
                if self.sender() is self.dataset_plot_density_clim_min_spin:
                    maximum = min(1_000_000_000.0, minimum + 0.0001)
                    self.dataset_plot_density_clim_max_spin.setValue(maximum)
                else:
                    minimum = max(0.0, maximum - 0.0001)
                    self.dataset_plot_density_clim_min_spin.setValue(minimum)
            finally:
                self._updating_dataset_density_clim = False
        if minimum < maximum:
            map_name = str(map_name)
            self._dataset_density_clims[map_name] = (minimum, maximum)
            self.dataset_plot_density_clim_changed.emit(map_name, minimum, maximum)

    def _set_dataset_plot_clim(self, minimum: float, maximum: float) -> None:
        minimum = float(minimum)
        maximum = float(maximum)
        if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if minimum < 0.0 or maximum > 1_000_000.0:
            raise ValueError("Velocity color limits must be between 0 and 1,000,000.")

        self._updating_dataset_plot_clim = True
        try:
            self.dataset_plot_clim_min_spin.setValue(minimum)
            self.dataset_plot_clim_max_spin.setValue(maximum)
        finally:
            self._updating_dataset_plot_clim = False
        self.dataset_plot_clim_changed.emit(
            float(self.dataset_plot_clim_min_spin.value()),
            float(self.dataset_plot_clim_max_spin.value()),
        )

    def _emit_dataset_plot_clim(self) -> None:
        if self._updating_dataset_plot_clim:
            return

        minimum = float(self.dataset_plot_clim_min_spin.value())
        maximum = float(self.dataset_plot_clim_max_spin.value())
        if minimum >= maximum:
            self._updating_dataset_plot_clim = True
            try:
                if self.sender() is self.dataset_plot_clim_min_spin:
                    maximum = min(1_000_000.0, minimum + 0.01)
                    self.dataset_plot_clim_max_spin.setValue(maximum)
                else:
                    minimum = max(0.0, maximum - 0.01)
                    self.dataset_plot_clim_min_spin.setValue(minimum)
            finally:
                self._updating_dataset_plot_clim = False
        if minimum < maximum:
            self.dataset_plot_clim_changed.emit(minimum, maximum)

    def _emit_multiple_dataset_plot_map(self) -> None:
        map_name = self.multiple_dataset_plot_map_combo.currentData()
        if map_name:
            self.multiple_dataset_plot_map_changed.emit(str(map_name))

    def _on_multiple_dataset_plot_map_changed(self) -> None:
        map_name = self.multiple_dataset_plot_map_combo.currentData()
        if map_name in DENSITY_TRACK_MAPS:
            self._load_multiple_dataset_density_clim(str(map_name))
        self._update_multiple_dataset_plot_control_mode()
        self._emit_multiple_dataset_plot_map()

    def _update_multiple_dataset_plot_control_mode(self) -> None:
        map_name = self.multiple_dataset_plot_map_combo.currentData()
        self.multiple_dataset_plot_density_clim_widget.setVisible(
            map_name in DENSITY_TRACK_MAPS
        )
        self.multiple_dataset_plot_velocity_clim_widget.setVisible(
            map_name in VELOCITY_TRACK_MAPS
        )

    def _load_multiple_dataset_density_clim(self, map_name: str) -> None:
        minimum, maximum = self._multiple_dataset_density_clims[map_name]
        self._updating_multiple_dataset_density_clim = True
        try:
            self.multiple_dataset_plot_density_clim_min_spin.setValue(minimum)
            self.multiple_dataset_plot_density_clim_max_spin.setValue(maximum)
        finally:
            self._updating_multiple_dataset_density_clim = False
        self.multiple_dataset_plot_density_clim_changed.emit(
            map_name,
            minimum,
            maximum,
        )

    def _set_multiple_dataset_density_clim(
        self,
        map_name: str,
        minimum: float,
        maximum: float,
    ) -> None:
        if map_name not in DENSITY_TRACK_MAPS:
            raise ValueError(f"Unsupported density map: {map_name}")
        minimum = float(minimum)
        maximum = float(maximum)
        self._validate_dataset_density_clim(minimum, maximum)
        self._multiple_dataset_density_clims[map_name] = (minimum, maximum)
        if self.multiple_dataset_plot_map_combo.currentData() == map_name:
            self._load_multiple_dataset_density_clim(map_name)
        else:
            self.multiple_dataset_plot_density_clim_changed.emit(
                map_name,
                minimum,
                maximum,
            )

    def _emit_multiple_dataset_density_clim(self) -> None:
        if self._updating_multiple_dataset_density_clim:
            return
        map_name = self.multiple_dataset_plot_map_combo.currentData()
        if map_name not in DENSITY_TRACK_MAPS:
            return

        minimum = float(self.multiple_dataset_plot_density_clim_min_spin.value())
        maximum = float(self.multiple_dataset_plot_density_clim_max_spin.value())
        if minimum >= maximum:
            self._updating_multiple_dataset_density_clim = True
            try:
                if self.sender() is self.multiple_dataset_plot_density_clim_min_spin:
                    maximum = min(1_000_000_000.0, minimum + 0.0001)
                    self.multiple_dataset_plot_density_clim_max_spin.setValue(maximum)
                else:
                    minimum = max(0.0, maximum - 0.0001)
                    self.multiple_dataset_plot_density_clim_min_spin.setValue(minimum)
            finally:
                self._updating_multiple_dataset_density_clim = False
        if minimum < maximum:
            map_name = str(map_name)
            self._multiple_dataset_density_clims[map_name] = (minimum, maximum)
            self.multiple_dataset_plot_density_clim_changed.emit(
                map_name,
                minimum,
                maximum,
            )

    def _set_multiple_dataset_plot_clim(
        self,
        minimum: float,
        maximum: float,
    ) -> None:
        minimum = float(minimum)
        maximum = float(maximum)
        if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if minimum < 0.0 or maximum > 1_000_000.0:
            raise ValueError("Velocity color limits must be between 0 and 1,000,000.")

        self._updating_multiple_dataset_plot_clim = True
        try:
            self.multiple_dataset_plot_clim_min_spin.setValue(minimum)
            self.multiple_dataset_plot_clim_max_spin.setValue(maximum)
        finally:
            self._updating_multiple_dataset_plot_clim = False
        self.multiple_dataset_plot_clim_changed.emit(
            float(self.multiple_dataset_plot_clim_min_spin.value()),
            float(self.multiple_dataset_plot_clim_max_spin.value()),
        )

    def _emit_multiple_dataset_plot_clim(self) -> None:
        if self._updating_multiple_dataset_plot_clim:
            return

        minimum = float(self.multiple_dataset_plot_clim_min_spin.value())
        maximum = float(self.multiple_dataset_plot_clim_max_spin.value())
        if minimum >= maximum:
            self._updating_multiple_dataset_plot_clim = True
            try:
                if self.sender() is self.multiple_dataset_plot_clim_min_spin:
                    maximum = min(1_000_000.0, minimum + 0.01)
                    self.multiple_dataset_plot_clim_max_spin.setValue(maximum)
                else:
                    minimum = max(0.0, maximum - 0.01)
                    self.multiple_dataset_plot_clim_min_spin.setValue(minimum)
            finally:
                self._updating_multiple_dataset_plot_clim = False
        if minimum < maximum:
            self.multiple_dataset_plot_clim_changed.emit(minimum, maximum)

    def _emit_track_plot_map(self) -> None:
        map_name = self.track_plot_map_combo.currentData()
        if map_name:
            self.track_plot_map_changed.emit(str(map_name))

    def _on_track_plot_map_changed(self) -> None:
        self._update_track_plot_control_mode()
        self._emit_track_plot_map()

    def _update_track_plot_control_mode(self) -> None:
        map_name = self.track_plot_map_combo.currentData()
        is_density = map_name in DENSITY_TRACK_MAPS
        is_velocity = map_name in VELOCITY_TRACK_MAPS
        self.track_plot_color_label.setVisible(is_density)
        self.track_plot_color_button.setVisible(is_density)
        self.track_plot_velocity_clim_widget.setVisible(is_velocity)

    def _set_track_plot_clim(self, minimum: float, maximum: float) -> None:
        minimum = float(minimum)
        maximum = float(maximum)
        if not isfinite(minimum) or not isfinite(maximum) or minimum >= maximum:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if minimum < 0.0 or maximum > 1_000_000.0:
            raise ValueError("Velocity color limits must be between 0 and 1,000,000.")

        self._updating_track_plot_clim = True
        try:
            self.track_plot_clim_min_spin.setValue(minimum)
            self.track_plot_clim_max_spin.setValue(maximum)
        finally:
            self._updating_track_plot_clim = False
        self.track_plot_clim_changed.emit(
            float(self.track_plot_clim_min_spin.value()),
            float(self.track_plot_clim_max_spin.value()),
        )

    def _emit_track_plot_clim(self) -> None:
        if self._updating_track_plot_clim:
            return

        minimum = float(self.track_plot_clim_min_spin.value())
        maximum = float(self.track_plot_clim_max_spin.value())
        if minimum >= maximum:
            self._updating_track_plot_clim = True
            try:
                if self.sender() is self.track_plot_clim_min_spin:
                    maximum = min(1_000_000.0, minimum + 0.01)
                    self.track_plot_clim_max_spin.setValue(maximum)
                else:
                    minimum = max(0.0, maximum - 0.01)
                    self.track_plot_clim_min_spin.setValue(minimum)
            finally:
                self._updating_track_plot_clim = False
        if minimum < maximum:
            self.track_plot_clim_changed.emit(minimum, maximum)

    def configure_track_plotting(
        self,
        supported_track_maps: Iterable[str],
        message: str,
        velocity_display_max: float | None = None,
    ) -> None:
        """Enable only map choices actually present in the loaded checkpoint."""

        available = {str(map_name) for map_name in supported_track_maps}
        model = self.track_plot_map_combo.model()
        for index in range(self.track_plot_map_combo.count()):
            map_name = self.track_plot_map_combo.itemData(index)
            item = model.item(index)
            if item is not None:
                item.setEnabled(map_name in available)
                item.setToolTip(
                    "" if map_name in available else "Unavailable in the loaded checkpoint."
                )
        if self.track_plot_map_combo.currentData() not in available and available:
            preferred = (
                AA_DENSITY
                if AA_DENSITY in available
                else next(map_name for map_name in FULL_TRACK_MAPS if map_name in available)
            )
            self.track_plot_map_combo.setCurrentIndex(
                self.track_plot_map_combo.findData(preferred)
            )
        if velocity_display_max is not None:
            self._set_track_plot_clim(0.0, float(velocity_display_max))
        self._update_track_plot_control_mode()
        for widget in self._track_plot_parameter_widgets:
            widget.setEnabled(bool(available))
        self.track_plot_group.setVisibilityControlEnabled(bool(available))
        self._sync_layer_slot("track_plot")
        self.set_track_plot_status(message)

    def configure_dataset_plotting(
        self,
        available: bool,
        message: str,
    ) -> None:
        """Enable dynamic all-track plotting when exact PALA settings exist."""

        if available:
            self._set_dataset_density_clim(
                HARD_DENSITY,
                *DEFAULT_ACCUMULATION_CLIM,
            )
            self._set_dataset_density_clim(
                AA_DENSITY,
                *DEFAULT_ACCUMULATION_CLIM,
            )
            self._set_dataset_plot_clim(*DEFAULT_ACCUMULATION_CLIM)
        self._update_dataset_plot_control_mode()
        for widget in self._dataset_parameter_widgets:
            widget.setEnabled(bool(available))
        self.dataset_plot_group.setVisibilityControlEnabled(bool(available))
        self._sync_layer_slot("dataset_plot")
        self.set_dataset_plot_status(message)

    def configure_multiple_dataset_plotting(
        self,
        available: bool,
        message: str,
    ) -> None:
        """Configure the retained snapshot without disabling its import button."""

        if available:
            self._set_multiple_dataset_density_clim(
                HARD_DENSITY,
                *DEFAULT_ACCUMULATION_CLIM,
            )
            self._set_multiple_dataset_density_clim(
                AA_DENSITY,
                *DEFAULT_ACCUMULATION_CLIM,
            )
            self._set_multiple_dataset_plot_clim(*DEFAULT_ACCUMULATION_CLIM)
        self._update_multiple_dataset_plot_control_mode()
        # The import action lives inside this layer card and must remain usable
        # before the first retained snapshot exists. Disable only parameters
        # that require snapshot data, not the section or its import button.
        for widget in self._multiple_dataset_parameter_widgets:
            widget.setEnabled(bool(available))
        self.set_multiple_dataset_plot_status(message)

    def set_track_plot_status(self, message: str) -> None:
        self.track_plot_status_label.setText(str(message))
        self.track_plot_group.refresh_minimum_height()

    def set_dataset_plot_status(self, message: str) -> None:
        self.dataset_plot_status_label.setText(str(message))
        self.dataset_plot_group.refresh_minimum_height()

    def set_multiple_dataset_plot_status(self, message: str) -> None:
        self.multiple_dataset_plot_status_label.setText(str(message))
        self.multiple_dataset_plot_group.refresh_minimum_height()

    def _update_toggle_button(self) -> None:
        any_visible = any(
            self._layer_groups[key].isChecked() for key in OVERLAY_LAYERS
        )
        self.toggle_button.setChecked(not any_visible)
        self.toggle_button.setText(
            "Keep images only (T)" if any_visible else "Restore layers (T)"
        )

    # ------------------------------------------------------------- actions

    def toggle_overlays(self) -> None:
        """Hide or restore every non-image layer, regardless of stack order.

        The Ultrasound Movie number-toggle state is independent. Hiding remembers
        which layers were visible; showing restores exactly that set (or all non-image
        layers if nothing was remembered). Other visual settings are untouched.
        """

        visible_now = [
            key for key in OVERLAY_LAYERS if self._layer_groups[key].isChecked()
        ]
        if visible_now:
            self._overlay_memory = visible_now
            for key in visible_now:
                self._layer_groups[key].setChecked(False)
        else:
            for key in self._overlay_memory or list(OVERLAY_LAYERS):
                self._layer_groups[key].setChecked(True)

    def toggle_all_lines(self) -> None:
        group = self._layer_groups["all_lines"]
        group.setChecked(not group.isChecked())
