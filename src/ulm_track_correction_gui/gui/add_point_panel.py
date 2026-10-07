"""Inline Add Point settings for automatic per-frame candidate localization."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ulm_track_correction_gui.reference_processing.wrappers import (
    FS_GRADIENT_NCC_METHOD,
    NCC_BASED_METHODS,
    RADIAL_NCC_METHOD,
    RADIAL_NON_NCC_METHOD,
)

MANUAL_PLACEMENT_METHOD = "manual_placement"
MANUAL_NUDGE_STEP_PX = 0.1
MANUAL_PLACEMENT_HINT = (
    "Manual Placement: click to place or move the draft. "
    "Nudge with Shift+W/A/S/D (0.1 px). "
    "Enter confirms; Backspace clears; Esc exits."
)


class _AdaptiveWordWrapLabel(QLabel):
    """Keep a wrapped label tall enough for its actual narrow-panel width."""

    def _sync_minimum_height(self, width: int) -> None:
        required = self.heightForWidth(max(1, int(width)))
        if required > 0 and self.minimumHeight() != required:
            self.setMinimumHeight(required)
            self.updateGeometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_minimum_height(event.size().width())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_minimum_height(self.width())


class AddPointPanel(QFrame):
    """Collapsible parameters and candidate-marker styling for Add Point.

    Localization runs are owned by the main-window editing workflow. This
    widget only keeps persistent UI settings and reports when they change.
    """

    parameters_changed = Signal()
    style_changed = Signal(dict)
    eraser_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("addPointPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame#addPointPanel { border: 1px solid palette(mid); border-radius: 4px; }"
        )
        self._candidate_color = QColor("#00dcff")
        self._build_ui()
        self._connect_signals()
        self._sync_method_page()
        self._sync_collapsed_state(True)
        self._update_color_button()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 5)
        outer.setSpacing(4)

        self.header_button = QToolButton()
        self.header_button.setText("Candidate settings")
        self.header_button.setCheckable(True)
        self.header_button.setChecked(True)
        self.header_button.setArrowType(Qt.RightArrow)
        self.header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header_button.setAutoRaise(True)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.header_button)
        header.addStretch(1)
        outer.addLayout(header)

        self.status_label = QLabel("Ready · candidates appear automatically in Add Point mode.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("addPointStatus")
        self.status_label.setStyleSheet(
            "QLabel#addPointStatus { color: palette(window-text); font-size: 11px; }"
        )
        outer.addWidget(self.status_label)

        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 3, 0, 0)
        content_layout.setSpacing(6)

        self.eraser_button = QPushButton("Erase unassigned added point")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setToolTip(
            "Click a visible Add Point cross to exclude that detection. "
            "Assigned or original localization points cannot be erased."
        )
        self.eraser_button.setStyleSheet(
            "QPushButton:checked { background-color: #d97706; color: white; }"
        )

        self.method_combo = QComboBox()
        self.method_combo.addItem(
            "Radial Symmetry (non-NCC)",
            RADIAL_NON_NCC_METHOD,
        )
        self.method_combo.addItem(
            "FS_Gradient (NCC)",
            FS_GRADIENT_NCC_METHOD,
        )
        self.method_combo.addItem(
            "Radial Symmetry (NCC)",
            RADIAL_NCC_METHOD,
        )
        self.method_combo.addItem("Manual Placement", MANUAL_PLACEMENT_METHOD)
        self.method_combo.setCurrentIndex(
            self.method_combo.findData(RADIAL_NCC_METHOD)
        )
        self.method_combo.setToolTip(
            "While Add Point is active, press , for the previous method or . for the next."
        )
        method_form = QFormLayout()
        method_form.setContentsMargins(0, 0, 0, 0)
        method_form.addRow("Method", self.method_combo)
        content_layout.addLayout(method_form)

        self.manual_hint_label = _AdaptiveWordWrapLabel(MANUAL_PLACEMENT_HINT)
        self.manual_hint_label.setWordWrap(True)
        self.manual_hint_label.setObjectName("manualPlacementHint")
        self.manual_hint_label.setStyleSheet(
            "QLabel#manualPlacementHint {"
            " background-color: rgba(0, 160, 220, 28);"
            " border: 1px solid rgba(0, 160, 220, 90);"
            " border-radius: 3px; padding: 5px;"
            "}"
        )
        content_layout.addWidget(self.manual_hint_label)

        self.parameter_stack = QStackedWidget()
        self._build_radial_page()
        self._build_ncc_page()
        content_layout.addWidget(self.parameter_stack)

        self.marker_style_label = QLabel("Candidate marker")
        self.marker_style_label.setStyleSheet("font-weight: 600;")
        content_layout.addWidget(self.marker_style_label)
        style_grid = QGridLayout()
        style_grid.setContentsMargins(0, 0, 0, 0)
        style_grid.setHorizontalSpacing(5)
        style_grid.setVerticalSpacing(4)

        self.marker_combo = QComboBox()
        for label, key in (
            ("Target", "target"),
            ("Ring", "ring"),
            ("Dot", "dot"),
            ("Cross", "cross"),
        ):
            self.marker_combo.addItem(label, key)
        self.color_button = QPushButton()
        self.color_button.setAccessibleName("Candidate marker color")
        self.color_button.setToolTip("Choose the candidate marker color")
        self.color_button.setFixedWidth(42)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 10.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(1.0)
        self.radius_spin.setSuffix(" px")
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(10, 100)
        self.opacity_spin.setValue(75)
        self.opacity_spin.setSuffix("%")

        style_grid.addWidget(QLabel("Shape"), 0, 0)
        style_grid.addWidget(self.marker_combo, 0, 1)
        style_grid.addWidget(QLabel("Color"), 0, 2)
        style_grid.addWidget(self.color_button, 0, 3)
        style_grid.addWidget(QLabel("Radius"), 1, 0)
        style_grid.addWidget(self.radius_spin, 1, 1)
        style_grid.addWidget(QLabel("Opacity"), 1, 2)
        style_grid.addWidget(self.opacity_spin, 1, 3)
        style_grid.setColumnStretch(1, 1)
        style_grid.setColumnStretch(3, 1)
        content_layout.addLayout(style_grid)
        content_layout.addSpacing(2)
        content_layout.addWidget(self.eraser_button)
        outer.addWidget(self.content)

    def _build_radial_page(self) -> None:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.num_mb_spin = QSpinBox()
        self.num_mb_spin.setRange(1, 10000)
        self.num_mb_spin.setValue(90)
        self.num_mb_spin.setToolTip("Estimated number of microbubbles per frame")
        self.num_local_max_spin = QSpinBox()
        self.num_local_max_spin.setRange(1, 100)
        self.num_local_max_spin.setValue(3)
        self.num_local_max_spin.setToolTip(
            "Maximum local maxima in the FWHM-sized neighborhood"
        )
        self.fwhm_z_spin = QSpinBox()
        self.fwhm_z_spin.setRange(1, 99)
        self.fwhm_z_spin.setValue(3)
        self.fwhm_x_spin = QSpinBox()
        self.fwhm_x_spin.setRange(1, 99)
        self.fwhm_x_spin.setValue(3)
        form.addRow("Num. MBs / frame", self.num_mb_spin)
        form.addRow("Num. local maxima", self.num_local_max_spin)
        form.addRow("FWHM z-axis", self.fwhm_z_spin)
        form.addRow("FWHM x-axis", self.fwhm_x_spin)
        self.parameter_stack.addWidget(page)

    def _build_ncc_page(self) -> None:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.psf_z_spin = QDoubleSpinBox()
        self.psf_z_spin.setRange(0.5, 99.0)
        self.psf_z_spin.setValue(3.0)
        self.psf_x_spin = QDoubleSpinBox()
        self.psf_x_spin.setRange(0.5, 99.0)
        self.psf_x_spin.setValue(4.5)
        self.psf_window_spin = QSpinBox()
        self.psf_window_spin.setRange(3, 31)
        self.psf_window_spin.setSingleStep(2)
        self.psf_window_spin.setValue(5)
        self.psf_window_spin.setToolTip("PSF kernel size in pixels (odd)")
        self.amp_threshold_spin = QDoubleSpinBox()
        self.amp_threshold_spin.setRange(0.0, 1.0)
        self.amp_threshold_spin.setSingleStep(0.05)
        self.amp_threshold_spin.setValue(0.10)
        self.corr_threshold_spin = QDoubleSpinBox()
        self.corr_threshold_spin.setRange(0.0, 1.0)
        self.corr_threshold_spin.setSingleStep(0.05)
        self.corr_threshold_spin.setValue(0.40)
        self.filter_size_combo = QComboBox()
        self.filter_size_combo.addItems(["5", "7"])
        self.filter_size_combo.setCurrentText("7")
        form.addRow("PSF size z-axis", self.psf_z_spin)
        form.addRow("PSF size x-axis", self.psf_x_spin)
        form.addRow("PSF window size", self.psf_window_spin)
        form.addRow("Amplitude threshold", self.amp_threshold_spin)
        form.addRow("Correlation threshold", self.corr_threshold_spin)
        self.filter_size_label = QLabel("Filter size")
        form.addRow(self.filter_size_label, self.filter_size_combo)
        self.parameter_stack.addWidget(page)

    def _connect_signals(self) -> None:
        self.header_button.toggled.connect(self._sync_collapsed_state)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        for spin in (
            self.num_mb_spin,
            self.num_local_max_spin,
            self.fwhm_z_spin,
            self.fwhm_x_spin,
            self.psf_z_spin,
            self.psf_x_spin,
            self.psf_window_spin,
            self.amp_threshold_spin,
            self.corr_threshold_spin,
        ):
            spin.valueChanged.connect(lambda _value: self.parameters_changed.emit())
        self.filter_size_combo.currentIndexChanged.connect(
            lambda _index: self.parameters_changed.emit()
        )
        self.marker_combo.currentIndexChanged.connect(
            lambda _index: self.style_changed.emit(self.candidate_style())
        )
        self.radius_spin.valueChanged.connect(
            lambda _value: self.style_changed.emit(self.candidate_style())
        )
        self.opacity_spin.valueChanged.connect(
            lambda _value: self.style_changed.emit(self.candidate_style())
        )
        self.color_button.clicked.connect(self._choose_color)
        self.eraser_button.toggled.connect(self._on_eraser_toggled)

    def _sync_collapsed_state(self, expanded: bool) -> None:
        self.content.setVisible(bool(expanded))
        self.header_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _on_method_changed(self, _index: int) -> None:
        self._sync_method_page()
        self.parameters_changed.emit()

    def _on_eraser_toggled(self, active: bool) -> None:
        self._sync_method_page()
        self.eraser_toggled.emit(bool(active))

    def _sync_method_page(self) -> None:
        method = self.current_method()
        eraser_active = self.eraser_active()
        is_manual = method == MANUAL_PLACEMENT_METHOD
        is_ncc = method.lower() in NCC_BASED_METHODS
        is_fs_gradient = method == FS_GRADIENT_NCC_METHOD
        self.method_combo.setEnabled(not eraser_active)
        self.manual_hint_label.setVisible(is_manual)
        self.manual_hint_label.setEnabled(not eraser_active)
        self.parameter_stack.setVisible(not is_manual)
        self.parameter_stack.setEnabled(not eraser_active)
        self.parameter_stack.setCurrentIndex(1 if is_ncc else 0)
        self.filter_size_label.setVisible(is_fs_gradient)
        self.filter_size_combo.setVisible(is_fs_gradient)
        self.marker_style_label.setEnabled(not eraser_active)
        for widget in (
            self.marker_combo,
            self.color_button,
            self.radius_spin,
            self.opacity_spin,
        ):
            widget.setEnabled(not eraser_active)

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(
            self._candidate_color,
            self,
            "Candidate marker color",
        )
        if not color.isValid() or color == self._candidate_color:
            return
        self._candidate_color = color
        self._update_color_button()
        self.style_changed.emit(self.candidate_style())

    def _update_color_button(self) -> None:
        color = self._candidate_color.name(QColor.HexRgb)
        self.color_button.setText("")
        self.color_button.setStyleSheet(
            f"QPushButton {{ background-color: {color}; border: 1px solid palette(mid); }}"
        )

    def current_method(self) -> str:
        return str(self.method_combo.currentData())

    def eraser_active(self) -> bool:
        return self.eraser_button.isChecked()

    def set_eraser_active(self, active: bool, *, notify: bool = True) -> None:
        previous = self.eraser_button.blockSignals(not notify)
        try:
            self.eraser_button.setChecked(bool(active))
        finally:
            self.eraser_button.blockSignals(previous)
        self._sync_method_page()

    def current_params(self) -> dict:
        method = self.current_method()
        if method == MANUAL_PLACEMENT_METHOD:
            return {
                "method": method,
                "nudgeStepPx": MANUAL_NUDGE_STEP_PX,
            }
        if method.lower() in NCC_BASED_METHODS:
            params = {
                "method": method,
                "psfSizeZAxis": float(self.psf_z_spin.value()),
                "psfSizeXAxis": float(self.psf_x_spin.value()),
                "psfWindowSize": int(self.psf_window_spin.value()),
                "ampThreshold": float(self.amp_threshold_spin.value()),
                "corrThreshold": float(self.corr_threshold_spin.value()),
            }
            if method == FS_GRADIENT_NCC_METHOD:
                params["filterSize"] = int(self.filter_size_combo.currentText())
            return params
        return {
            "method": method,
            "fwhmXAxis": int(self.fwhm_x_spin.value()),
            "fwhmZAxis": int(self.fwhm_z_spin.value()),
            "numMB": int(self.num_mb_spin.value()),
            "numLocalMax": int(self.num_local_max_spin.value()),
            "filterSize": int(self.filter_size_combo.currentText()),
        }

    def candidate_style(self) -> dict:
        return {
            "marker": str(self.marker_combo.currentData()),
            "color": self._candidate_color.name(QColor.HexRgb),
            "radius": float(self.radius_spin.value()),
            "opacity": float(self.opacity_spin.value()) / 100.0,
        }

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(str(text))
        if error:
            self.status_label.setStyleSheet(
                "QLabel#addPointStatus { color: #d9534f; font-size: 11px; }"
            )
        else:
            self.status_label.setStyleSheet(
                "QLabel#addPointStatus { color: palette(window-text); font-size: 11px; }"
            )
