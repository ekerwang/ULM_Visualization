"""Shared Qt controls for dB and Power Law image compression."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ulm_track_correction_gui.gui.display_compression import (
    DEFAULT_DISPLAY_MODE,
    DEFAULT_POWER_CLIM_MAX,
    DEFAULT_POWER_CLIM_MIN,
    DEFAULT_POWER_GAMMA,
    DISPLAY_MODE_DB,
    DISPLAY_MODE_POWER,
    DisplaySettings,
)
from ulm_track_correction_gui.gui.layer_styles import DEFAULT_DB_MAX, DEFAULT_DB_MIN


class DoubleRangeSlider(QWidget):
    """Compact two-handle slider representing a floating-point interval."""

    rangeChanged = Signal(float, float)

    _HANDLE_RADIUS = 7
    _TRACK_MARGIN = 9

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lower = 0
        self._upper = 1000
        self._active_handle = "lower"
        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Power Law display Clim range")

    @staticmethod
    def _to_float(value: int) -> float:
        return value / 1000.0

    @staticmethod
    def _to_int(value: float) -> int:
        return max(0, min(1000, int(round(float(value) * 1000.0))))

    def values(self) -> tuple[float, float]:
        return self._to_float(self._lower), self._to_float(self._upper)

    def setValues(self, lower: float, upper: float) -> None:
        lower_i = self._to_int(lower)
        upper_i = self._to_int(upper)
        if upper_i <= lower_i:
            upper_i = min(1000, lower_i + 1)
            lower_i = max(0, upper_i - 1)
        if (lower_i, upper_i) == (self._lower, self._upper):
            return
        self._lower, self._upper = lower_i, upper_i
        self.update()
        self.rangeChanged.emit(*self.values())

    def _x_for_value(self, value: int) -> float:
        width = max(1, self.width() - 2 * self._TRACK_MARGIN)
        return self._TRACK_MARGIN + width * value / 1000.0

    def _value_for_x(self, x: float) -> int:
        width = max(1, self.width() - 2 * self._TRACK_MARGIN)
        ratio = (x - self._TRACK_MARGIN) / width
        return max(0, min(1000, int(round(ratio * 1000.0))))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        center_y = self.height() / 2.0
        left = self._x_for_value(0)
        right = self._x_for_value(1000)
        lower_x = self._x_for_value(self._lower)
        upper_x = self._x_for_value(self._upper)

        painter.setPen(QPen(palette.mid().color(), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(left, center_y), QPointF(right, center_y))
        painter.setPen(QPen(palette.highlight().color(), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(QPointF(lower_x, center_y), QPointF(upper_x, center_y))

        painter.setPen(QPen(palette.highlight().color(), 1.5))
        painter.setBrush(palette.base())
        for handle_x in (lower_x, upper_x):
            painter.drawEllipse(
                QPointF(handle_x, center_y),
                self._HANDLE_RADIUS,
                self._HANDLE_RADIUS,
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        lower_distance = abs(event.position().x() - self._x_for_value(self._lower))
        upper_distance = abs(event.position().x() - self._x_for_value(self._upper))
        self._active_handle = "lower" if lower_distance <= upper_distance else "upper"
        self._move_active_handle(event.position().x())
        self.setFocus()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.LeftButton:
            self._move_active_handle(event.position().x())

    def _move_active_handle(self, x: float) -> None:
        value = self._value_for_x(x)
        if self._active_handle == "lower":
            value = min(value, self._upper - 1)
            self.setValues(self._to_float(value), self._to_float(self._upper))
        else:
            value = max(value, self._lower + 1)
            self.setValues(self._to_float(self._lower), self._to_float(value))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() not in (Qt.Key_Left, Qt.Key_Right):
            super().keyPressEvent(event)
            return
        step = -10 if event.key() == Qt.Key_Left else 10
        current = self._lower if self._active_handle == "lower" else self._upper
        self._move_active_handle(self._x_for_value(current + step))


class ClimRangeControl(QWidget):
    """Synchronized slider and numeric editors for a normalized Clim."""

    rangeChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider = DoubleRangeSlider(self)
        self.minimum_spin = self._spin(DEFAULT_POWER_CLIM_MIN)
        self.maximum_spin = self._spin(DEFAULT_POWER_CLIM_MAX)
        self.slider.setValues(DEFAULT_POWER_CLIM_MIN, DEFAULT_POWER_CLIM_MAX)
        self.minimum_spin.setAccessibleName("Power Law Clim minimum")
        self.minimum_spin.setToolTip("Clim minimum (black point)")
        self.maximum_spin.setAccessibleName("Power Law Clim maximum")
        self.maximum_spin.setToolTip("Clim maximum (white point)")

        self.slider.rangeChanged.connect(self._from_slider)
        self.minimum_spin.valueChanged.connect(self._from_spins)
        self.maximum_spin.valueChanged.connect(self._from_spins)

        values = QHBoxLayout()
        values.setContentsMargins(0, 0, 0, 0)
        values.setSpacing(4)
        values.addWidget(self.minimum_spin)
        values.addStretch(1)
        values.addWidget(self.maximum_spin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.slider)
        layout.addLayout(values)

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setFixedWidth(60)
        return spin

    def values(self) -> tuple[float, float]:
        return self.slider.values()

    def _from_slider(self, lower: float, upper: float) -> None:
        for spin, value in ((self.minimum_spin, lower), (self.maximum_spin, upper)):
            previous = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(previous)
        self.rangeChanged.emit(lower, upper)

    def _from_spins(self, _value: float) -> None:
        lower = float(self.minimum_spin.value())
        upper = float(self.maximum_spin.value())
        if upper <= lower:
            if self.sender() is self.minimum_spin:
                lower = max(0.0, upper - 0.001)
                self.minimum_spin.setValue(lower)
            else:
                upper = min(1.0, lower + 0.001)
                self.maximum_spin.setValue(upper)
        self.slider.setValues(lower, upper)


def create_dynamic_range_panel(owner, refresh_callback) -> QGroupBox:
    """Create mutually exclusive dB and Power Law display controls."""

    owner.db_min_spin = QDoubleSpinBox()
    owner.db_min_spin.setRange(-200.0, 0.0)
    owner.db_min_spin.setValue(DEFAULT_DB_MIN)
    owner.db_min_spin.setKeyboardTracking(False)

    owner.db_max_spin = QDoubleSpinBox()
    owner.db_max_spin.setRange(-200.0, 50.0)
    owner.db_max_spin.setValue(DEFAULT_DB_MAX)
    owner.db_max_spin.setKeyboardTracking(False)

    owner.power_gamma_spin = QDoubleSpinBox()
    owner.power_gamma_spin.setRange(0.0, 1.0)
    owner.power_gamma_spin.setDecimals(2)
    owner.power_gamma_spin.setSingleStep(0.05)
    owner.power_gamma_spin.setValue(DEFAULT_POWER_GAMMA)
    owner.power_gamma_spin.setKeyboardTracking(False)
    owner.power_gamma_spin.setToolTip(
        "Power Law exponent in (|I| / ensemble reference)^gamma"
    )

    owner.power_clim_control = ClimRangeControl()
    owner.power_clim_control.setToolTip(
        "Black and white display limits after Power Law compression"
    )

    owner.db_compression_radio = QRadioButton("dB scale")
    owner.power_compression_radio = QRadioButton("Power law")
    owner.compression_button_group = QButtonGroup(owner)
    owner.compression_button_group.setExclusive(True)
    owner.compression_button_group.addButton(owner.db_compression_radio)
    owner.compression_button_group.addButton(owner.power_compression_radio)

    mode_row = QHBoxLayout()
    mode_row.setContentsMargins(0, 0, 0, 0)
    mode_row.addWidget(owner.db_compression_radio)
    mode_row.addWidget(owner.power_compression_radio)

    owner.db_min_spin.setMaximumWidth(72)
    owner.db_max_spin.setMaximumWidth(72)
    db_widget = QWidget()
    db_row = QHBoxLayout(db_widget)
    db_row.setContentsMargins(0, 0, 0, 0)
    db_row.setSpacing(4)
    db_row.addWidget(QLabel("Min"))
    db_row.addWidget(owner.db_min_spin)
    db_row.addStretch(1)
    db_row.addWidget(QLabel("Max"))
    db_row.addWidget(owner.db_max_spin)

    power_widget = QWidget()
    power_layout = QVBoxLayout(power_widget)
    power_layout.setContentsMargins(0, 0, 0, 0)
    power_layout.setSpacing(3)
    gamma_row = QHBoxLayout()
    gamma_row.setContentsMargins(0, 0, 0, 0)
    gamma_row.addWidget(QLabel("Gamma"))
    gamma_row.addStretch(1)
    owner.power_gamma_spin.setMaximumWidth(72)
    gamma_row.addWidget(owner.power_gamma_spin)
    power_layout.addLayout(gamma_row)
    power_layout.addWidget(owner.power_clim_control)

    owner.compression_options_stack = QStackedWidget()
    owner.compression_options_stack.addWidget(db_widget)
    owner.compression_options_stack.addWidget(power_widget)

    panel = QGroupBox("Dynamic range")
    panel.setObjectName("dynamicRangePanel")
    panel.setToolTip("Compression and display limits for the bubble movie")
    panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    panel.setStyleSheet("QGroupBox#dynamicRangePanel { font-weight: 600; }")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(7, 12, 7, 6)
    layout.setSpacing(4)
    layout.addLayout(mode_row)
    layout.addWidget(owner.compression_options_stack)

    power_is_default = DEFAULT_DISPLAY_MODE == DISPLAY_MODE_POWER
    owner.db_compression_radio.setChecked(not power_is_default)
    owner.power_compression_radio.setChecked(power_is_default)
    owner.compression_options_stack.setCurrentIndex(1 if power_is_default else 0)

    owner.db_min_spin.valueChanged.connect(refresh_callback)
    owner.db_max_spin.valueChanged.connect(refresh_callback)
    owner.power_gamma_spin.valueChanged.connect(refresh_callback)
    owner.power_clim_control.rangeChanged.connect(refresh_callback)

    def select_mode(power_checked: bool) -> None:
        owner.compression_options_stack.setCurrentIndex(1 if power_checked else 0)
        refresh_callback()

    owner.power_compression_radio.toggled.connect(select_mode)
    return panel


def current_display_settings(owner) -> DisplaySettings:
    """Read all dynamic-range widgets into immutable display state."""

    clim_min, clim_max = owner.power_clim_control.values()
    return DisplaySettings(
        mode=(
            DISPLAY_MODE_POWER
            if owner.power_compression_radio.isChecked()
            else DISPLAY_MODE_DB
        ),
        db_min=float(owner.db_min_spin.value()),
        db_max=float(owner.db_max_spin.value()),
        gamma=float(owner.power_gamma_spin.value()),
        clim_min=clim_min,
        clim_max=clim_max,
    )
