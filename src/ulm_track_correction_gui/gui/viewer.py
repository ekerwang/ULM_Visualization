"""QGraphicsScene-based PALA stack viewer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)
from skimage.transform import resize

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    DENSITY_TRACK_MAPS,
    HARD_DENSITY,
    TRACK_MAP_LABELS,
    VELOCITY_TRACK_MAPS,
    AccumulationCheckpoint,
    TrackMapUnavailableError,
)
from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_FLAGGED,
    TRACK_STATUS_VERIFIED,
    CorrectionSession,
)
from ulm_track_correction_gui.core.dataset_accumulation import DatasetAccumulator
from ulm_track_correction_gui.core.pala_contracts import (
    ensure_mat_tracking,
    pala_to_scene_xy,
    scene_to_pala_zx,
)
from ulm_track_correction_gui.gui.display_compression import (
    DisplaySettings,
    display_references,
    normalize_display_frame,
)
from ulm_track_correction_gui.gui.layer_styles import LayerStyle, default_layer_styles
from ulm_track_correction_gui.gui.layer_order import (
    DEFAULT_LAYER_ORDER,
    LAYER_LABELS,
    z_values_for_order,
)
from ulm_track_correction_gui.gui.movie_export import (
    MovieWatermark,
    MovieWatermarkPreview,
)
from ulm_track_correction_gui.gui.track_plot_rendering import (
    DEFAULT_ACCUMULATION_GAMMA,
    render_sparse_dataset_map,
    render_sparse_track_map,
)


def interpolate_frame_for_display(
    frame: np.ndarray,
    factor: float,
    aspect_ratio: float,
) -> np.ndarray:
    """Cubic-resample one frame for display without modifying session data."""

    frame = np.asarray(frame)
    factor = float(factor)
    aspect_ratio = float(aspect_ratio)
    if frame.ndim != 2:
        raise ValueError("Display interpolation requires a 2-D image frame.")
    if not np.isfinite(factor) or factor < 1.0:
        raise ValueError("Interpolation factor must be finite and at least 1.0.")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("Aspect ratio must be finite and positive.")

    source = np.abs(frame).astype(np.float32, copy=False)
    output_shape = (
        max(1, int(source.shape[0] * factor * aspect_ratio)),
        max(1, int(source.shape[1] * factor)),
    )
    if output_shape == source.shape:
        return source.copy()
    return resize(
        source,
        output_shape,
        order=3,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32, copy=False)


def gray_to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img.astype(np.uint8))
    height, width = img.shape
    qimg = QImage(
        img.data,
        width,
        height,
        img.strides[0],
        QImage.Format_Grayscale8,
    )
    return QPixmap.fromImage(qimg.copy())


def rgba_to_qpixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img.astype(np.uint8, copy=False))
    height, width, channels = img.shape
    if channels != 4:
        raise ValueError("RGBA image must have four channels")
    qimg = QImage(
        img.data,
        width,
        height,
        img.strides[0],
        QImage.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())


def _styled_color(style: LayerStyle, alpha_scale: float = 1.0) -> QColor:
    color = QColor(style.color)
    color.setAlpha(int(round(255 * max(0.0, min(1.0, style.opacity * alpha_scale)))))
    return color


# Public constants describe the reviewed default stack. Runtime drag reordering
# updates the viewer's per-layer map without changing these defaults.
_DEFAULT_LAYER_Z = z_values_for_order(DEFAULT_LAYER_ORDER)
IMAGE_Z = _DEFAULT_LAYER_Z["image"]
SELECTED_POINT_Z = _DEFAULT_LAYER_Z["track_point"]
ALL_LINES_Z = _DEFAULT_LAYER_Z["all_lines"]
TRACK_PATH_Z = _DEFAULT_LAYER_Z["track_path"]
LOCALIZATION_Z = _DEFAULT_LAYER_Z["points"]
DATASET_PLOT_Z = _DEFAULT_LAYER_Z["dataset_plot"]
MULTIPLE_DATASET_PLOT_Z = _DEFAULT_LAYER_Z["multiple_dataset_plot"]
TRACK_PLOT_Z = _DEFAULT_LAYER_Z["track_plot"]
CANDIDATE_Z = 1_000_000.0


class PalaSceneViewer(QGraphicsView):
    """Viewer with a reorderable visual layer stack for the correction workflow.

    Zoom (mouse wheel) and pan (drag) are preserved across frame steps and
    track switches; the view only auto-fits when a new session is loaded.
    All layer visuals (visibility, opacity, color, point radius, line width)
    come from ``self.styles`` and are adjustable at runtime via
    ``update_style``.
    """

    detection_clicked = Signal(int, int)  # frame_idx, local_idx
    candidate_clicked = Signal(int, int)  # frame_idx, candidate_idx
    manual_candidate_placed = Signal(int, float, float)  # frame_idx, z, x
    manual_detection_erase_requested = Signal(int, int)  # frame_idx, local_idx
    track_picked = Signal(int)  # track_id (Inspect click or any-mode double-click)
    track_selection_cleared = Signal()  # Inspect click on empty image space
    dataset_plot_status_changed = Signal(str)
    multiple_dataset_plot_status_changed = Signal(str)
    track_plot_status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setFocusPolicy(Qt.StrongFocus)

        self.manual_mode_badge = QLabel(self.viewport())
        self.manual_mode_badge.setObjectName("manualModeBadge")
        self.manual_mode_badge.setAccessibleName("Manual Placement status")
        self.manual_mode_badge.setAlignment(Qt.AlignCenter)
        self.manual_mode_badge.setWordWrap(True)
        self.manual_mode_badge.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.manual_mode_badge.hide()
        self.movie_watermark_preview = MovieWatermarkPreview(self.viewport())
        self.movie_watermark_preview.setGeometry(self.viewport().rect())

        self.session: CorrectionSession | None = None
        self.interaction_mode = "inspect"
        self.frame_idx = 0
        self.selected_track_id: int | None = None
        self.display_settings = DisplaySettings()
        self.db_min = self.display_settings.db_min
        self.db_max = self.display_settings.db_max
        self.interpolation_enabled = False
        self.interpolation_factor = 5.0
        self.interpolation_aspect_ratio = 1.0
        self.styles: dict[str, LayerStyle] = default_layer_styles()
        self._layer_order = tuple(DEFAULT_LAYER_ORDER)
        self._layer_z = z_values_for_order(self._layer_order)

        self._stack_max: float | None = None
        self._power_reference: float | None = None
        self._image_cache_key: tuple | None = None
        self._interpolated_image_cache_key: tuple | None = None
        self._interpolated_image_pixmap = QPixmap()
        self._did_initial_fit = False
        self._track_geometry: list[dict] | None = None
        self.accumulation_checkpoint: AccumulationCheckpoint | None = None
        self._checkpoint_track_id_map: dict[int, int] | None = None
        self.dataset_accumulator: DatasetAccumulator | None = None
        self.dataset_plot_map_name = AA_DENSITY
        self.dataset_plot_density_clims = {
            HARD_DENSITY: (0.0, 1.0),
            AA_DENSITY: (0.0, 1.0),
        }
        self.dataset_plot_gamma = DEFAULT_ACCUMULATION_GAMMA
        self.dataset_plot_velocity_clim: tuple[float, float] | None = None
        self._dataset_plot_cache_key: tuple | None = None
        self._dataset_plot_geometry = None
        self._dataset_plot_status = ""
        self.multiple_dataset_accumulator: DatasetAccumulator | None = None
        self.multiple_dataset_source_paths: tuple[str, ...] = ()
        self.multiple_dataset_plot_map_name = AA_DENSITY
        self.multiple_dataset_plot_density_clims = {
            HARD_DENSITY: (0.0, 1.0),
            AA_DENSITY: (0.0, 1.0),
        }
        self.multiple_dataset_plot_gamma = DEFAULT_ACCUMULATION_GAMMA
        self.multiple_dataset_plot_velocity_clim: tuple[float, float] | None = None
        self._multiple_dataset_plot_cache_key: tuple | None = None
        self._multiple_dataset_plot_geometry = None
        self._multiple_dataset_plot_status = ""
        self.track_plot_map_name = HARD_DENSITY
        self.track_plot_velocity_clim: tuple[float, float] | None = None
        self._track_plot_cache_key: (
            tuple[int, int, str, str, tuple[float, float] | None] | None
        ) = None
        self._track_plot_geometry = None
        self._track_plot_status = ""
        self._inspect_press_pos: QPoint | None = None
        self._candidate_frame_idx: int | None = None
        self._candidate_rows = np.empty((0, 4), dtype=float)
        self._manual_candidate_placement_enabled = False
        self._manual_detection_eraser_enabled = False
        self._candidate_style = {
            "marker": "target",
            "color": "#00dcff",
            "radius": 1.0,
            "opacity": 0.75,
        }

        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(self._layer_z["image"])
        self.image_item.setTransformationMode(Qt.FastTransformation)
        self.scene.addItem(self.image_item)
        self.dataset_plot_item = QGraphicsPixmapItem()
        self.dataset_plot_item.setZValue(self._layer_z["dataset_plot"])
        self.dataset_plot_item.setTransformationMode(Qt.FastTransformation)
        self.dataset_plot_item.setAcceptedMouseButtons(Qt.NoButton)
        self.dataset_plot_item.setAcceptHoverEvents(False)
        self.scene.addItem(self.dataset_plot_item)
        self.multiple_dataset_plot_item = QGraphicsPixmapItem()
        self.multiple_dataset_plot_item.setZValue(
            self._layer_z["multiple_dataset_plot"]
        )
        self.multiple_dataset_plot_item.setTransformationMode(Qt.FastTransformation)
        self.multiple_dataset_plot_item.setAcceptedMouseButtons(Qt.NoButton)
        self.multiple_dataset_plot_item.setAcceptHoverEvents(False)
        self.scene.addItem(self.multiple_dataset_plot_item)
        self.track_plot_item = QGraphicsPixmapItem()
        self.track_plot_item.setZValue(self._layer_z["track_plot"])
        self.track_plot_item.setTransformationMode(Qt.FastTransformation)
        self.track_plot_item.setAcceptedMouseButtons(Qt.NoButton)
        self.track_plot_item.setAcceptHoverEvents(False)
        self.scene.addItem(self.track_plot_item)
        self.overlay_items = []

    # -------------------------------------------------------------- state

    def set_session(self, session: CorrectionSession | None) -> None:
        self.session = session
        self.frame_idx = 0
        self.selected_track_id = None
        self._did_initial_fit = False
        self._image_cache_key = None
        self._interpolated_image_cache_key = None
        self._interpolated_image_pixmap = QPixmap()
        self.invalidate_track_geometry()
        self.accumulation_checkpoint = None
        self._checkpoint_track_id_map = None
        self.dataset_accumulator = None
        self.dataset_plot_density_clims = {
            HARD_DENSITY: (0.0, 1.0),
            AA_DENSITY: (0.0, 1.0),
        }
        self.dataset_plot_velocity_clim = None
        self._clear_dataset_plot()
        self.track_plot_velocity_clim = None
        self._clear_track_plot()
        self._candidate_frame_idx = None
        self._candidate_rows = np.empty((0, 4), dtype=float)
        self._manual_candidate_placement_enabled = False
        self._manual_detection_eraser_enabled = False
        self._update_manual_mode_badge()
        if session is not None and session.image_stack is not None:
            self._stack_max, self._power_reference = display_references(
                session.image_stack
            )
        else:
            self._stack_max = None
            self._power_reference = None
        self.render()

    def set_frame(self, frame_idx: int) -> None:
        self.frame_idx = max(0, int(frame_idx))
        self.render()

    def set_selected_track(self, track_id: int | None) -> None:
        self.selected_track_id = track_id
        self._track_plot_cache_key = None
        self.render()

    def set_movie_watermark_preview(
        self,
        watermark: MovieWatermark | None,
        frame_number: int,
        font_size_px: int,
        *,
        visible: bool,
    ) -> None:
        """Update the fixed viewport preview without changing scene contents."""

        self.movie_watermark_preview.setGeometry(self.viewport().rect())
        self.movie_watermark_preview.set_content(
            watermark,
            frame_number,
            font_size_px,
            visible=visible,
        )

    def set_candidate_overlay(
        self,
        frame_idx: int,
        candidates: np.ndarray,
        style: dict | None = None,
    ) -> None:
        """Show temporary Add Point candidates above every display layer."""

        self._candidate_frame_idx = int(frame_idx)
        self._candidate_rows = ensure_mat_tracking(candidates)
        if style is not None:
            self._candidate_style = {
                "marker": str(style.get("marker", "target")),
                "color": str(style.get("color", "#00dcff")),
                "radius": max(1.0, float(style.get("radius", 1.0))),
                "opacity": max(0.0, min(1.0, float(style.get("opacity", 0.75)))),
            }
        self._update_manual_mode_badge()
        self.render()

    def update_candidate_style(self, style: dict) -> None:
        """Update candidate marker appearance without rerunning localization."""

        self._candidate_style = {
            "marker": str(style.get("marker", "target")),
            "color": str(style.get("color", "#00dcff")),
            "radius": max(1.0, float(style.get("radius", 1.0))),
            "opacity": max(0.0, min(1.0, float(style.get("opacity", 0.75)))),
        }
        self.render()

    def set_manual_candidate_placement_enabled(self, enabled: bool) -> None:
        """Choose whether Add Point clicks place a draft instead of applying one."""

        self._manual_candidate_placement_enabled = bool(enabled)
        if enabled:
            self._manual_detection_eraser_enabled = False
        self._update_manual_mode_badge()

    def set_manual_detection_eraser_enabled(self, enabled: bool) -> None:
        """Choose whether Add Point clicks target stored manual detections."""

        self._manual_detection_eraser_enabled = bool(enabled)
        if enabled:
            self._manual_candidate_placement_enabled = False
        self._update_manual_mode_badge()

    def clear_candidate_overlay(self) -> None:
        """Remove the temporary Add Point candidates from the scene."""

        if self._candidate_frame_idx is None and self._candidate_rows.size == 0:
            return
        self._candidate_frame_idx = None
        self._candidate_rows = np.empty((0, 4), dtype=float)
        self._update_manual_mode_badge()
        self.render()

    def _update_manual_mode_badge(self) -> None:
        """Keep Manual Placement state visible independently of scene overlays."""

        if (
            not self._manual_candidate_placement_enabled
            or self.interaction_mode != "add_point"
        ):
            self.manual_mode_badge.hide()
            return
        has_draft = self._candidate_rows.shape == (1, 4)
        if has_draft:
            text = (
                "MANUAL DRAFT PENDING\n"
                "Shift+W/A/S/D nudge · Enter confirm · Backspace clear"
            )
            background = "rgba(245, 158, 11, 235)"
        else:
            text = "MANUAL PLACEMENT ACTIVE\nClick image to place a draft"
            background = "rgba(0, 220, 255, 225)"
        self.manual_mode_badge.setText(text)
        self.manual_mode_badge.setStyleSheet(
            "QLabel#manualModeBadge {"
            f" background-color: {background};"
            " color: #08272d; border: 2px solid rgba(255, 255, 255, 220);"
            " border-radius: 6px; padding: 7px 12px; font-weight: 700;"
            "}"
        )
        self._position_manual_mode_badge()
        self.manual_mode_badge.show()
        self.manual_mode_badge.raise_()

    def _position_manual_mode_badge(self) -> None:
        available_width = max(160, self.viewport().width() - 24)
        self.manual_mode_badge.setMaximumWidth(available_width)
        self.manual_mode_badge.adjustSize()
        left = max(12, (self.viewport().width() - self.manual_mode_badge.width()) // 2)
        self.manual_mode_badge.move(left, 12)

    @property
    def candidate_rows(self) -> np.ndarray:
        """Return a defensive copy of the currently available candidates."""

        return self._candidate_rows.copy()

    def set_accumulation_checkpoint(
        self,
        checkpoint: AccumulationCheckpoint | None,
        current_to_source_track_id: dict[int, int] | None = None,
    ) -> None:
        """Attach read-only Batch plotting state without changing the session."""

        self.accumulation_checkpoint = checkpoint
        self._checkpoint_track_id_map = current_to_source_track_id
        self.dataset_accumulator = None
        self._dataset_plot_cache_key = None
        self._clear_dataset_plot()
        self._track_plot_cache_key = None
        self._clear_track_plot()
        self._refresh_dataset_plot_overlay()
        self._refresh_track_plot_overlay()

    def set_checkpoint_track_id_map(self, mapping: dict[int, int]) -> None:
        """Refresh current -> Batch source IDs after structural track edits."""

        self._checkpoint_track_id_map = {
            int(track_id): int(source_id)
            for track_id, source_id in mapping.items()
        }
        self._track_plot_cache_key = None
        self._refresh_track_plot_overlay()

    def set_dataset_plot_map(self, map_name: str) -> None:
        if map_name == self.dataset_plot_map_name:
            return
        self.dataset_plot_map_name = str(map_name)
        self._dataset_plot_cache_key = None
        self._refresh_dataset_plot_overlay()

    def set_dataset_plot_velocity_clim(self, minimum: float, maximum: float) -> None:
        """Set display-only Jet limits for current-dataset velocity maps."""

        clim = (float(minimum), float(maximum))
        if not np.all(np.isfinite(clim)) or clim[0] >= clim[1]:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if clim == self.dataset_plot_velocity_clim:
            return
        self.dataset_plot_velocity_clim = clim
        if self.dataset_plot_map_name in VELOCITY_TRACK_MAPS:
            self._dataset_plot_cache_key = None
            self._refresh_dataset_plot_overlay()

    def set_dataset_plot_gamma(self, gamma: float) -> None:
        """Set display-only Gamma for the normalized current-dataset map."""

        gamma = float(gamma)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("Accumulation Gamma must be finite and in [0, 1].")
        if gamma == self.dataset_plot_gamma:
            return
        self.dataset_plot_gamma = gamma
        self._dataset_plot_cache_key = None
        self._refresh_dataset_plot_overlay()

    def set_dataset_plot_density_clim(
        self,
        map_name: str,
        minimum: float,
        maximum: float,
    ) -> None:
        """Set independent raw-density display limits for Hard or AA Density."""

        map_name = str(map_name)
        if map_name not in DENSITY_TRACK_MAPS:
            raise ValueError(f"Unsupported density map: {map_name}")
        clim = (float(minimum), float(maximum))
        if not np.all(np.isfinite(clim)) or clim[0] < 0.0 or clim[0] >= clim[1]:
            raise ValueError(
                "Density color limits must be finite, non-negative, and minimum < maximum."
            )
        if clim == self.dataset_plot_density_clims[map_name]:
            return
        self.dataset_plot_density_clims[map_name] = clim
        if self.dataset_plot_map_name == map_name:
            self._dataset_plot_cache_key = None
            self._refresh_dataset_plot_overlay()

    def set_multiple_dataset_accumulation(
        self,
        accumulator: DatasetAccumulator | None,
        source_paths: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Replace the retained snapshot only after an explicit successful import."""

        self.multiple_dataset_accumulator = accumulator
        self.multiple_dataset_source_paths = tuple(str(path) for path in source_paths)
        self._multiple_dataset_plot_cache_key = None
        self._clear_multiple_dataset_plot()
        self._refresh_multiple_dataset_plot_overlay()

    def set_multiple_dataset_plot_map(self, map_name: str) -> None:
        if map_name == self.multiple_dataset_plot_map_name:
            return
        self.multiple_dataset_plot_map_name = str(map_name)
        self._multiple_dataset_plot_cache_key = None
        self._refresh_multiple_dataset_plot_overlay()

    def set_multiple_dataset_plot_velocity_clim(
        self,
        minimum: float,
        maximum: float,
    ) -> None:
        clim = (float(minimum), float(maximum))
        if not np.all(np.isfinite(clim)) or clim[0] >= clim[1]:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if clim == self.multiple_dataset_plot_velocity_clim:
            return
        self.multiple_dataset_plot_velocity_clim = clim
        if self.multiple_dataset_plot_map_name in VELOCITY_TRACK_MAPS:
            self._multiple_dataset_plot_cache_key = None
            self._refresh_multiple_dataset_plot_overlay()

    def set_multiple_dataset_plot_gamma(self, gamma: float) -> None:
        """Set display-only Gamma for the normalized retained snapshot."""

        gamma = float(gamma)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("Accumulation Gamma must be finite and in [0, 1].")
        if gamma == self.multiple_dataset_plot_gamma:
            return
        self.multiple_dataset_plot_gamma = gamma
        self._multiple_dataset_plot_cache_key = None
        self._refresh_multiple_dataset_plot_overlay()

    def set_multiple_dataset_plot_density_clim(
        self,
        map_name: str,
        minimum: float,
        maximum: float,
    ) -> None:
        map_name = str(map_name)
        if map_name not in DENSITY_TRACK_MAPS:
            raise ValueError(f"Unsupported density map: {map_name}")
        clim = (float(minimum), float(maximum))
        if not np.all(np.isfinite(clim)) or clim[0] < 0.0 or clim[0] >= clim[1]:
            raise ValueError(
                "Density color limits must be finite, non-negative, and minimum < maximum."
            )
        if clim == self.multiple_dataset_plot_density_clims[map_name]:
            return
        self.multiple_dataset_plot_density_clims[map_name] = clim
        if self.multiple_dataset_plot_map_name == map_name:
            self._multiple_dataset_plot_cache_key = None
            self._refresh_multiple_dataset_plot_overlay()

    def set_track_plot_map(self, map_name: str) -> None:
        if map_name == self.track_plot_map_name:
            return
        self.track_plot_map_name = str(map_name)
        self._track_plot_cache_key = None
        self._refresh_track_plot_overlay()

    def set_track_plot_velocity_clim(self, minimum: float, maximum: float) -> None:
        """Set display-only Jet limits for selected-track velocity maps."""

        clim = (float(minimum), float(maximum))
        if not np.all(np.isfinite(clim)) or clim[0] >= clim[1]:
            raise ValueError("Velocity color limits must be finite with minimum < maximum.")
        if clim == self.track_plot_velocity_clim:
            return
        self.track_plot_velocity_clim = clim
        if self.track_plot_map_name in VELOCITY_TRACK_MAPS:
            self._track_plot_cache_key = None
            self._refresh_track_plot_overlay()

    def set_db_range(self, db_min: float, db_max: float) -> None:
        self.set_display_settings(
            replace(
                self.display_settings,
                db_min=float(db_min),
                db_max=float(db_max),
            )
        )

    def set_display_settings(self, settings: DisplaySettings) -> None:
        """Apply display-only dB or Power Law compression state."""

        if not isinstance(settings, DisplaySettings):
            raise TypeError("Display settings must be a DisplaySettings instance.")
        self.display_settings = settings
        self.db_min = settings.db_min
        self.db_max = settings.db_max
        self._image_cache_key = None
        self.render()

    def set_interpolation(
        self,
        enabled: bool,
        factor: float,
        aspect_ratio: float,
    ) -> None:
        """Toggle display-only interpolation and preserve the viewed data center."""

        factor = float(factor)
        aspect_ratio = float(aspect_ratio)
        if not np.isfinite(factor) or factor < 1.0:
            raise ValueError("Interpolation factor must be finite and at least 1.0.")
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
            raise ValueError("Aspect ratio must be finite and positive.")

        old_aspect = self._active_aspect_ratio()
        center = self.mapToScene(self.viewport().rect().center())
        self.interpolation_enabled = bool(enabled)
        self.interpolation_factor = factor
        self.interpolation_aspect_ratio = aspect_ratio
        new_aspect = self._active_aspect_ratio()
        self._image_cache_key = None
        if not np.isclose(old_aspect, new_aspect):
            self.invalidate_track_geometry()
        self.render()
        if self.session is not None and old_aspect > 0.0:
            self.centerOn(center.x(), center.y() * new_aspect / old_aspect)

    def _active_factor(self) -> float:
        return self.interpolation_factor if self.interpolation_enabled else 1.0

    def _active_aspect_ratio(self) -> float:
        return self.interpolation_aspect_ratio if self.interpolation_enabled else 1.0

    def _pala_to_display_xy(self, z: float, x: float) -> tuple[float, float]:
        """Map a PALA point into the aspect-corrected display scene."""

        scene_x, scene_y = pala_to_scene_xy(z, x)
        return scene_x, scene_y * self._active_aspect_ratio()

    def _display_to_pala_zx(
        self,
        display_x: float,
        display_y: float,
    ) -> tuple[float, float]:
        """Invert the aspect-corrected display mapping through the contract helper."""

        return scene_to_pala_zx(
            float(display_x),
            float(display_y) / self._active_aspect_ratio(),
        )

    def set_interaction_mode(self, mode: str) -> None:
        """Switch between pan/inspect and click-to-edit behavior.

        In edit modes the left button is reserved for picking, so hand-drag
        panning is disabled; switch back to Inspect to pan.
        """

        self.interaction_mode = mode
        if mode == "inspect":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        self._update_manual_mode_badge()

    def update_style(self, layer_key: str, field: str, value: object) -> None:
        style = self.styles.get(layer_key)
        if style is None or not hasattr(style, field):
            return
        setattr(style, field, value)
        if layer_key == "dataset_plot":
            if field == "color":
                self._dataset_plot_cache_key = None
            self._refresh_dataset_plot_overlay()
            return
        if layer_key == "multiple_dataset_plot":
            if field == "color":
                self._multiple_dataset_plot_cache_key = None
            self._refresh_multiple_dataset_plot_overlay()
            return
        if layer_key == "track_plot":
            self._refresh_track_plot_overlay()
            return
        self.render()

    def layer_order(self) -> tuple[str, ...]:
        """Return the runtime stack from foreground (top card) to background."""

        return self._layer_order

    def layer_z(self, layer_key: str) -> float:
        """Return the runtime scene z-value for one display layer."""

        return self._layer_z[str(layer_key)]

    def set_layer_order(self, order: tuple[str, ...] | list[str]) -> None:
        """Apply a foreground-to-background stack emitted by DisplayPanel."""

        normalized = tuple(str(key) for key in order)
        layer_z = z_values_for_order(normalized)
        if normalized == self._layer_order:
            return
        self._layer_order = normalized
        self._layer_z = layer_z
        self.image_item.setZValue(layer_z["image"])
        self.dataset_plot_item.setZValue(layer_z["dataset_plot"])
        self.multiple_dataset_plot_item.setZValue(
            layer_z["multiple_dataset_plot"]
        )
        self.track_plot_item.setZValue(layer_z["track_plot"])
        # Generated vector overlays are recreated with their new z-values;
        # accumulator pixmaps and all-track geometry remain cached.
        self.render()

    def _is_verified(self, track_id: int) -> bool:
        return (
            self.session is not None
            and self.session.track_status.get(track_id) == TRACK_STATUS_VERIFIED
        )

    def invalidate_track_geometry(self, track_ids=None) -> None:
        """Drop polylines and incrementally replace edited ULM contributions."""

        self._track_geometry = None
        if (
            track_ids is None
            or self.dataset_accumulator is None
            or self.session is None
        ):
            return
        ids = {int(track_id) for track_id in track_ids if track_id is not None}
        if not ids:
            return
        try:
            self.dataset_accumulator.update_tracks(self.session, ids)
        except (IndexError, MemoryError, ValueError) as exc:
            self.dataset_accumulator = None
            self._clear_dataset_plot()
            self._set_dataset_plot_status(
                f"{LAYER_LABELS['dataset_plot']} is unavailable after an edit: {exc}"
            )
            return
        self._dataset_plot_cache_key = None
        self._refresh_dataset_plot_overlay()

    def fit_view(self) -> None:
        rect = (
            self.image_item.sceneBoundingRect()
            if not self.image_item.pixmap().isNull()
            else self.scene.itemsBoundingRect()
        )
        if rect.isValid():
            self.fitInView(rect, Qt.KeepAspectRatio)

    # -------------------------------------------------------------- events

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_manual_mode_badge()
        self.movie_watermark_preview.setGeometry(self.viewport().rect())
        if self.movie_watermark_preview.isVisible():
            self.movie_watermark_preview.raise_()

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event) -> None:
        # Arrow keys are global frame/track navigation (handled by the main
        # window); do not let QGraphicsView consume them for scrolling.
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            event.ignore()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        self._inspect_press_pos = None
        if (
            self.session is not None
            and self.interaction_mode == "inspect"
            and event.button() == Qt.LeftButton
        ):
            self._inspect_press_pos = event.position().toPoint()
        if (
            self.session is not None
            and self.interaction_mode != "inspect"
            and event.button() == Qt.LeftButton
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            if self.interaction_mode == "add_point":
                if self._manual_detection_eraser_enabled:
                    local_idx = self._find_detection_near(
                        scene_pos.x(),
                        scene_pos.y(),
                        manual_only=True,
                    )
                    if local_idx is not None:
                        self.manual_detection_erase_requested.emit(
                            self.frame_idx,
                            local_idx,
                        )
                elif self._manual_candidate_placement_enabled:
                    image_rect = self.image_item.sceneBoundingRect()
                    if image_rect.contains(scene_pos):
                        z, x = self._display_to_pala_zx(
                            scene_pos.x(),
                            scene_pos.y(),
                        )
                        nz, nx = self.session.image_stack.shape[:2]
                        self.manual_candidate_placed.emit(
                            self.frame_idx,
                            float(np.clip(z, 1.0, nz)),
                            float(np.clip(x, 1.0, nx)),
                        )
                else:
                    candidate_idx = self._find_candidate_near(
                        scene_pos.x(),
                        scene_pos.y(),
                    )
                    if candidate_idx is not None:
                        self.candidate_clicked.emit(self.frame_idx, candidate_idx)
            else:
                local_idx = self._find_detection_near(scene_pos.x(), scene_pos.y())
                if local_idx is not None:
                    self.detection_clicked.emit(self.frame_idx, local_idx)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        inspect_press_pos = self._inspect_press_pos
        self._inspect_press_pos = None
        super().mouseReleaseEvent(event)
        if (
            self.session is None
            or event.button() != Qt.LeftButton
            or inspect_press_pos is None
        ):
            return
        release_pos = event.position().toPoint()
        if (release_pos - inspect_press_pos).manhattanLength() >= (
            QApplication.startDragDistance()
        ):
            return
        self._handle_inspect_click(release_pos)

    def _handle_inspect_click(self, view_pos: QPoint) -> None:
        """Select an assigned point's track or clear on true image background."""

        if self.session is None:
            return
        scene_pos = self.mapToScene(view_pos)
        local_idx = self._find_detection_near(scene_pos.x(), scene_pos.y())
        if local_idx is None:
            if self.selected_track_id is not None:
                self.track_selection_cleared.emit()
            return
        owner = self.session.track_id_for_detection(self.frame_idx, local_idx)
        if owner is not None:
            self.track_picked.emit(owner)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click an assigned detection to select its track (any mode)."""

        self._inspect_press_pos = None
        if self.session is not None and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            local_idx = self._find_detection_near(scene_pos.x(), scene_pos.y())
            if local_idx is not None:
                owner = self.session.track_id_for_detection(self.frame_idx, local_idx)
                if owner is not None:
                    self.track_picked.emit(owner)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def _find_detection_near(
        self,
        scene_x: float,
        scene_y: float,
        *,
        manual_only: bool = False,
    ) -> int | None:
        """Return the nearest current-frame detection within a pick radius.

        The pick radius is ~8 screen pixels converted to scene units, so
        picking stays comfortable at any zoom level.
        """

        if self.session is None:
            return None
        points = ensure_mat_tracking(
            self.session.localized_by_frame.get(self.frame_idx, [])
        )
        if points.size == 0:
            return None
        visible_local_indices = [
            local_idx
            for local_idx in range(points.shape[0])
            if not self.session.is_discarded_manual_detection(
                self.frame_idx,
                local_idx,
            )
            and (
                not manual_only
                or self.session.is_manual_detection(self.frame_idx, local_idx)
            )
        ]
        if not visible_local_indices:
            return None
        visible_points = points[visible_local_indices]
        display_points = np.asarray(
            [self._pala_to_display_xy(row[1], row[2]) for row in visible_points],
            dtype=float,
        )
        dx = display_points[:, 0] - scene_x
        dy = display_points[:, 1] - scene_y
        transform = self.transform()
        scale_x = max(float(np.hypot(transform.m11(), transform.m12())), 1e-6)
        scale_y = max(float(np.hypot(transform.m21(), transform.m22())), 1e-6)
        threshold_x = max(self.styles["points"].size + 0.5, 8.0 / scale_x)
        threshold_y = max(self.styles["points"].size + 0.5, 8.0 / scale_y)
        dist = np.hypot(dx / threshold_x, dy / threshold_y)
        visible_idx = int(np.argmin(dist))
        return (
            visible_local_indices[visible_idx]
            if dist[visible_idx] <= 1.0
            else None
        )

    def _find_candidate_near(self, scene_x: float, scene_y: float) -> int | None:
        """Return the nearest visible Add Point candidate within ~10 screen px."""

        if (
            self._candidate_frame_idx != self.frame_idx
            or self._candidate_rows.size == 0
        ):
            return None
        display_points = np.asarray(
            [self._pala_to_display_xy(row[1], row[2]) for row in self._candidate_rows],
            dtype=float,
        )
        dx = display_points[:, 0] - scene_x
        dy = display_points[:, 1] - scene_y
        transform = self.transform()
        scale_x = max(float(np.hypot(transform.m11(), transform.m12())), 1e-6)
        scale_y = max(float(np.hypot(transform.m21(), transform.m22())), 1e-6)
        radius = float(self._candidate_style["radius"])
        threshold_x = max(radius + 1.0, 10.0 / scale_x)
        threshold_y = max(radius + 1.0, 10.0 / scale_y)
        dist = np.hypot(dx / threshold_x, dy / threshold_y)
        candidate_idx = int(np.argmin(dist))
        return candidate_idx if dist[candidate_idx] <= 1.0 else None

    # ------------------------------------------------------------ rendering

    def clear_overlays(self) -> None:
        for item in self.overlay_items:
            if item.scene() is not None:
                self.scene.removeItem(item)
        self.overlay_items.clear()

    def render(self) -> None:
        self.clear_overlays()
        if self.session is None:
            self.image_item.setPixmap(QPixmap())
            return

        self._draw_frame()
        self._draw_selected_track_point()
        self._draw_all_track_lines()
        self._draw_selected_track_path()
        self._draw_localizations()
        self._refresh_dataset_plot_overlay()
        self._refresh_multiple_dataset_plot_overlay()
        self._refresh_track_plot_overlay()
        self._draw_candidates()
        if not self._did_initial_fit:
            self.fit_view()
            self._did_initial_fit = True

    def _draw_candidates(self) -> None:
        """Draw temporary Add Point candidates above the reorderable layer stack."""

        if (
            self.interaction_mode != "add_point"
            or self._candidate_frame_idx != self.frame_idx
            or self._candidate_rows.size == 0
        ):
            return

        marker = str(self._candidate_style["marker"])
        radius = float(self._candidate_style["radius"])
        opacity = float(self._candidate_style["opacity"])
        color = QColor(str(self._candidate_style["color"]))
        if not color.isValid():
            color = QColor("#00dcff")
        color.setAlphaF(opacity)
        halo = QColor(0, 0, 0)
        halo.setAlphaF(min(1.0, opacity * 0.9))

        halo_pen = QPen(halo, 3.5)
        halo_pen.setCosmetic(True)
        color_pen = QPen(color, 1.6)
        color_pen.setCosmetic(True)

        def register(item, candidate_idx: int) -> None:
            item.setZValue(CANDIDATE_Z)
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setData(0, "candidate")
            item.setData(1, self.frame_idx)
            item.setData(2, candidate_idx)
            self.overlay_items.append(item)

        for candidate_idx, row in enumerate(self._candidate_rows):
            sx, sy = self._pala_to_display_xy(row[1], row[2])
            if marker in {"target", "ring"}:
                for pen in (halo_pen, color_pen):
                    item = self.scene.addEllipse(
                        sx - radius,
                        sy - radius,
                        2.0 * radius,
                        2.0 * radius,
                        pen,
                        QBrush(Qt.NoBrush),
                    )
                    register(item, candidate_idx)
                if marker == "target":
                    center_radius = max(0.55, radius * 0.18)
                    center = self.scene.addEllipse(
                        sx - center_radius,
                        sy - center_radius,
                        2.0 * center_radius,
                        2.0 * center_radius,
                        QPen(Qt.NoPen),
                        QBrush(color),
                    )
                    register(center, candidate_idx)
            elif marker == "dot":
                halo_item = self.scene.addEllipse(
                    sx - radius,
                    sy - radius,
                    2.0 * radius,
                    2.0 * radius,
                    halo_pen,
                    QBrush(Qt.NoBrush),
                )
                register(halo_item, candidate_idx)
                dot_radius = radius * 0.78
                dot = self.scene.addEllipse(
                    sx - dot_radius,
                    sy - dot_radius,
                    2.0 * dot_radius,
                    2.0 * dot_radius,
                    QPen(Qt.NoPen),
                    QBrush(color),
                )
                register(dot, candidate_idx)
            else:
                for x1, y1, x2, y2 in (
                    (sx - radius, sy, sx + radius, sy),
                    (sx, sy - radius, sx, sy + radius),
                ):
                    for pen in (halo_pen, color_pen):
                        item = self.scene.addLine(x1, y1, x2, y2, pen)
                        register(item, candidate_idx)

    def _set_dataset_plot_status(self, message: str) -> None:
        message = str(message)
        if message != self._dataset_plot_status:
            self._dataset_plot_status = message
            self.dataset_plot_status_changed.emit(message)

    def _clear_dataset_plot(self) -> None:
        self.dataset_plot_item.setVisible(False)
        self.dataset_plot_item.setPixmap(QPixmap())
        self.dataset_plot_item.setTransform(QTransform())
        self._dataset_plot_cache_key = None
        self._dataset_plot_geometry = None

    def _ensure_dataset_accumulator(self) -> DatasetAccumulator | None:
        if self.dataset_accumulator is not None:
            return self.dataset_accumulator
        if self.session is None or self.accumulation_checkpoint is None:
            return None
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.dataset_accumulator = DatasetAccumulator.build(
                self.session,
                self.accumulation_checkpoint,
            )
        except (IndexError, MemoryError, ValueError) as exc:
            self.dataset_accumulator = None
            self._set_dataset_plot_status(
                f"{LAYER_LABELS['dataset_plot']} is unavailable: {exc}"
            )
        finally:
            QApplication.restoreOverrideCursor()
        return self.dataset_accumulator

    def _refresh_dataset_plot_overlay(self) -> None:
        """Render all current tracks, rebuilding only edited contributions."""

        style = self.styles["dataset_plot"]
        self.dataset_plot_item.setOpacity(float(style.opacity))
        if not style.visible or style.opacity <= 0.0:
            self.dataset_plot_item.setVisible(False)
            return
        if self.accumulation_checkpoint is None:
            self._clear_dataset_plot()
            self._set_dataset_plot_status(
                "No Batch checkpoint is available to define the PALA ULM grid."
            )
            return
        accumulator = self._ensure_dataset_accumulator()
        if accumulator is None:
            self._clear_dataset_plot()
            return

        velocity_clim = (
            self.dataset_plot_velocity_clim
            if self.dataset_plot_map_name in VELOCITY_TRACK_MAPS
            else None
        )
        density_clim = (
            self.dataset_plot_density_clims[self.dataset_plot_map_name]
            if self.dataset_plot_map_name in DENSITY_TRACK_MAPS
            else None
        )
        cache_key = (
            accumulator.revision,
            self.dataset_plot_map_name,
            style.color,
            self.dataset_plot_gamma,
            density_clim,
            velocity_clim,
        )
        if cache_key != self._dataset_plot_cache_key:
            try:
                rows, cols, values = accumulator.map_values(
                    self.dataset_plot_map_name
                )
                rendered = render_sparse_dataset_map(
                    rows,
                    cols,
                    values,
                    self.dataset_plot_map_name,
                    accumulator.settings.display_params,
                    accumulator.settings.ulm_scale_z,
                    accumulator.settings.ulm_scale_x,
                    density_rgb=(
                        QColor(style.color).red(),
                        QColor(style.color).green(),
                        QColor(style.color).blue(),
                    ),
                    density_clim=density_clim,
                    velocity_clim=velocity_clim,
                    gamma=self.dataset_plot_gamma,
                )
            except ValueError as exc:
                self._clear_dataset_plot()
                self._set_dataset_plot_status(
                    f"{LAYER_LABELS['dataset_plot']} cannot be displayed: {exc}"
                )
                return
            if rendered is None:
                self._clear_dataset_plot()
                self._set_dataset_plot_status(
                    f"The current tracks have no occupied pixels for "
                    f"{TRACK_MAP_LABELS[self.dataset_plot_map_name]}."
                )
                return
            self.dataset_plot_item.setPixmap(rgba_to_qpixmap(rendered.rgba))
            self._dataset_plot_geometry = rendered.geometry
            self._dataset_plot_cache_key = cache_key

        geometry = self._dataset_plot_geometry
        if geometry is not None:
            aspect_ratio = self._active_aspect_ratio()
            self.dataset_plot_item.setPos(
                geometry.pos_x,
                geometry.pos_y * aspect_ratio,
            )
            self.dataset_plot_item.setTransform(
                QTransform.fromScale(
                    geometry.scale_x,
                    geometry.scale_y * aspect_ratio,
                )
            )
        self.dataset_plot_item.setVisible(True)
        self._set_dataset_plot_status(
            f"Current Dataset: {TRACK_MAP_LABELS[self.dataset_plot_map_name]} from "
            f"{accumulator.track_count} current tracks; values are normalized to 0–1 "
            "before Gamma and Clim, and edited tracks update incrementally."
        )

    def _set_multiple_dataset_plot_status(self, message: str) -> None:
        message = str(message)
        if message != self._multiple_dataset_plot_status:
            self._multiple_dataset_plot_status = message
            self.multiple_dataset_plot_status_changed.emit(message)

    def _clear_multiple_dataset_plot(self) -> None:
        self.multiple_dataset_plot_item.setVisible(False)
        self.multiple_dataset_plot_item.setPixmap(QPixmap())
        self.multiple_dataset_plot_item.setTransform(QTransform())
        self._multiple_dataset_plot_cache_key = None
        self._multiple_dataset_plot_geometry = None

    def _refresh_multiple_dataset_plot_overlay(self) -> None:
        """Render the explicit-import snapshot without consulting current tracks."""

        style = self.styles["multiple_dataset_plot"]
        self.multiple_dataset_plot_item.setOpacity(float(style.opacity))
        if not style.visible or style.opacity <= 0.0:
            self.multiple_dataset_plot_item.setVisible(False)
            return
        accumulator = self.multiple_dataset_accumulator
        if accumulator is None:
            self._clear_multiple_dataset_plot()
            self._set_multiple_dataset_plot_status(
                f"Import multiple dataset MAT files to create "
                f"{LAYER_LABELS['multiple_dataset_plot']}."
            )
            return
        if self.session is not None and self.session.image_stack is not None:
            current_shape = tuple(
                int(value) for value in self.session.image_stack.shape[:2]
            )
            if current_shape != accumulator.settings.input_shape:
                self._clear_multiple_dataset_plot()
                self._set_multiple_dataset_plot_status(
                    f"The retained map uses input grid {accumulator.settings.input_shape}; "
                    f"the current dataset uses {current_shape}. The imported data is kept "
                    "but hidden for this dataset."
                )
                return

        velocity_clim = (
            self.multiple_dataset_plot_velocity_clim
            if self.multiple_dataset_plot_map_name in VELOCITY_TRACK_MAPS
            else None
        )
        density_clim = (
            self.multiple_dataset_plot_density_clims[
                self.multiple_dataset_plot_map_name
            ]
            if self.multiple_dataset_plot_map_name in DENSITY_TRACK_MAPS
            else None
        )
        cache_key = (
            id(accumulator),
            accumulator.revision,
            self.multiple_dataset_plot_map_name,
            style.color,
            self.multiple_dataset_plot_gamma,
            density_clim,
            velocity_clim,
        )
        if cache_key != self._multiple_dataset_plot_cache_key:
            try:
                rows, cols, values = accumulator.map_values(
                    self.multiple_dataset_plot_map_name
                )
                rendered = render_sparse_dataset_map(
                    rows,
                    cols,
                    values,
                    self.multiple_dataset_plot_map_name,
                    accumulator.settings.display_params,
                    accumulator.settings.ulm_scale_z,
                    accumulator.settings.ulm_scale_x,
                    density_rgb=(
                        QColor(style.color).red(),
                        QColor(style.color).green(),
                        QColor(style.color).blue(),
                    ),
                    density_clim=density_clim,
                    velocity_clim=velocity_clim,
                    gamma=self.multiple_dataset_plot_gamma,
                )
            except ValueError as exc:
                self._clear_multiple_dataset_plot()
                self._set_multiple_dataset_plot_status(
                    f"{LAYER_LABELS['multiple_dataset_plot']} cannot be displayed: "
                    f"{exc}"
                )
                return
            if rendered is None:
                self._clear_multiple_dataset_plot()
                self._set_multiple_dataset_plot_status(
                    f"The imported tracks have no occupied pixels for "
                    f"{TRACK_MAP_LABELS[self.multiple_dataset_plot_map_name]}."
                )
                return
            self.multiple_dataset_plot_item.setPixmap(rgba_to_qpixmap(rendered.rgba))
            self._multiple_dataset_plot_geometry = rendered.geometry
            self._multiple_dataset_plot_cache_key = cache_key

        geometry = self._multiple_dataset_plot_geometry
        if geometry is not None:
            aspect_ratio = self._active_aspect_ratio()
            self.multiple_dataset_plot_item.setPos(
                geometry.pos_x,
                geometry.pos_y * aspect_ratio,
            )
            self.multiple_dataset_plot_item.setTransform(
                QTransform.fromScale(
                    geometry.scale_x,
                    geometry.scale_y * aspect_ratio,
                )
            )
        self.multiple_dataset_plot_item.setVisible(True)
        self._set_multiple_dataset_plot_status(
            f"{LAYER_LABELS['multiple_dataset_plot']}: "
            f"{TRACK_MAP_LABELS[self.multiple_dataset_plot_map_name]} from "
            f"{accumulator.track_count} tracks across {accumulator.dataset_count} "
            "imported datasets; values are normalized to 0–1 before Gamma and Clim; "
            "current-dataset changes do not update this snapshot."
        )

    def _set_track_plot_status(self, message: str) -> None:
        message = str(message)
        if message != self._track_plot_status:
            self._track_plot_status = message
            self.track_plot_status_changed.emit(message)

    def _clear_track_plot(self) -> None:
        self.track_plot_item.setVisible(False)
        self.track_plot_item.setPixmap(QPixmap())
        self.track_plot_item.setTransform(QTransform())
        self._track_plot_cache_key = None
        self._track_plot_geometry = None

    def _source_track_id(self) -> int | None:
        if self.selected_track_id is None:
            return None
        if self._checkpoint_track_id_map is None:
            return self.selected_track_id
        return self._checkpoint_track_id_map.get(self.selected_track_id)

    def _refresh_track_plot_overlay(self) -> None:
        """Refresh only when checkpoint/track/map changes, never on frame steps."""

        style = self.styles["track_plot"]
        self.track_plot_item.setOpacity(float(style.opacity))
        if not style.visible or style.opacity <= 0:
            self.track_plot_item.setVisible(False)
            return
        checkpoint = self.accumulation_checkpoint
        if checkpoint is None or checkpoint.mode == "none":
            self._clear_track_plot()
            self._set_track_plot_status(
                "No per-track Batch checkpoint contribution is available."
            )
            return
        if self.selected_track_id is None:
            self._clear_track_plot()
            self._set_track_plot_status("Select a track to show its Batch checkpoint overlay.")
            return
        source_track_id = self._source_track_id()
        if source_track_id is None:
            self._clear_track_plot()
            self._set_track_plot_status(
                f"Track {self.selected_track_id} has no source-ID mapping for the checkpoint."
            )
            return
        if not checkpoint.has_track(source_track_id):
            self._clear_track_plot()
            self._set_track_plot_status(
                f"Source track {source_track_id} is unavailable in the Batch checkpoint."
            )
            return
        if self.track_plot_map_name not in checkpoint.available_track_maps(source_track_id):
            self._clear_track_plot()
            label = TRACK_MAP_LABELS.get(self.track_plot_map_name, self.track_plot_map_name)
            self._set_track_plot_status(
                f"{label} is unavailable in schema v{checkpoint.schema_version} "
                f"{checkpoint.mode} checkpoint."
            )
            return

        velocity_clim = (
            self.track_plot_velocity_clim
            if self.track_plot_map_name in VELOCITY_TRACK_MAPS
            else None
        )
        cache_key = (
            id(checkpoint),
            source_track_id,
            self.track_plot_map_name,
            style.color,
            velocity_clim,
        )
        if cache_key != self._track_plot_cache_key:
            try:
                rows, cols, values = checkpoint.track_map_values(
                    source_track_id,
                    self.track_plot_map_name,
                )
                rendered = render_sparse_track_map(
                    rows,
                    cols,
                    values,
                    self.track_plot_map_name,
                    checkpoint.params,
                    checkpoint.ulm_scale_z,
                    checkpoint.ulm_scale_x,
                    density_rgb=(
                        QColor(style.color).red(),
                        QColor(style.color).green(),
                        QColor(style.color).blue(),
                    ),
                    velocity_clim=velocity_clim,
                )
            except (KeyError, TrackMapUnavailableError, ValueError) as exc:
                self._clear_track_plot()
                self._set_track_plot_status(f"Batch checkpoint overlay unavailable: {exc}")
                return
            if rendered is None:
                self._clear_track_plot()
                self._set_track_plot_status(
                    f"Source track {source_track_id} has an empty contribution for "
                    f"{TRACK_MAP_LABELS[self.track_plot_map_name]}."
                )
                return
            self.track_plot_item.setPixmap(rgba_to_qpixmap(rendered.rgba))
            self._track_plot_geometry = rendered.geometry
            self._track_plot_cache_key = cache_key

        geometry = self._track_plot_geometry
        if geometry is not None:
            aspect_ratio = self._active_aspect_ratio()
            self.track_plot_item.setPos(
                geometry.pos_x,
                geometry.pos_y * aspect_ratio,
            )
            self.track_plot_item.setTransform(
                QTransform.fromScale(
                    geometry.scale_x,
                    geometry.scale_y * aspect_ratio,
                )
            )

        self.track_plot_item.setVisible(True)
        source_note = (
            ""
            if source_track_id == self.selected_track_id
            else f" (source track {source_track_id})"
        )
        self._set_track_plot_status(
            f"Batch checkpoint overlay: {TRACK_MAP_LABELS[self.track_plot_map_name]}"
            f"{source_note}. Edits are not re-plotted."
        )

    def _draw_frame(self) -> None:
        if self.session is None or self.session.image_stack is None:
            return
        stack = self.session.image_stack
        frame_idx = min(self.frame_idx, stack.shape[2] - 1)
        stack_max = self._stack_max if self._stack_max is not None else 1.0
        power_reference = (
            self._power_reference if self._power_reference is not None else 1.0
        )
        factor = self._active_factor()
        aspect_ratio = self._active_aspect_ratio()
        cache_key = (
            id(stack),
            frame_idx,
            self.display_settings,
            factor,
            aspect_ratio,
        )
        if cache_key != self._image_cache_key:
            if (
                self.interpolation_enabled
                and cache_key == self._interpolated_image_cache_key
                and not self._interpolated_image_pixmap.isNull()
            ):
                pixmap = self._interpolated_image_pixmap
            else:
                frame = stack[:, :, frame_idx]
                if self.interpolation_enabled:
                    frame = interpolate_frame_for_display(frame, factor, aspect_ratio)
                pixmap = gray_to_qpixmap(
                    normalize_display_frame(
                        frame,
                        settings=self.display_settings,
                        db_reference=stack_max,
                        power_reference=power_reference,
                    )
                )
                if self.interpolation_enabled:
                    self._interpolated_image_cache_key = cache_key
                    self._interpolated_image_pixmap = pixmap
            self.image_item.setPixmap(pixmap)
            self.image_item.setTransform(
                QTransform.fromScale(1.0 / factor, 1.0 / factor)
            )
            self._image_cache_key = cache_key
        image_style = self.styles["image"]
        self.image_item.setOpacity(image_style.opacity)
        self.image_item.setVisible(image_style.visible and image_style.opacity > 0.0)
        self.scene.setSceneRect(self.image_item.sceneBoundingRect())

    def _draw_localizations(self) -> None:
        if self.session is None:
            return
        style = self.styles["points"]
        if not style.visible or style.opacity <= 0:
            return
        points = ensure_mat_tracking(
            self.session.localized_by_frame.get(self.frame_idx, [])
        )
        if points.size == 0:
            return

        radius = style.size
        pens: dict[str, QPen] = {}
        brushes: dict[str, QBrush] = {}
        for key, color_hex in (
            ("unassigned", style.color),
            ("current", self.styles["points_current"].color),
            ("other", self.styles["points_other"].color),
            ("verified", self.styles["points_verified"].color),
            ("flagged", self.styles["points_flagged"].color),
        ):
            state_style = LayerStyle(
                opacity=style.opacity, color=color_hex, size=radius
            )
            pens[key] = QPen(_styled_color(state_style), 1.0)
            brushes[key] = QBrush(_styled_color(state_style, alpha_scale=0.55))

        for local_idx, row in enumerate(points):
            if self.session.is_discarded_manual_detection(
                self.frame_idx,
                local_idx,
            ):
                continue
            _, z, x, _ = row
            owner = self.session.track_id_for_detection(self.frame_idx, local_idx)
            if owner is None:
                key = "unassigned"
            elif owner == self.selected_track_id:
                key = "current"
            else:
                key = "other"
                owner_status = self.session.track_status.get(owner)
                if (
                    owner_status == TRACK_STATUS_VERIFIED
                    and self.styles["points_verified"].visible
                ):
                    key = "verified"
                elif (
                    owner_status == TRACK_STATUS_FLAGGED
                    and self.styles["points_flagged"].visible
                ):
                    key = "flagged"
            sx, sy = self._pala_to_display_xy(z, x)
            item = self.scene.addEllipse(
                sx - radius,
                sy - radius,
                2 * radius,
                2 * radius,
                pens[key],
                brushes[key],
            )
            item.setZValue(self._layer_z["points"])
            item.setData(0, "localization")
            item.setData(1, self.frame_idx)
            item.setData(2, int(local_idx))
            self.overlay_items.append(item)

            # human-added rows modified the localization result: mark with a +
            if self.session.is_manual_detection(self.frame_idx, local_idx):
                arm = radius * 1.8
                for x1, y1, x2, y2 in (
                    (sx - arm, sy, sx + arm, sy),
                    (sx, sy - arm, sx, sy + arm),
                ):
                    cross = self.scene.addLine(x1, y1, x2, y2, pens[key])
                    cross.setZValue(self._layer_z["points"])
                    self.overlay_items.append(cross)

    def _ensure_track_geometry(self) -> list[dict]:
        """Cache per-track scene-space polylines for the all-track line layer."""

        if self._track_geometry is not None:
            return self._track_geometry

        geometry: list[dict] = []
        if self.session is not None:
            for track_id, track in enumerate(self.session.tracks):
                frames = np.flatnonzero(~np.isnan(track))
                if frames.size < 2:
                    continue
                points: list[tuple[float, float]] = []
                for frame_idx in frames:
                    locs = ensure_mat_tracking(
                        self.session.localized_by_frame.get(int(frame_idx), [])
                    )
                    local_idx = int(track[frame_idx])
                    if local_idx >= locs.shape[0]:
                        continue
                    _, z, x, _ = locs[local_idx]
                    points.append(self._pala_to_display_xy(z, x))
                if len(points) >= 2:
                    geometry.append(
                        {
                            "track_id": track_id,
                            "start": int(frames[0]),
                            "end": int(frames[-1]),
                            "points": points,
                        }
                    )
        self._track_geometry = geometry
        return geometry

    def _draw_all_track_lines(self) -> None:
        """Draw full polylines of every track active at the current frame.

        All segments go into one QGraphicsPathItem so thousands of tracks do
        not create thousands of scene items.
        """

        style = self.styles["all_lines"]
        if self.session is None or not style.visible or style.opacity <= 0:
            return

        # verified tracks get their own path so they render in the
        # track_path_verified color
        paths = {"normal": QPainterPath(), "verified": QPainterPath()}
        for geom in self._ensure_track_geometry():
            if geom["track_id"] == self.selected_track_id:
                continue
            if not (geom["start"] <= self.frame_idx <= geom["end"]):
                continue
            key = "verified" if self._is_verified(geom["track_id"]) else "normal"
            path = paths[key]
            points = geom["points"]
            path.moveTo(points[0][0], points[0][1])
            for sx, sy in points[1:]:
                path.lineTo(sx, sy)
        colors = {
            "normal": style.color,
            "verified": self.styles["track_path_verified"].color,
        }
        for key, path in paths.items():
            if path.isEmpty():
                continue
            pen_style = LayerStyle(opacity=style.opacity, color=colors[key])
            item = self.scene.addPath(path, QPen(_styled_color(pen_style), style.size))
            item.setZValue(self._layer_z["all_lines"])
            self.overlay_items.append(item)

    def _draw_selected_track_path(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        if self.selected_track_id >= self.session.n_tracks:
            return
        style = self.styles["track_path"]
        if not style.visible or style.opacity <= 0:
            return

        color = style.color
        if self._is_verified(self.selected_track_id):
            color = self.styles["track_path_verified"].color

        track = self.session.tracks[self.selected_track_id]
        pen = QPen(
            _styled_color(LayerStyle(opacity=style.opacity, color=color)),
            style.size,
        )
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        path = QPainterPath()
        previous = None
        for frame_idx, local_idx in enumerate(track):
            if np.isnan(local_idx):
                continue
            locs = ensure_mat_tracking(self.session.localized_by_frame.get(frame_idx, []))
            local_idx_int = int(local_idx)
            if local_idx_int >= locs.shape[0]:
                continue
            _, z, x, _ = locs[local_idx_int]
            current = self._pala_to_display_xy(z, x)
            if previous is None:
                path.moveTo(current[0], current[1])
            else:
                path.lineTo(current[0], current[1])
            previous = current

        if path.elementCount() < 2:
            return
        item = self.scene.addPath(path, pen)
        item.setZValue(self._layer_z["track_path"])
        item.setData(0, "selected_track_path")
        item.setData(1, self.selected_track_id)
        self.overlay_items.append(item)

    def _draw_selected_track_point(self) -> None:
        if self.session is None or self.selected_track_id is None:
            return
        if self.selected_track_id >= self.session.n_tracks:
            return
        style = self.styles["track_point"]
        if not style.visible or style.opacity <= 0:
            return
        local_idx = self.session.local_idx_for_track_frame(
            self.selected_track_id,
            self.frame_idx,
        )
        if local_idx is None:
            self._draw_gap_indicator(style)
            return
        locs = ensure_mat_tracking(self.session.localized_by_frame.get(self.frame_idx, []))
        if local_idx >= locs.shape[0]:
            return
        _, z, x, _ = locs[local_idx]
        sx, sy = self._pala_to_display_xy(z, x)
        radius = style.size
        item = self.scene.addEllipse(
            sx - radius,
            sy - radius,
            2 * radius,
            2 * radius,
            QPen(_styled_color(style), 2.0),
            QBrush(_styled_color(style, alpha_scale=0.35)),
        )
        item.setZValue(self._layer_z["track_point"])
        self.overlay_items.append(item)

    def _draw_gap_indicator(self, style) -> None:
        """Dashed ring at the interpolated position of a gap frame.

        Shown when the current frame lies inside the selected track's range
        but has no assigned detection — the spot where the user most likely
        needs to click or manually localize.
        """

        track = self.session.tracks[self.selected_track_id]
        frames = np.flatnonzero(~np.isnan(track))
        prev_frames = frames[frames < self.frame_idx]
        next_frames = frames[frames > self.frame_idx]
        if prev_frames.size == 0 or next_frames.size == 0:
            return

        coords = []
        for frame_idx in (int(prev_frames[-1]), int(next_frames[0])):
            locs = ensure_mat_tracking(self.session.localized_by_frame.get(frame_idx, []))
            local_idx = int(track[frame_idx])
            if local_idx >= locs.shape[0]:
                return
            _, z, x, _ = locs[local_idx]
            coords.append((frame_idx, *self._pala_to_display_xy(z, x)))

        (f0, sx0, sy0), (f1, sx1, sy1) = coords
        t = (self.frame_idx - f0) / max(f1 - f0, 1)
        sx = sx0 + t * (sx1 - sx0)
        sy = sy0 + t * (sy1 - sy0)

        pen = QPen(_styled_color(style), 1.5)
        pen.setStyle(Qt.DashLine)
        radius = style.size + 1.0
        item = self.scene.addEllipse(
            sx - radius,
            sy - radius,
            2 * radius,
            2 * radius,
            pen,
            QBrush(Qt.NoBrush),
        )
        item.setZValue(self._layer_z["track_point"])
        self.overlay_items.append(item)
