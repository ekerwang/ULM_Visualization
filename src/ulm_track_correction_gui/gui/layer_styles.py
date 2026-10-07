"""Per-layer visual style state shared by the viewer and the display panel."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DB_MIN = -30.0
DEFAULT_DB_MAX = 0.0
DEFAULT_IMAGE_OPACITY = 0.68
DEFAULT_MULTIPLE_DATASET_OPACITY = 0.5


@dataclass
class LayerStyle:
    """Visual parameters of one viewer layer.

    ``color`` is a ``#rrggbb`` hex string; ``opacity`` is 0..1 and is applied
    on top of the color. ``size`` is a point radius or a line width in scene
    (pixel) units depending on the layer.
    """

    visible: bool = True
    opacity: float = 1.0
    color: str = "#ffffff"
    size: float = 1.0


def default_layer_styles() -> dict[str, LayerStyle]:
    """Reviewed defaults matching the correction workflow's visual preset.

    ``points_current`` / ``points_other`` only contribute their color. The
    checked ``points_verified`` / ``points_flagged`` states contribute a color
    override for those subsets of other-track points. Their shared opacity and
    size continue to live in ``points``.
    ``track_path_verified`` likewise only contributes the color used for
    verified tracks (selected path and all-lines layer).
    """

    return {
        "image": LayerStyle(opacity=DEFAULT_IMAGE_OPACITY),
        # Building every current track is intentionally lazy; keeping this new
        # layer off at startup avoids blocking large-session load.
        "dataset_plot": LayerStyle(visible=False, opacity=1.0, color="#00d8ff"),
        "multiple_dataset_plot": LayerStyle(
            visible=False,
            opacity=DEFAULT_MULTIPLE_DATASET_OPACITY,
            color="#00d8ff",
        ),
        "track_plot": LayerStyle(visible=True, opacity=1.0, color="#00d8ff"),
        "all_lines": LayerStyle(
            visible=False, opacity=0.43, color="#9696dc", size=0.5
        ),
        "points": LayerStyle(opacity=0.51, color="#ff9632", size=0.5),
        "points_current": LayerStyle(color="#f050dc"),
        "points_other": LayerStyle(color="#3cf078"),
        "points_verified": LayerStyle(color="#969696"),
        "points_flagged": LayerStyle(color="#e5484d"),
        "track_path": LayerStyle(opacity=0.58, color="#f050dc", size=0.5),
        "track_path_verified": LayerStyle(color="#3ce65a"),
        "track_point": LayerStyle(opacity=0.20, color="#ffe650", size=2.0),
    }
