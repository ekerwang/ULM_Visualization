"""Shared Display-card names and QGraphicsScene layer-order definitions."""

from __future__ import annotations

# Canonical user-facing names, in the default Photoshop-style
# foreground-to-background order. Keeping the labels and order together avoids
# reintroducing inconsistent terms in the Display panel.
LAYER_LABELS = {
    "track_plot": "Selected Track (sub-pixel plotting)",
    "track_path": "Selected Track",
    "track_point": "Selected Track Point",
    "points": "Localization Points",
    "all_lines": "All Tracks",
    "image": "Ultrasound Movie",
    "dataset_plot": "Single Dataset Accumulation",
    "multiple_dataset_plot": "Multi Dataset Accumulation (ULM)",
}

LAYER_KEYS = tuple(LAYER_LABELS)
DEFAULT_LAYER_ORDER = LAYER_KEYS

LAYER_Z_STEP = 10.0


def z_values_for_order(order: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Return scene z-values for a foreground-to-background layer order."""

    normalized = tuple(str(key) for key in order)
    if len(normalized) != len(LAYER_KEYS) or set(normalized) != set(LAYER_KEYS):
        raise ValueError("Layer order must contain every layer exactly once.")
    count = len(normalized)
    return {
        key: float((count - index - 1) * LAYER_Z_STEP)
        for index, key in enumerate(normalized)
    }
