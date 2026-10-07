"""Sub-pixel refinement of a manual localization click.

Given a click position on one frame, snap to the brightest pixel in a small
search window and refine to sub-pixel precision with the migrated PALA
radial-symmetry kernel. Coordinates are 1-based PALA pixel-center coordinates
throughout (the GUI converts the Qt click via ``scene_to_pala_zx`` first).
"""

from __future__ import annotations

import numpy as np

from ulm_track_correction_gui.reference_processing.localization_funcs.localizeRadialSymmetry import (
    localizeRadialSymmetry,
)

DEFAULT_HALF_WINDOW = 4


def sample_manual_intensity(
    image: np.ndarray,
    z: float,
    x: float,
) -> float:
    """Bilinearly sample ``|image|`` at a 1-based PALA coordinate.

    Human-defined points do not run a localization algorithm, but their
    appended localization row still needs a deterministic intensity value.
    Sampling the raw-frame magnitude preserves the confirmed sub-pixel
    position without snapping it to a neighboring pixel.
    """

    magnitude = np.abs(np.asarray(image)).astype(np.float64)
    if magnitude.ndim != 2:
        raise ValueError("Manual intensity sampling expects a single 2D frame.")
    nz, nx = magnitude.shape
    z = float(z)
    x = float(x)
    if not (np.isfinite(z) and np.isfinite(x)):
        raise ValueError("Manual point coordinates must be finite.")
    if not (1.0 <= z <= nz and 1.0 <= x <= nx):
        raise ValueError(
            f"Manual point (z={z:g}, x={x:g}) lies outside the "
            f"{nz}x{nx} image."
        )

    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    z0_float = z - 1.0
    x0_float = x - 1.0
    z0 = int(np.floor(z0_float))
    x0 = int(np.floor(x0_float))
    z1 = min(z0 + 1, nz - 1)
    x1 = min(x0 + 1, nx - 1)
    wz = z0_float - z0
    wx = x0_float - x0
    return float(
        (1.0 - wz) * (1.0 - wx) * magnitude[z0, x0]
        + (1.0 - wz) * wx * magnitude[z0, x1]
        + wz * (1.0 - wx) * magnitude[z1, x0]
        + wz * wx * magnitude[z1, x1]
    )


def refine_manual_click(
    image: np.ndarray,
    z_click: float,
    x_click: float,
    half_window: int = DEFAULT_HALF_WINDOW,
) -> tuple[float, float, float]:
    """Return ``(intensity, z, x)`` for a manual click on one frame.

    Steps: take ``|image|``; find the peak pixel within ``half_window`` of the
    click; extract the largest centered odd window that fits; run radial
    symmetry for the sub-pixel offset (clipped to the window, falling back to
    the peak pixel if the kernel fails or the window is degenerate).
    """

    img = np.abs(np.asarray(image)).astype(np.float64)
    if img.ndim != 2:
        raise ValueError("Manual localization expects a single 2D frame.")
    nz, nx = img.shape
    hw = max(1, int(half_window))

    zi = int(np.clip(round(z_click), 1, nz))
    xi = int(np.clip(round(x_click), 1, nx))

    z_lo, z_hi = max(zi - hw, 1), min(zi + hw, nz)
    x_lo, x_hi = max(xi - hw, 1), min(xi + hw, nx)
    search = img[z_lo - 1 : z_hi, x_lo - 1 : x_hi]
    dz, dx = np.unravel_index(int(np.argmax(search)), search.shape)
    z_pk = z_lo + int(dz)
    x_pk = x_lo + int(dx)
    intensity = float(img[z_pk - 1, x_pk - 1])

    hz = min(z_pk - 1, nz - z_pk, hw)
    hx = min(x_pk - 1, nx - x_pk, hw)
    if hz < 1 or hx < 1:
        return intensity, float(z_pk), float(x_pk)

    window = img[z_pk - 1 - hz : z_pk + hz, x_pk - 1 - hx : x_pk + hx]
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            zc, xc = localizeRadialSymmetry(window)
    except Exception:
        zc = xc = 0.0
    if not (np.isfinite(zc) and np.isfinite(xc)):
        zc = xc = 0.0
    zc = float(np.clip(zc, -hz, hz))
    xc = float(np.clip(xc, -hx, hx))
    return intensity, float(z_pk + zc), float(x_pk + xc)
