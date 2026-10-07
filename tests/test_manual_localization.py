import numpy as np
import pytest

from ulm_track_correction_gui.core.manual_localization import (
    refine_manual_click,
    sample_manual_intensity,
)


def gaussian_frame(z0: float, x0: float, nz: int = 32, nx: int = 32, sigma: float = 1.5):
    zz, xx = np.meshgrid(
        np.arange(1, nz + 1, dtype=float),
        np.arange(1, nx + 1, dtype=float),
        indexing="ij",
    )
    return np.exp(-((zz - z0) ** 2 + (xx - x0) ** 2) / (2 * sigma**2))


def test_refine_recovers_subpixel_center_from_offset_click():
    z0, x0 = 14.3, 19.6
    frame = gaussian_frame(z0, x0)
    # click ~2 px away from the true center
    intensity, z, x = refine_manual_click(frame, z0 + 1.8, x0 - 1.4)
    assert abs(z - z0) < 0.15
    assert abs(x - x0) < 0.15
    assert intensity > 0.5


def test_refine_handles_complex_input_and_edges():
    z0, x0 = 3.0, 30.0
    frame = gaussian_frame(z0, x0).astype(np.complex128) * (1 + 1j)
    intensity, z, x = refine_manual_click(frame, 2.0, 31.5)
    assert abs(z - z0) < 0.5
    assert abs(x - x0) < 0.5
    assert intensity > 0


def test_refine_flat_image_returns_click_pixel():
    frame = np.zeros((16, 16))
    _, z, x = refine_manual_click(frame, 8.2, 9.7)
    # peak search on a flat image stays near the clicked pixel
    assert 4 <= z <= 12
    assert 5 <= x <= 14


def test_manual_intensity_bilinearly_samples_complex_magnitude():
    frame = np.asarray([[0.0, 2.0], [4.0, 6.0]], dtype=np.complex128) * (1 + 1j)
    assert sample_manual_intensity(frame, 1.5, 1.5) == pytest.approx(
        3.0 * np.sqrt(2.0)
    )
    assert sample_manual_intensity(frame, 2.0, 2.0) == pytest.approx(
        6.0 * np.sqrt(2.0)
    )


def test_manual_intensity_rejects_out_of_bounds_coordinates():
    with pytest.raises(ValueError, match="outside"):
        sample_manual_intensity(np.ones((2, 3)), 0.9, 2.0)
