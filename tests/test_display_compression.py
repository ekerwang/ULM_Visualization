import numpy as np
import pytest

from ulm_track_correction_gui.gui.display_compression import (
    DISPLAY_MODE_POWER,
    DisplaySettings,
    display_references,
    normalize_db_frame,
    normalize_display_frame,
    normalize_power_frame,
)


def test_display_references_use_stack_max_and_median_frame_maximum():
    stack = np.asarray(
        [
            [[1.0, 4.0, 2.0], [0.5, 3.0, 9.0]],
            [[0.0, 2.0, 1.0], [0.0, 1.0, 8.0]],
        ],
        dtype=np.float32,
    )

    db_reference, power_reference = display_references(stack)

    assert db_reference == 9.0
    assert power_reference == 4.0


def test_display_references_treat_nonfinite_samples_as_zero():
    stack = np.asarray([[[np.nan, 2.0], [np.inf, 1.0]]], dtype=np.float32)

    assert display_references(stack) == (2.0, 1.0)


def test_db_scale_uses_complete_stack_maximum_reference():
    frame = np.asarray([[0.0, 0.1, 1.0, 2.0, np.nan]], dtype=np.float32)

    result = normalize_db_frame(frame, reference=1.0, db_min=-20.0, db_max=0.0)

    np.testing.assert_array_equal(result, [[0, 0, 255, 255, 0]])


def test_gamma_half_uses_ensemble_reference_and_saturates_above_it():
    frame = np.asarray([[0.0, 0.25, 1.0, 4.0]], dtype=np.float32)

    result = normalize_power_frame(
        frame,
        reference=1.0,
        gamma=0.5,
        clim_min=0.0,
        clim_max=1.0,
    )

    np.testing.assert_allclose(result, [[0, 127, 255, 255]], atol=1)


def test_clim_maps_selected_power_range_to_black_and_white():
    frame = np.asarray([[0.2, 0.5, 0.8]], dtype=np.float32)

    result = normalize_power_frame(
        frame,
        reference=1.0,
        gamma=1.0,
        clim_min=0.2,
        clim_max=0.8,
    )

    np.testing.assert_allclose(result, [[0, 127, 255]], atol=1)


def test_gamma_zero_keeps_zero_black_and_maps_positive_values_to_white():
    result = normalize_power_frame(
        np.asarray([[0.0, 0.01, 1.0]], dtype=np.float32),
        reference=1.0,
        gamma=0.0,
        clim_min=0.0,
        clim_max=1.0,
    )

    np.testing.assert_array_equal(result, [[0, 255, 255]])


def test_invalid_power_clim_is_rejected():
    with pytest.raises(ValueError, match="Clim"):
        normalize_power_frame(
            np.ones((1, 1)),
            reference=1.0,
            gamma=0.5,
            clim_min=0.8,
            clim_max=0.8,
        )


def test_display_settings_route_to_power_law():
    frame = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32)
    settings = DisplaySettings(
        mode=DISPLAY_MODE_POWER,
        gamma=0.5,
        clim_min=0.0,
        clim_max=1.0,
    )

    result = normalize_display_frame(
        frame,
        settings=settings,
        db_reference=100.0,
        power_reference=1.0,
    )

    np.testing.assert_allclose(result, [[0, 127, 255]], atol=1)
