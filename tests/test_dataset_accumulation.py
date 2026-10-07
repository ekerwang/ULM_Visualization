from pathlib import Path

import numpy as np
import pytest

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    HARD_DENSITY,
    HARD_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    AccumulationCheckpoint,
)
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.dataset_accumulation import (
    DatasetAccumulationSettings,
    DatasetAccumulator,
    pala_moving_average,
)
from ulm_track_correction_gui.gui.track_plot_rendering import (
    render_sparse_dataset_map,
)


def make_checkpoint(*, scale=(1.0, 1.0)) -> AccumulationCheckpoint:
    input_shape = (3, 6)
    map_shape = (
        int(np.ceil((input_shape[0] - 1) * scale[0])) + 1,
        int(np.ceil((input_shape[1] - 1) * scale[1])) + 1,
    )
    input_z = 1000.0
    input_x = 2000.0
    ulm_z = input_z / scale[0]
    ulm_x = input_x / scale[1]
    return AccumulationCheckpoint(
        schema_version=2,
        mode="none",
        input_shape=input_shape,
        map_shape=map_shape,
        ulm_scale_z=scale[0],
        ulm_scale_x=scale[1],
        params={
            "movingAverageSpan": 1,
            "maxSamplingStepUlmpx": 0.8,
            "prfHz": 1000.0,
            "antiAliasSize": np.sqrt(0.5),
            "counterPower": 1.0 / 3.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 3000.0,
        },
        metadata={
            "inputPixelSizeZUm": input_z,
            "inputPixelSizeXUm": input_x,
            "ulmPixelSizeZUm": ulm_z,
            "ulmPixelSizeXUm": ulm_x,
        },
        contributions=None,
        source_path=Path("checkpoint.mat"),
    )


def make_session() -> CorrectionSession:
    return CorrectionSession(
        localized_by_frame={
            0: np.asarray(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            1: np.asarray(
                [
                    [1.0, 1.0, 3.0, 2.0],
                    [1.0, 1.0, 3.0, 2.0],
                    [1.0, 3.0, 3.0, 2.0],
                ]
            ),
            2: np.asarray(
                [
                    [1.0, 1.0, 6.0, 3.0],
                    [1.0, 1.0, 6.0, 3.0],
                ]
            ),
        },
        tracks=[
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 1.0, 1.0]),
        ],
        image_stack=np.ones((3, 6, 3), dtype=np.float32),
    )


def test_settings_keep_anisotropic_checkpoint_pixel_grid():
    settings = DatasetAccumulationSettings.from_checkpoint(
        make_checkpoint(scale=(2.0, 4.0))
    )

    assert settings.map_shape == (5, 21)
    assert settings.ulm_scale_z == 2.0
    assert settings.ulm_scale_x == 4.0
    assert settings.ulm_pixel_size_z_um == 500.0
    assert settings.ulm_pixel_size_x_um == 500.0


def test_moving_average_matches_pala_endpoint_behavior():
    coords = np.column_stack(
        [np.asarray([0.0, 0.0, 9.0, 0.0, 0.0]), np.zeros(5)]
    )

    smoothed = pala_moving_average(coords, span=6)

    np.testing.assert_allclose(smoothed[:, 0], [0.0, 3.0, 1.8, 3.0, 0.0])
    np.testing.assert_array_equal(smoothed[[0, -1]], coords[[0, -1]])


def test_current_tracks_build_all_four_pala_maps():
    accumulator = DatasetAccumulator.build(make_session(), make_checkpoint())

    rows, cols, density = accumulator.map_values(HARD_DENSITY)
    assert len(rows) == 6
    np.testing.assert_array_equal(rows, np.zeros(6, dtype=int))
    np.testing.assert_array_equal(cols, np.arange(6))
    np.testing.assert_array_equal(density, np.full(6, 2.0))

    aa_rows, aa_cols, aa_density = accumulator.map_values(AA_DENSITY)
    assert len(aa_rows) == len(aa_cols) == len(aa_density)
    assert float(aa_density.sum()) > float(density.sum())

    local_rows, local_cols, local_velocity = accumulator.map_values(
        HARD_LOCAL_VELOCITY
    )
    np.testing.assert_array_equal(local_rows, rows)
    np.testing.assert_array_equal(local_cols, cols)
    np.testing.assert_allclose(local_velocity, [4000, 4000, 4000, 6000, 6000, 6000])

    mean_rows, mean_cols, mean_velocity = accumulator.map_values(
        HARD_TRACK_MEAN_VELOCITY
    )
    np.testing.assert_array_equal(mean_rows, rows)
    np.testing.assert_array_equal(mean_cols, cols)
    np.testing.assert_allclose(mean_velocity, 5000.0)


def test_multiple_datasets_sum_maps_and_counts_without_replacement_cache():
    first = make_session()
    second = make_session()

    accumulator = DatasetAccumulator.build_multiple(
        iter(
            (
                (first, make_checkpoint()),
                (second, make_checkpoint()),
            )
        )
    )

    assert accumulator.dataset_count == 2
    assert accumulator.track_count == 4
    assert accumulator.revision == 1
    _rows, _cols, density = accumulator.map_values(HARD_DENSITY)
    np.testing.assert_array_equal(density, np.full(6, 4.0))


def test_multiple_datasets_reject_incompatible_accumulation_settings():
    with pytest.raises(ValueError, match="dataset 2.*incompatible.*map_shape"):
        DatasetAccumulator.build_multiple(
            (
                (make_session(), make_checkpoint()),
                (make_session(), make_checkpoint(scale=(2.0, 1.0))),
            )
        )


def test_anisotropic_grid_scales_track_coordinates_without_axis_swap():
    accumulator = DatasetAccumulator.build(
        make_session(),
        make_checkpoint(scale=(2.0, 4.0)),
    )

    rows, cols, density = accumulator.map_values(HARD_DENSITY)

    np.testing.assert_array_equal(rows, np.zeros(21, dtype=int))
    np.testing.assert_array_equal(cols, np.arange(21))
    np.testing.assert_array_equal(density, np.full(21, 2.0))


def test_incremental_track_replacement_matches_clean_full_rebuild():
    session = make_session()
    checkpoint = make_checkpoint()
    accumulator = DatasetAccumulator.build(session, checkpoint)
    revision = accumulator.revision

    session.assign_detection(1, 1, 2)
    accumulator.update_tracks(session, [1])
    clean = DatasetAccumulator.build(session, checkpoint)

    assert accumulator.revision == revision + 1
    np.testing.assert_allclose(accumulator.density, clean.density)
    np.testing.assert_allclose(accumulator.density_aa, clean.density_aa, atol=1e-6)
    np.testing.assert_allclose(
        accumulator.local_velocity_sum,
        clean.local_velocity_sum,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        accumulator.track_velocity_sum,
        clean.track_velocity_sum,
        atol=1e-5,
    )


def test_dataset_density_rendering_normalizes_before_gamma_and_unbounded_clim():
    rendered = render_sparse_dataset_map(
        np.asarray([0, 0, 0]),
        np.asarray([0, 1, 2]),
        np.asarray([0.0, 1.0, 4.0]),
        HARD_DENSITY,
        {
            "counterPower": 1.0 / 3.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 20.0,
        },
        1.0,
        1.0,
        density_rgb=(255, 64, 128),
        density_clim=(0.0, 2.0),
        gamma=0.5,
    )

    assert rendered is not None
    np.testing.assert_array_equal(
        rendered.rgba,
        [
            [
                [255, 64, 128, 0],
                [255, 64, 128, 64],
                [255, 64, 128, 128],
            ]
        ],
    )


def test_dataset_rendering_is_invariant_to_raw_value_scale():
    kwargs = {
        "rows": np.asarray([0, 0, 0]),
        "cols": np.asarray([0, 1, 2]),
        "map_name": AA_DENSITY,
        "params": {
            "counterPower": 1.0 / 3.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 20.0,
        },
        "ulm_scale_z": 1.0,
        "ulm_scale_x": 1.0,
        "density_clim": (0.1, 1.4),
        "gamma": 0.4,
    }
    original = render_sparse_dataset_map(
        values=np.asarray([1.0, 5.0, 20.0]),
        **kwargs,
    )
    scaled = render_sparse_dataset_map(
        values=np.asarray([1000.0, 5000.0, 20_000.0]),
        **kwargs,
    )

    assert original is not None
    assert scaled is not None
    np.testing.assert_array_equal(original.rgba, scaled.rgba)


def test_dataset_velocity_uses_the_same_normalized_gamma_clim_pipeline():
    rendered = render_sparse_dataset_map(
        np.asarray([0, 0]),
        np.asarray([0, 1]),
        np.asarray([10.0, 20.0]),
        HARD_LOCAL_VELOCITY,
        {},
        1.0,
        1.0,
        velocity_clim=(0.0, 2.0),
        gamma=1.0,
    )

    assert rendered is not None
    np.testing.assert_array_equal(
        rendered.rgba,
        [
            [
                [0, 128, 255, 255],
                [128, 255, 128, 255],
            ]
        ],
    )
