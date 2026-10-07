from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    AA_DENSITY,
    AA_LOCAL_VELOCITY,
    AA_TRACK_MEAN_VELOCITY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    HARD_DENSITY,
    HARD_LOCAL_VELOCITY,
    HARD_TRACK_MEAN_VELOCITY,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    TrackMapUnavailableError,
    current_to_checkpoint_track_ids,
    raster_crop_geometry,
    raster_pixel_scene_center,
)
from ulm_track_correction_gui.gui.track_plot_rendering import (
    DENSITY_CYAN_RGB,
    render_sparse_track_map,
)
from ulm_track_correction_gui.io.accumulation_checkpoint_io import (
    checkpoint_path_for_result,
    discover_accumulation_checkpoint,
    load_accumulation_checkpoint,
)


def checkpoint_payload(
    mode="full",
    *,
    schema_version=2,
    input_shape=(3, 4),
    scale=(2.0, 3.0),
):
    map_shape = (
        int(np.ceil((input_shape[0] - 1) * scale[0])) + 1,
        int(np.ceil((input_shape[1] - 1) * scale[1])) + 1,
    )
    zeros = np.zeros(map_shape, dtype=np.float32)
    payload = {
        "checkpointSchemaVersion": np.int32(schema_version),
        "checkpointKind": "pala_accumulation",
        "perTrackContributionMode": mode,
        "inputShape": np.asarray(input_shape, dtype=np.uint64).reshape(-1, 1),
        "mapShape": np.asarray(map_shape, dtype=np.uint64).reshape(-1, 1),
        "inputPixelSizeZUm": np.float64(20.0),
        "inputPixelSizeXUm": np.float64(30.0),
        "ulmPixelSizeZUm": np.float64(20.0 / scale[0]),
        "ulmPixelSizeXUm": np.float64(30.0 / scale[1]),
        "ulmScaleZ": np.float64(scale[0]),
        "ulmScaleX": np.float64(scale[1]),
        "rasterRowColumnIndexBase": np.int8(0),
        "sparseOffsetsIndexBase": np.int8(0),
        "sparsePixelIndexConvention": "row_col",
        "axisOrder": "z_x",
        "counterPower": np.float64(1.0),
        "counterDisplayMax": np.float64(2.0),
        "counterAADisplayMax": np.float64(4.0),
        "velocityDisplayMax": np.float64(50.0),
        "accumulation_param": {
            "movingAverageSpan": 19,
            "maxSamplingStepUlmpx": 0.8,
            "prfHz": 1000.0,
            "antiAliasSize": 0.5,
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 50.0,
        },
        "mapCounter": zeros,
        "mapCounter_AA": zeros.copy(),
        "VelocityDisplacement": zeros.copy(),
        "VelocityTrackMean": zeros.copy(),
    }
    if mode != "none":
        payload.update(
            {
                # Deliberately non-contiguous and not equal to contribution rows.
                "trackIds": np.asarray([7, 3], dtype=np.int64).reshape(-1, 1),
                "hardOffsets": np.asarray([0, 2, 3], dtype=np.uint64).reshape(-1, 1),
                "hardRows": np.asarray([0, 2, 4], dtype=np.uint32).reshape(-1, 1),
                "hardCols": np.asarray([0, 4, 9], dtype=np.uint32).reshape(-1, 1),
                "aaOffsets": np.asarray([0, 3, 4], dtype=np.uint64).reshape(-1, 1),
                "aaRows": np.asarray([0, 1, 2, 4], dtype=np.uint32).reshape(-1, 1),
                "aaCols": np.asarray([0, 1, 4, 9], dtype=np.uint32).reshape(-1, 1),
                "aaDensityWeight": np.asarray([0.5, 2.0, 1.0, 0.25], dtype=np.float32).reshape(
                    -1, 1
                ),
            }
        )
    if mode == "full":
        payload.update(
            {
                "hardLocalVelocityNumerator": np.asarray(
                    [10.0, 20.0, 30.0], dtype=np.float32
                ).reshape(-1, 1),
                "hardTrackMeanVelocityNumerator": np.asarray(
                    [15.0, 15.0, 35.0], dtype=np.float32
                ).reshape(-1, 1),
            }
        )
    return payload


def write_checkpoint(path: Path, mode="full", **kwargs) -> Path:
    savemat(path, checkpoint_payload(mode, **kwargs))
    return path


@pytest.mark.parametrize("mode", ["none", "density_only", "full"])
def test_loads_all_contribution_modes(tmp_path, mode):
    checkpoint = load_accumulation_checkpoint(write_checkpoint(tmp_path / f"{mode}.mat", mode))

    assert checkpoint.mode == mode
    assert checkpoint.input_shape == (3, 4)
    assert checkpoint.map_shape == (5, 10)
    if mode == "none":
        assert checkpoint.contributions is None
        assert checkpoint.available_track_maps(7) == ()
    elif mode == "density_only":
        assert checkpoint.available_track_maps(7) == (HARD_DENSITY, AA_DENSITY)
        with pytest.raises(TrackMapUnavailableError, match="unavailable"):
            checkpoint.track_map_values(7, HARD_LOCAL_VELOCITY)
    else:
        assert checkpoint.schema_version == 2
        assert checkpoint.supported_track_maps == (
            HARD_DENSITY,
            AA_DENSITY,
            HARD_LOCAL_VELOCITY,
            HARD_TRACK_MEAN_VELOCITY,
        )
        assert checkpoint.available_track_maps(7) == checkpoint.supported_track_maps


def test_loads_schema_v2_full_with_exactly_four_supported_maps(tmp_path):
    checkpoint = load_accumulation_checkpoint(
        write_checkpoint(tmp_path / "schema_v2_full.mat", schema_version=2)
    )

    assert CURRENT_CHECKPOINT_SCHEMA_VERSION == 2
    assert SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS == {2}
    assert checkpoint.schema_version == 2
    assert checkpoint.mode == "full"
    assert checkpoint.params["movingAverageSpan"] == 19
    assert checkpoint.params["maxSamplingStepUlmpx"] == 0.8
    assert checkpoint.supported_track_maps == (
        HARD_DENSITY,
        AA_DENSITY,
        HARD_LOCAL_VELOCITY,
        HARD_TRACK_MEAN_VELOCITY,
    )
    assert checkpoint.available_track_maps(7) == checkpoint.supported_track_maps
    np.testing.assert_allclose(
        checkpoint.track_map_values(7, HARD_LOCAL_VELOCITY)[2],
        [10.0, 20.0],
    )
    np.testing.assert_allclose(
        checkpoint.track_map_values(7, HARD_TRACK_MEAN_VELOCITY)[2],
        [15.0, 15.0],
    )
    for unavailable in (AA_LOCAL_VELOCITY, AA_TRACK_MEAN_VELOCITY):
        with pytest.raises(TrackMapUnavailableError, match="schema v2 full"):
            checkpoint.track_map_values(7, unavailable)


def test_track_ids_and_offsets_select_ragged_slices(tmp_path):
    checkpoint = load_accumulation_checkpoint(write_checkpoint(tmp_path / "full.mat"))

    contribution = checkpoint.track_contribution(3)
    np.testing.assert_array_equal(contribution.hard_rows, [4])
    np.testing.assert_array_equal(contribution.hard_cols, [9])
    np.testing.assert_array_equal(contribution.aa_density_weight, [0.25])
    assert checkpoint.has_track(0) is False
    with pytest.raises(KeyError, match="source track 0"):
        checkpoint.track_contribution(0)


def test_schema_v2_single_track_map_formulas(tmp_path):
    checkpoint = load_accumulation_checkpoint(write_checkpoint(tmp_path / "full.mat"))

    rows, cols, hard_density = checkpoint.track_map_values(7, HARD_DENSITY)
    np.testing.assert_array_equal(rows, [0, 2])
    np.testing.assert_array_equal(cols, [0, 4])
    np.testing.assert_allclose(hard_density, [1.0, 1.0])
    np.testing.assert_allclose(
        checkpoint.track_map_values(7, HARD_LOCAL_VELOCITY)[2],
        [10.0, 20.0],
    )
    np.testing.assert_allclose(
        checkpoint.track_map_values(7, HARD_TRACK_MEAN_VELOCITY)[2],
        [15.0, 15.0],
    )
    np.testing.assert_allclose(
        checkpoint.track_map_values(7, AA_DENSITY)[2],
        [0.5, 2.0, 1.0],
    )


@pytest.mark.parametrize(
    "field",
    [
        "hardLocalVelocityNumerator",
        "hardTrackMeanVelocityNumerator",
    ],
)
def test_full_mode_validates_every_velocity_numerator_length(tmp_path, field):
    payload = checkpoint_payload("full")
    payload[field] = np.asarray([1.0], dtype=np.float32)
    path = tmp_path / f"bad_{field}.mat"
    savemat(path, payload)

    with pytest.raises(ValueError, match=field):
        load_accumulation_checkpoint(path)


@pytest.mark.parametrize(
    "field",
    ["hardLocalVelocityNumerator", "hardTrackMeanVelocityNumerator"],
)
def test_schema_v2_full_requires_hard_velocity_numerators(tmp_path, field):
    payload = checkpoint_payload("full")
    del payload[field]
    path = tmp_path / f"schema_v2_missing_{field}.mat"
    savemat(path, payload)

    with pytest.raises(ValueError, match=field):
        load_accumulation_checkpoint(path)


@pytest.mark.parametrize(
    "field",
    ["mapCounter", "mapCounter_AA", "VelocityDisplacement", "VelocityTrackMean"],
)
def test_schema_v2_requires_each_global_map(tmp_path, field):
    payload = checkpoint_payload("full")
    del payload[field]
    path = tmp_path / f"schema_v2_missing_{field}.mat"
    savemat(path, payload)

    with pytest.raises(ValueError, match=field):
        load_accumulation_checkpoint(path)


def test_schema_v1_is_rejected_with_clear_error(tmp_path):
    payload = checkpoint_payload("full", schema_version=1)
    path = tmp_path / "schema_v1.mat"
    savemat(path, payload)

    with pytest.raises(ValueError, match="schema version 1; supported version is 2"):
        load_accumulation_checkpoint(path)


def test_loader_does_not_require_retired_plotted_geometry(tmp_path):
    checkpoint = load_accumulation_checkpoint(write_checkpoint(tmp_path / "minimal.mat"))
    assert checkpoint.has_track(7)
    assert "smoothedTrackPaths" not in checkpoint.metadata
    assert "interpolatedTracks" not in checkpoint.metadata


def test_singleton_map_dimension_is_preserved(tmp_path):
    checkpoint = load_accumulation_checkpoint(
        write_checkpoint(
            tmp_path / "singleton.mat",
            mode="none",
            input_shape=(1, 4),
            scale=(2.0, 3.0),
        )
    )
    assert checkpoint.map_shape == (1, 10)


def test_anisotropic_raster_scene_alignment_and_crop_position():
    assert raster_pixel_scene_center(0, 0, 2.0, 4.0) == (0.5, 0.5)
    assert raster_pixel_scene_center(6, 8, 2.0, 4.0) == (2.5, 3.5)

    geometry = raster_crop_geometry(
        np.asarray([6, 7]),
        np.asarray([8, 10]),
        2.0,
        4.0,
    )
    assert geometry.pos_x == pytest.approx(0.5 + (8 - 0.5) / 4.0)
    assert geometry.pos_y == pytest.approx(0.5 + (6 - 0.5) / 2.0)
    assert geometry.scale_x == 0.25
    assert geometry.scale_y == 0.5


def test_cropped_density_rgba_uses_cyan_weighted_alpha_and_transparent_background():
    rendered = render_sparse_track_map(
        np.asarray([2, 4]),
        np.asarray([3, 5]),
        np.asarray([1.0, 1.0]),
        HARD_DENSITY,
        {
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 50.0,
        },
        2.0,
        3.0,
    )
    assert rendered is not None
    assert rendered.rgba.shape == (3, 3, 4)
    np.testing.assert_array_equal(rendered.rgba[0, 0, :3], DENSITY_CYAN_RGB)
    np.testing.assert_array_equal(rendered.rgba[2, 2, :3], DENSITY_CYAN_RGB)
    assert rendered.rgba[0, 0, 3] == 255
    assert rendered.rgba[2, 2, 3] == 255
    np.testing.assert_array_equal(rendered.rgba[1, 1], [0, 0, 0, 0])


def test_aa_density_uses_selected_track_max_for_alpha():
    rendered = render_sparse_track_map(
        np.asarray([0, 0]),
        np.asarray([0, 1]),
        np.asarray([1.0, 4.0]),
        AA_DENSITY,
        {
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 400.0,
            "velocityDisplayMax": 50.0,
        },
        2.0,
        3.0,
    )
    assert rendered is not None
    np.testing.assert_array_equal(
        rendered.rgba[:, :, :3],
        [[[0, 216, 255], [0, 216, 255]]],
    )
    np.testing.assert_array_equal(rendered.rgba[:, :, 3], [[64, 255]])


def test_density_rgb_is_configurable_without_changing_alpha():
    rendered = render_sparse_track_map(
        np.asarray([0, 0]),
        np.asarray([0, 1]),
        np.asarray([1.0, 4.0]),
        AA_DENSITY,
        {
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 400.0,
            "velocityDisplayMax": 50.0,
        },
        2.0,
        3.0,
        density_rgb=(255, 64, 128),
    )
    assert rendered is not None
    np.testing.assert_array_equal(
        rendered.rgba[:, :, :3],
        [[[255, 64, 128], [255, 64, 128]]],
    )
    np.testing.assert_array_equal(rendered.rgba[:, :, 3], [[64, 255]])


def test_velocity_support_remains_opaque():
    rendered = render_sparse_track_map(
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([25.0]),
        HARD_LOCAL_VELOCITY,
        {
            "counterPower": 1.0,
            "counterDisplayMax": 2.0,
            "counterAADisplayMax": 4.0,
            "velocityDisplayMax": 50.0,
        },
        2.0,
        3.0,
    )
    assert rendered is not None
    assert rendered.rgba[0, 0, 3] == 255


def test_velocity_clim_controls_jet_endpoints():
    rendered = render_sparse_track_map(
        np.asarray([0, 0]),
        np.asarray([0, 1]),
        np.asarray([5.0, 15.0]),
        HARD_LOCAL_VELOCITY,
        {"velocityDisplayMax": 50.0},
        2.0,
        3.0,
        velocity_clim=(5.0, 15.0),
    )
    assert rendered is not None
    np.testing.assert_array_equal(
        rendered.rgba,
        [[[0, 0, 128, 255], [128, 0, 0, 255]]],
    )


@pytest.mark.parametrize("velocity_clim", [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)])
def test_velocity_clim_must_be_finite_and_increasing(velocity_clim):
    with pytest.raises(ValueError, match="velocity color limits"):
        render_sparse_track_map(
            np.asarray([0]),
            np.asarray([0]),
            np.asarray([1.0]),
            HARD_LOCAL_VELOCITY,
            {"velocityDisplayMax": 50.0},
            2.0,
            3.0,
            velocity_clim=velocity_clim,
        )


def test_discovers_inferred_companion_for_results_mat(tmp_path):
    result = tmp_path / "sample_results.mat"
    savemat(result, {"filtered": np.zeros((2, 2, 1))})
    companion = write_checkpoint(checkpoint_path_for_result(result), mode="none")

    discovery = discover_accumulation_checkpoint(result)
    assert discovery.path == companion.resolve()


def test_discovers_relative_checkpoint_reference(tmp_path):
    folder = tmp_path / "plotting"
    folder.mkdir()
    companion = write_checkpoint(folder / "custom.mat", mode="none")
    result = tmp_path / "sample_results.mat"
    savemat(
        result,
        {"accumulation_checkpoint_file": "plotting/custom.mat"},
    )

    discovery = discover_accumulation_checkpoint(result)
    assert discovery.path == companion.resolve()


def test_corrected_mat_discovers_original_result_companion(tmp_path):
    original = tmp_path / "sample_results.mat"
    savemat(original, {"filtered": np.zeros((2, 2, 1))})
    companion = write_checkpoint(checkpoint_path_for_result(original), mode="none")
    corrected = tmp_path / "corrected.mat"
    savemat(corrected, {"n_frames": 1})

    discovery = discover_accumulation_checkpoint(
        corrected,
        {"source_path": str(original)},
    )
    assert discovery.path == companion.resolve()


def test_corrected_track_id_remap_is_reversed_for_checkpoint_lookup():
    mapping = current_to_checkpoint_track_ids(
        {"track_id_remap": {"0": 0, "5": 1, "9": 2}},
        3,
    )
    assert mapping == {0: 0, 1: 5, 2: 9}


def test_checkpoint_input_shape_must_match_bubble_movie(tmp_path):
    checkpoint = load_accumulation_checkpoint(write_checkpoint(tmp_path / "full.mat"))
    with pytest.raises(ValueError, match="bubble movie shape"):
        checkpoint.validate_input_shape((4, 4))


def test_future_schema_version_has_clear_error(tmp_path):
    payload = checkpoint_payload("none")
    payload["checkpointSchemaVersion"] = np.int32(3)
    path = tmp_path / "future.mat"
    savemat(path, payload)
    with pytest.raises(ValueError, match="schema version 3; supported version is 2"):
        load_accumulation_checkpoint(path)
