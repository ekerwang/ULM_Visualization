import numpy as np
import pytest

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.pala_contracts import (
    localized_by_frame_to_mat_tracking,
    split_mat_tracking_by_frame,
    tracks_from_track_lines_array,
    tracks_to_track_lines_array,
)


def test_split_and_flatten_mat_tracking_round_trip():
    mat_tracking = np.array(
        [
            [10.0, 2.5, 4.5, 1.0],
            [11.0, 3.0, 5.0, 1.0],
            [12.0, 8.0, 9.0, 2.0],
        ]
    )

    localized = split_mat_tracking_by_frame(mat_tracking)

    assert sorted(localized) == [0, 1]
    assert localized[0].shape == (2, 4)
    assert localized[1].shape == (1, 4)
    np.testing.assert_allclose(
        localized_by_frame_to_mat_tracking(localized),
        mat_tracking,
    )


def test_correction_session_assigns_detection_without_changing_localization_shape():
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0]]),
    }
    tracks = [np.array([0.0, np.nan])]

    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    session.assign_detection(track_id=0, frame_idx=1, local_idx=0)

    assert session.tracks[0].tolist() == [0.0, 0.0]
    assert session.localized_by_frame[1].shape == (1, 4)
    assert session.edit_log[-1].action == "assign_detection"


def test_manual_detection_appends_pala_row_and_assigns_it():
    localized = {0: np.array([[10.0, 2.5, 4.5, 1.0]])}
    tracks = [np.array([np.nan])]

    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    local_idx = session.append_manual_detection(
        frame_idx=0,
        intensity=20.0,
        z=3.25,
        x=5.25,
        track_id=0,
    )

    assert local_idx == 1
    assert session.localized_by_frame[0].shape == (2, 4)
    assert session.tracks[0][0] == 1
    assert session.localized_by_frame[0][1].tolist() == [20.0, 3.25, 5.25, 1.0]


def test_track_lines_uses_authoritative_local_idx_for_duplicate_coordinates():
    localized = {
        0: np.array(
            [
                [10.0, 2.0, 3.0, 1.0],
                [20.0, 2.0, 3.0, 1.0],
            ]
        )
    }

    tracks = tracks_from_track_lines_array(
        np.array([[0.0, 0.0, 2.0, 3.0, 1.0]]),
        localized,
        n_frames=1,
    )

    assert tracks[0][0] == 1


def test_track_lines_accepts_squeezed_single_row_and_serialization_noise():
    localized = {0: np.array([[10.0, 2.0, 3.0, 1.0]])}

    tracks = tracks_from_track_lines_array(
        np.array([0.0, 0.0, 2.0 + 5e-10, 3.0 - 5e-10, 0.0]),
        localized,
        n_frames=1,
    )

    assert tracks[0].tolist() == [0.0]


def test_empty_track_lines_is_a_valid_empty_track_list():
    assert tracks_from_track_lines_array(np.empty((0, 5)), {}, n_frames=3) == []


def test_track_lines_export_matches_upstream_five_column_contract():
    localized = {
        0: np.array([[10.0, 2.0, 3.0, 1.0]]),
        1: np.array([[20.0, 4.0, 5.0, 2.0]]),
    }
    tracks = [
        np.array([0.0, np.nan]),
        np.array([np.nan, 0.0]),
    ]

    track_lines = tracks_to_track_lines_array(tracks, localized)

    np.testing.assert_allclose(
        track_lines,
        [
            [0.0, 0.0, 2.0, 3.0, 0.0],
            [1.0, 1.0, 4.0, 5.0, 0.0],
        ],
    )


def test_empty_track_lines_can_restore_trailing_correction_slots():
    tracks = tracks_from_track_lines_array(
        np.empty((0, 5)),
        {},
        n_frames=2,
        n_track_slots=3,
    )

    assert len(tracks) == 3
    assert all(np.all(np.isnan(track)) for track in tracks)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (0, -1.0),
        (0, 0.5),
        (0, np.nan),
        (0, float(2**63)),
        (1, np.inf),
        (4, -np.inf),
        (4, 1.8),
    ],
)
def test_track_lines_rejects_invalid_id_and_index_values(column, value):
    localized = {0: np.array([[10.0, 2.0, 3.0, 1.0]])}
    row = np.array([[0.0, 0.0, 2.0, 3.0, 0.0]])
    row[0, column] = value

    with pytest.raises(ValueError, match="finite non-negative integers"):
        tracks_from_track_lines_array(row, localized, n_frames=1)


@pytest.mark.parametrize(
    "track_lines",
    [
        np.array([[0.0, 0.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 2.0, 3.0, 0.0, 99.0]]),
        np.ones((1, 1, 5)),
    ],
)
def test_track_lines_requires_exactly_five_columns(track_lines):
    with pytest.raises(ValueError, match=r"shaped \(N, 5\).+current Batch Processing"):
        tracks_from_track_lines_array(track_lines, {}, n_frames=1)


def test_track_lines_rejects_frame_outside_stack():
    localized = {1: np.array([[10.0, 2.0, 3.0, 2.0]])}

    with pytest.raises(ValueError, match="outside.+n_frames"):
        tracks_from_track_lines_array(
            np.array([[0.0, 1.0, 2.0, 3.0, 0.0]]),
            localized,
            n_frames=1,
        )


def test_track_lines_rejects_missing_localization_frame():
    with pytest.raises(ValueError, match="missing localization frame.+local_idx=0"):
        tracks_from_track_lines_array(
            np.array([[0.0, 0.0, 2.0, 3.0, 0.0]]),
            {},
            n_frames=1,
        )


def test_track_lines_rejects_local_idx_outside_frame_table():
    localized = {0: np.array([[10.0, 2.0, 3.0, 1.0]])}

    with pytest.raises(ValueError, match="local_idx=1 outside frame 0"):
        tracks_from_track_lines_array(
            np.array([[0.0, 0.0, 2.0, 3.0, 1.0]]),
            localized,
            n_frames=1,
        )


def test_track_lines_rejects_duplicate_track_frame_assignment():
    localized = {
        0: np.array(
            [
                [10.0, 2.0, 3.0, 1.0],
                [20.0, 4.0, 5.0, 1.0],
            ]
        )
    }
    track_lines = np.array(
        [
            [0.0, 0.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0, 5.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="more than one assignment"):
        tracks_from_track_lines_array(track_lines, localized, n_frames=1)


def test_track_lines_preserves_sparse_track_ids_with_empty_slots():
    localized = {
        0: np.array(
            [
                [10.0, 2.0, 3.0, 1.0],
                [20.0, 4.0, 5.0, 1.0],
            ]
        )
    }

    tracks = tracks_from_track_lines_array(
        np.array(
            [
                [2.0, 0.0, 2.0, 3.0, 0.0],
                [5.0, 0.0, 4.0, 5.0, 1.0],
            ]
        ),
        localized,
        n_frames=1,
    )

    assert len(tracks) == 6
    assert tracks[2].tolist() == [0.0]
    assert tracks[5].tolist() == [1.0]
    for track_id in (0, 1, 3, 4):
        assert np.isnan(tracks[track_id][0])


def test_track_lines_rejects_coordinate_mismatch_without_fallback():
    localized = {
        0: np.array(
            [
                [10.0, 2.0, 3.0, 1.0],
                [20.0, 4.0, 5.0, 1.0],
            ]
        )
    }

    with pytest.raises(
        ValueError,
        match=r"track_id=0, frame_idx=0, local_idx=1.+will not substitute",
    ):
        tracks_from_track_lines_array(
            np.array([[0.0, 0.0, 2.0, 3.0, 1.0]]),
            localized,
            n_frames=1,
        )
