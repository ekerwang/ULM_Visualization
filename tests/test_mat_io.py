import numpy as np
import pytest
from scipy.io import loadmat, savemat

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.io import mat_io
from ulm_track_correction_gui.io.editlog import replay_edits
from ulm_track_correction_gui.io.mat_io import (
    load_correction_session,
    save_correction_session,
    save_correction_session_atomic,
)


def test_save_and_load_correction_session(tmp_path):
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0]]),
    }
    tracks = [np.array([0.0, 0.0])]
    stack = np.zeros((4, 5, 2), dtype=float)
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=tracks,
        image_stack=stack,
        metadata={"units": "pixel"},
    )
    session.set_track_status(0, "verified")

    out_path = tmp_path / "corrected.mat"
    save_correction_session(session, out_path)
    loaded = load_correction_session(out_path)

    assert loaded.n_frames == 2
    assert loaded.n_tracks == 1
    assert loaded.track_status[0] == "verified"
    np.testing.assert_allclose(loaded.tracks[0], np.array([0.0, 0.0]))


def test_save_uses_upstream_track_lines_and_omits_stale_derived_outputs(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
        track_status={0: "verified"},
    )
    out_path = tmp_path / "clean_corrected.mat"

    save_correction_session(session, out_path)
    raw = loadmat(out_path)
    saved_keys = {key for key in raw if not key.startswith("__")}

    assert saved_keys == {
        "localized_points",
        "track_lines",
        "manual_detections",
        "n_frames",
        "metadata_json",
        "track_status_json",
        "track_notes_json",
        "edit_log_json",
        "audit_log_segments_json",
    }
    np.testing.assert_allclose(raw["track_lines"], [[0.0, 0.0, 2.0, 3.0, 0.0]])
    assert not {
        "MatTracking",
        "corrected_tracks_matrix",
        "corrected_track_points",
        "corrected_track_paths",
        "map_counter",
        "map_counter_aa",
        "velocity_displacement",
        "velocity_track_mean",
        "mapCounter",
        "mapCounter_AA",
        "VelocityDisplacement",
        "VelocityTrackMean",
    } & saved_keys


def test_save_preserves_track_ids_and_empty_slots_after_merge(tmp_path):
    localized = {
        0: np.array([[10.0, 2.0, 2.0, 1.0]]),
        1: np.array([[11.0, 3.0, 3.0, 2.0]]),
    }
    tracks = [
        np.array([0.0, np.nan]),
        np.array([np.nan, np.nan]),  # empty husk (merged away)
        np.array([np.nan, 0.0]),
    ]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    session.set_track_status(0, "flagged")
    session.set_track_status(2, "verified")

    out_path = tmp_path / "corrected.mat"
    save_correction_session(session, out_path)
    raw = loadmat(out_path)
    loaded = load_correction_session(out_path)

    # Correction metadata restores empty slots that have no track_lines rows,
    # so the live track keeps the same Track ID shown in the GUI.
    assert loaded.n_tracks == 3
    assert loaded.retired_track_ids == {0, 1}
    assert loaded.track_status[2] == "verified"
    assert loaded.metadata["track_id_remap"] == {"2": 2}
    assert loaded.metadata["exported_verified_track_ids"] == [2]
    assert loaded.metadata["session_track_id_to_export_id"] == {
        "0": 0,
        "1": 1,
        "2": 2,
    }
    np.testing.assert_array_equal(
        np.nan_to_num(loaded.tracks[2], nan=-1), [-1, 0]
    )
    assert np.all(np.isnan(loaded.tracks[0]))
    assert np.unique(raw["track_lines"][:, 0]).astype(int).tolist() == [2]
    assert "corrected_tracks_matrix" not in raw
    assert "corrected_track_points" not in raw
    assert "corrected_track_paths" not in raw
    # in-session state untouched by saving
    assert session.n_tracks == 3


def test_save_preserves_sparse_gui_track_ids_in_all_tracking_outputs(tmp_path):
    gui_track_ids = [2, 14, 95, 155, 179, 183, 221, 223]
    localized = {
        0: np.array(
            [
                [10.0 + local_idx, 2.0 + local_idx, 3.0, 1.0]
                for local_idx in range(len(gui_track_ids))
            ]
        )
    }
    tracks = [np.array([np.nan]) for _track_id in range(max(gui_track_ids) + 1)]
    for local_idx, track_id in enumerate(gui_track_ids):
        tracks[track_id][0] = local_idx
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=tracks,
        track_status={track_id: "verified" for track_id in gui_track_ids},
    )

    out_path = tmp_path / "sparse_track_ids.mat"
    save_correction_session(session, out_path)
    raw = loadmat(out_path)
    loaded = load_correction_session(out_path)

    assert raw["track_lines"].shape == (len(gui_track_ids), 5)
    assert np.unique(raw["track_lines"][:, 0]).astype(int).tolist() == gui_track_ids
    assert loaded.active_nonempty_track_ids == gui_track_ids
    assert [
        track_id
        for track_id, status in loaded.track_status.items()
        if status == "verified"
    ] == gui_track_ids
    assert loaded.initialize_track() == 224


def test_manual_detection_marker_survives_save_load(tmp_path):
    localized = {0: np.array([[10.0, 2.5, 4.5, 1.0]])}
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=[np.array([0.0])],
    )
    local_idx = session.append_manual_detection(0, intensity=5.0, z=7.0, x=8.0)

    out_path = tmp_path / "corrected.mat"
    save_correction_session(session, out_path)
    loaded = load_correction_session(out_path)

    assert loaded.is_manual_detection(0, local_idx)
    assert loaded.manual_detections == {(0, local_idx)}
    np.testing.assert_allclose(
        loaded.localized_by_frame[0][local_idx], [5.0, 7.0, 8.0, 1.0]
    )


def test_discarded_manual_detection_is_omitted_and_later_indices_are_remapped(
    tmp_path,
):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.5, 4.5, 1.0]])},
        tracks=[np.array([0.0])],
    )
    discarded_idx = session.append_manual_detection(
        0,
        intensity=5.0,
        z=7.0,
        x=8.0,
    )
    retained_idx = session.append_manual_detection(
        0,
        intensity=6.0,
        z=9.0,
        x=10.0,
        track_id=0,
    )
    session.discard_manual_detection(0, discarded_idx)
    session.set_track_status(0, "verified")

    out_path = tmp_path / "corrected.mat"
    save_correction_session_atomic(session, out_path)
    loaded = load_correction_session(out_path)

    assert session.localized_by_frame[0].shape == (3, 4)
    assert session.local_idx_for_track_frame(0, 0) == retained_idx == 2
    assert loaded.localized_by_frame[0].shape == (2, 4)
    assert loaded.local_idx_for_track_frame(0, 0) == 1
    assert loaded.manual_detections == {(0, 1)}
    assert loaded.discarded_manual_detections == set()
    assert loaded.metadata["session_local_idx_to_export_idx"] == {
        "0": {"0": 0, "2": 1}
    }
    assert loaded.audit_segments[-1]["session_local_idx_to_export_idx"] == {
        "0": {"0": 0, "2": 1}
    }
    np.testing.assert_allclose(
        loaded.localized_by_frame[0],
        [[10.0, 2.5, 4.5, 1.0], [6.0, 9.0, 10.0, 1.0]],
    )


def test_repeated_save_composes_archived_manual_detection_index_mapping(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.5, 4.5, 1.0]])},
        tracks=[np.array([0.0])],
    )
    first_discarded_idx = session.append_manual_detection(
        0,
        intensity=5.0,
        z=7.0,
        x=8.0,
    )
    retained_idx = session.append_manual_detection(
        0,
        intensity=6.0,
        z=9.0,
        x=10.0,
        track_id=0,
    )
    session.discard_manual_detection(0, first_discarded_idx)

    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"
    save_correction_session(session, first)

    reopened = load_correction_session(first)
    second_discarded_idx = reopened.append_manual_detection(
        0,
        intensity=7.0,
        z=11.0,
        x=12.0,
    )
    reopened.discard_manual_detection(0, second_discarded_idx)
    save_correction_session(reopened, second)
    reloaded = load_correction_session(second)

    assert retained_idx == 2
    assert reloaded.localized_by_frame[0].shape == (2, 4)
    assert reloaded.metadata["session_local_idx_to_export_idx"] == {
        "0": {"0": 0, "1": 1}
    }
    assert len(reloaded.audit_segments) == 2
    assert reloaded.audit_segments[0]["session_local_idx_to_export_idx"] == {
        "0": {"0": 0, "2": 1}
    }
    assert reloaded.audit_segments[1]["session_local_idx_to_export_idx"] == {
        "0": {"0": 0, "1": 1}
    }


def test_repeated_save_composes_original_track_id_remap(tmp_path):
    localized = {
        0: np.array([[10.0, 2.0, 2.0, 1.0]]),
        1: np.array([[11.0, 3.0, 3.0, 2.0]]),
    }
    # This corrected session came from source tracks 2, 5, and 9.
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=[
            np.array([0.0, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, 0.0]),
        ],
        metadata={"track_id_remap": {"2": 0, "5": 1, "9": 2}},
    )
    session.set_track_status(0, "verified")
    session.set_track_status(2, "verified")

    out_path = tmp_path / "corrected_again.mat"
    save_correction_session(session, out_path)
    loaded = load_correction_session(out_path)

    assert loaded.metadata["track_id_remap"] == {"2": 0, "9": 2}


def test_atomic_save_creates_parent_and_replaces_existing_result(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
    )
    out_path = tmp_path / "corrected" / "source_results.mat"

    save_correction_session_atomic(session, out_path)
    first_size = out_path.stat().st_size
    session.set_track_status(0, "verified")
    save_correction_session_atomic(session, out_path)

    assert first_size > 0
    assert load_correction_session(out_path).track_status[0] == "verified"
    assert not list(out_path.parent.glob("*.tmp.mat"))


def test_atomic_save_failure_preserves_previous_result(tmp_path, monkeypatch):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
    )
    out_path = tmp_path / "corrected" / "source_results.mat"
    out_path.parent.mkdir()
    out_path.write_bytes(b"previous corrected result")

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(mat_io, "save_correction_session", fail_save)
    with pytest.raises(RuntimeError, match="simulated export failure"):
        save_correction_session_atomic(session, out_path)

    assert out_path.read_bytes() == b"previous corrected result"
    assert not list(out_path.parent.glob("*.tmp.mat"))


def test_loads_current_batch_track_lines_without_coordinate_fallback(tmp_path):
    path = tmp_path / "batch_results.mat"
    localized_points = np.array(
        [
            [0.0, 10.0, 2.0, 3.0, 1.0],
            [0.0, 20.0, 2.0, 3.0, 1.0],
            [1.0, 30.0, 4.0, 5.0, 2.0],
            [2.0, 40.0, 6.0, 7.0, 3.0],
        ]
    )
    track_lines = np.array(
        [
            [0.0, 0.0, 2.0, 3.0, 1.0],
            [0.0, 2.0, 6.0, 7.0, 0.0],
            [1.0, 1.0, 4.0, 5.0, 0.0],
        ]
    )
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((8, 9, 3)),
            "localized_points": localized_points,
            "track_lines": track_lines,
        },
    )

    session = load_correction_session(path)

    assert session.n_frames == 3
    assert session.n_tracks == 2
    np.testing.assert_allclose(session.tracks[0], [1.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(session.tracks[1], [np.nan, 0.0, np.nan], equal_nan=True)
    assert sum(np.count_nonzero(~np.isnan(track)) for track in session.tracks) == 3
    assert "import_warnings" not in session.metadata


def test_loads_sparse_batch_track_ids_without_renumbering(tmp_path):
    path = tmp_path / "sparse_track_ids.mat"
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((8, 9, 1)),
            "localized_points": np.array(
                [
                    [0.0, 10.0, 2.0, 3.0, 1.0],
                    [0.0, 20.0, 4.0, 5.0, 1.0],
                ]
            ),
            "track_lines": np.array(
                [
                    [2.0, 0.0, 2.0, 3.0, 0.0],
                    [5.0, 0.0, 4.0, 5.0, 1.0],
                ]
            ),
        },
    )

    session = load_correction_session(path)

    assert session.n_tracks == 6
    assert session.active_nonempty_track_ids == [2, 5]
    assert session.retired_track_ids == {0, 1, 3, 4}
    assert session.source_track_ids == {2: 2, 5: 5}
    assert session.tracks[2].tolist() == [0.0]
    assert session.tracks[5].tolist() == [1.0]

    session.set_track_status(2, "verified")
    session.set_track_status(5, "verified")
    corrected_path = tmp_path / "sparse_track_ids_corrected.mat"
    save_correction_session(session, corrected_path)
    reloaded = load_correction_session(corrected_path)

    assert reloaded.n_tracks == 6
    assert reloaded.active_nonempty_track_ids == [2, 5]
    assert reloaded.retired_track_ids == {0, 1, 3, 4}
    assert reloaded.source_track_ids == {2: 2, 5: 5}


def test_loads_single_track_line_squeezed_by_loadmat(tmp_path):
    path = tmp_path / "single_track_line.mat"
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((2, 2, 1)),
            "localized_points": np.array([[0.0, 10.0, 2.0, 3.0, 1.0]]),
            "track_lines": np.array([[0.0, 0.0, 2.0, 3.0, 0.0]]),
        },
    )

    session = load_correction_session(path)

    assert session.n_frames == 1
    assert session.tracks[0].tolist() == [0.0]


def test_empty_track_lines_is_a_valid_batch_tracking_source(tmp_path):
    path = tmp_path / "empty_tracks.mat"
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((2, 2, 2)),
            "localized_points": np.empty((0, 5)),
            "track_lines": np.empty((0, 5)),
        },
    )

    session = load_correction_session(path)

    assert session.n_frames == 2
    assert session.tracks == []


@pytest.mark.parametrize(
    "legacy_key",
    ["tracks", "track_points", "corrected_track_points", "corrected_track_paths"],
)
def test_legacy_tracking_keys_are_explicitly_rejected(tmp_path, legacy_key):
    path = tmp_path / f"legacy_{legacy_key}.mat"
    legacy_value = (
        np.array([[0.0]])
        if legacy_key == "tracks"
        else np.array([[0.0, 0.0, 2.0, 3.0, 0.0]])
    )
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((2, 2, 1)),
            "localized_points": np.array([[0.0, 10.0, 2.0, 3.0, 1.0]]),
            legacy_key: legacy_value,
        },
    )

    with pytest.raises(
        ValueError,
        match=rf"unsupported legacy key\(s\): {legacy_key}.+track_lines shaped \(N, 5\)",
    ):
        load_correction_session(path)


def test_four_column_track_lines_are_explicitly_rejected(tmp_path):
    path = tmp_path / "old_track_lines.mat"
    savemat(
        path,
        {
            "absIQ_SVDfiltered": np.zeros((2, 2, 1)),
            "localized_points": np.array([[0.0, 10.0, 2.0, 3.0, 1.0]]),
            "track_lines": np.array([[0.0, 0.0, 2.0, 3.0]]),
        },
    )

    with pytest.raises(ValueError, match="Legacy 4-column.+current Batch Processing"):
        load_correction_session(path)


def test_missing_tracking_source_is_explicitly_rejected(tmp_path):
    path = tmp_path / "missing_tracking.mat"
    savemat(path, {"absIQ_SVDfiltered": np.zeros((2, 2, 1))})

    with pytest.raises(
        ValueError,
        match=r"track_lines shaped \(N, 5\).+corrected_tracks_matrix",
    ):
        load_correction_session(path)


def test_corrected_round_trip_uses_track_lines_not_legacy_tracking_fields(tmp_path):
    localized = {
        0: np.array([[10.0, 2.0, 3.0, 1.0]]),
        1: np.array(
            [
                [11.0, 4.0, 5.0, 2.0],
                [12.0, 6.0, 7.0, 2.0],
            ]
        ),
        2: np.array([[13.0, 8.0, 9.0, 3.0]]),
    }
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=[
            np.array([0.0, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, 0.0, 0.0]),
        ],
        track_status={2: "verified"},
        track_notes={2: "keep this note"},
        manual_detections={(1, 1)},
    )
    path = tmp_path / "corrected_track_lines.mat"
    save_correction_session(session, path)

    saved = {
        key: value
        for key, value in loadmat(path).items()
        if not key.startswith("__")
    }
    saved["tracks"] = np.array([[99.0]])
    saved["track_points"] = np.array([[99.0]])
    saved["corrected_track_points"] = np.array([[99.0]])
    saved["corrected_track_paths"] = np.array([[99.0]])
    saved["corrected_tracks_matrix"] = np.array([[99.0]])
    savemat(path, saved)

    loaded = load_correction_session(path)

    assert loaded.n_tracks == 3
    np.testing.assert_allclose(
        loaded.tracks[0], [np.nan, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(loaded.tracks[1], [np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(loaded.tracks[2], [np.nan, 0.0, 0.0], equal_nan=True)
    assert loaded.retired_track_ids == {0, 1}
    assert loaded.metadata["track_id_remap"] == {"2": 2}
    assert loaded.metadata["exported_verified_track_ids"] == [2]
    assert loaded.track_status[2] == "verified"
    assert loaded.track_notes[2] == "keep this note"
    assert loaded.manual_detections == {(1, 1)}


def test_export_split_children_have_lineage_but_no_false_source_mapping(tmp_path):
    localized = {
        0: np.array([[10.0, 2.0, 3.0, 1.0]]),
        1: np.array([[11.0, 4.0, 5.0, 2.0]]),
    }
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=[np.array([0.0, 0.0])],
        image_stack=np.zeros((2, 2, 2)),
    )
    split = session.split_track(0, split_after_frame=0)
    draft_id = session.initialize_track()

    path = tmp_path / "split_corrected.mat"
    save_correction_session(session, path)
    loaded = load_correction_session(path)

    assert loaded.n_tracks == 4
    assert loaded.retired_track_ids == {0, 1, 2, 3}
    assert loaded.active_empty_track_ids == []
    assert loaded.metadata["track_id_remap"] == {}
    assert loaded.metadata["session_track_id_to_export_id"] == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }
    assert loaded.metadata["active_empty_track_ids"] == []
    assert loaded.metadata["exported_verified_track_ids"] == []
    assert loaded.source_track_ids == {}
    assert loaded.track_status[split["first_child_id"]] == "unreviewed"
    assert loaded.track_status[split["second_child_id"]] == "unreviewed"
    assert loaded.track_lineage == {}
    assert draft_id == 3


def test_export_merge_preserves_only_surviving_source_checkpoint_mapping(tmp_path):
    localized = {
        0: np.array([[10.0, 2.0, 3.0, 1.0]]),
        1: np.array([[11.0, 4.0, 5.0, 2.0]]),
    }
    session = CorrectionSession(
        localized_by_frame=localized,
        tracks=[np.array([0.0, np.nan]), np.array([np.nan, 0.0])],
    )
    session.merge_tracks(0, 1)
    session.set_track_status(0, "verified")

    path = tmp_path / "merged_source_mapping.mat"
    save_correction_session(session, path)
    loaded = load_correction_session(path)

    assert loaded.metadata["track_id_remap"] == {"0": 0}
    assert loaded.source_track_ids == {0: 0}
    assert loaded.retired_track_ids == {1}


@pytest.mark.parametrize("invalid_idx", [-1.0, 0.5, np.inf])
def test_corrected_matrix_rejects_invalid_local_indices(tmp_path, invalid_idx):
    path = tmp_path / "invalid_corrected.mat"
    savemat(
        path,
        {
            "localized_points": np.array([[0.0, 10.0, 2.0, 3.0, 1.0]]),
            "corrected_tracks_matrix": np.array([[invalid_idx]]),
            "n_frames": 1.0,
        },
    )

    with pytest.raises(ValueError, match="Track assignments"):
        load_correction_session(path)


def test_corrected_matrix_rejects_out_of_range_local_index(tmp_path):
    path = tmp_path / "invalid_corrected.mat"
    savemat(
        path,
        {
            "localized_points": np.array([[0.0, 10.0, 2.0, 3.0, 1.0]]),
            "corrected_tracks_matrix": np.array([[1.0]]),
            "n_frames": 1.0,
        },
    )

    with pytest.raises(ValueError, match="has only 1 localization row"):
        load_correction_session(path)


def test_embedded_audit_survives_reopen_and_resave(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
    )
    session.set_track_status(0, "verified")
    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"

    save_correction_session(session, first)
    reopened = load_correction_session(first)
    assert reopened.edit_log == []
    assert sum(len(segment["edits"]) for segment in reopened.audit_segments) == 1

    save_correction_session(reopened, second)
    reloaded = load_correction_session(second)

    assert sum(len(segment["edits"]) for segment in reloaded.audit_segments) == 1
    assert reloaded.metadata["verification_complete"] is True
    assert len(reloaded.edits_for_track_history(0)) == 1


def test_archived_audit_track_ids_stay_stable_across_repeated_exports(tmp_path):
    localizations = {
        0: np.array(
            [
                [10.0, 2.0, 3.0, 1.0],
                [11.0, 4.0, 5.0, 1.0],
                [12.0, 6.0, 7.0, 1.0],
            ]
        )
    }
    session = CorrectionSession(
        localized_by_frame=localizations,
        tracks=[np.array([0.0]), np.array([1.0]), np.array([2.0])],
    )
    session.set_track_status(2, "verified")
    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"
    save_correction_session(session, first)

    reopened = load_correction_session(first)
    reopened.set_track_status(2, "flagged")
    save_correction_session(reopened, second)
    reloaded = load_correction_session(second)

    assert reloaded.n_tracks == 3
    assert reloaded.retired_track_ids == {0, 1, 2}
    assert reloaded.track_status[2] == "unreviewed"
    archived_mapping = reloaded.audit_segments[0][
        "session_track_id_to_export_id"
    ]
    assert archived_mapping == {"0": 0, "1": 1, "2": 2}
    history = reloaded.edits_for_track_history(2)
    assert len(history) == 2
    assert all(edit.action == "set_track_status" for edit in history)
    assert all(edit.track_id == 2 for edit in history)


def test_loading_rejects_missing_embedded_audit(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
    )
    session.set_track_status(0, "verified")
    path = tmp_path / "tampered.mat"
    save_correction_session(session, path)
    raw = loadmat(path)
    raw["audit_log_segments_json"] = "[]"
    savemat(path, {key: value for key, value in raw.items() if not key.startswith("__")})

    with pytest.raises(ValueError, match="audit-log fingerprint check"):
        load_correction_session(path)


def test_loading_rejects_modified_corrected_state(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
    )
    path = tmp_path / "tampered_state.mat"
    save_correction_session(session, path)
    raw = loadmat(path)
    raw["localized_points"][0, 2] = 99.0
    savemat(path, {key: value for key, value in raw.items() if not key.startswith("__")})

    with pytest.raises(ValueError, match="correction-state fingerprint check"):
        load_correction_session(path)


def test_completed_export_allows_unresolved_tracks_but_exports_none(tmp_path):
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
        metadata={"dataset_review_state": "completed"},
    )

    path = tmp_path / "completed_with_unresolved.mat"
    save_correction_session(session, path)
    loaded = load_correction_session(path)

    assert loaded.metadata["dataset_review_state"] == "completed"
    assert loaded.metadata["verification_complete"] is False
    assert loaded.metadata["completion_override"] is True
    assert loaded.metadata["exported_verified_track_ids"] == []
    assert loaded.retired_track_ids == {0}
    assert np.all(np.isnan(loaded.tracks[0]))


def test_atomic_export_refuses_source_symlink_alias(tmp_path):
    source = tmp_path / "source.mat"
    source.write_bytes(b"precious source")
    alias = tmp_path / "alias.mat"
    alias.symlink_to(source)
    session = CorrectionSession(
        localized_by_frame={0: np.array([[10.0, 2.0, 3.0, 1.0]])},
        tracks=[np.array([0.0])],
        metadata={"source_path": str(source)},
    )

    with pytest.raises(ValueError, match="Refusing to overwrite the source MAT"):
        save_correction_session_atomic(session, alias)

    assert source.read_bytes() == b"precious source"


def test_completed_combined_correction_roundtrip_and_audit_replay(tmp_path):
    initial_localizations = {
        0: np.array([[10.0, 2.0, 3.0, 1.0]]),
        1: np.array(
            [
                [11.0, 4.0, 5.0, 2.0],
                [12.0, 6.0, 7.0, 2.0],
            ]
        ),
        2: np.empty((0, 4), dtype=float),
    }
    initial_tracks = [
        np.array([0.0, np.nan, np.nan]),
        np.array([np.nan, 0.0, np.nan]),
        np.array([np.nan, 1.0, np.nan]),
    ]
    session = CorrectionSession(
        localized_by_frame=initial_localizations,
        tracks=initial_tracks,
    )

    session.merge_tracks(0, 1)
    session.delete_track(2)
    manual_idx = session.append_manual_detection(
        2,
        intensity=13.0,
        z=8.25,
        x=9.75,
        track_id=0,
    )
    session.set_track_status(0, "verified")
    session.metadata["dataset_review_state"] = "completed"
    path = tmp_path / "completed.mat"
    save_correction_session_atomic(session, path)

    loaded = load_correction_session(path)
    assert loaded.n_tracks == 3
    assert loaded.retired_track_ids == {1, 2}
    np.testing.assert_allclose(loaded.tracks[0], [0.0, 0.0, 0.0])
    assert loaded.track_status[0] == "verified"
    assert loaded.metadata["dataset_review_state"] == "completed"
    assert loaded.metadata["verification_complete"] is True
    assert loaded.metadata["session_track_id_to_export_id"] == {
        "0": 0,
        "1": 1,
        "2": 2,
    }
    assert loaded.metadata["track_id_remap"] == {"0": 0}
    assert loaded.localized_by_frame[1].shape == (2, 4)
    assert loaded.manual_detections == {(2, manual_idx)}
    np.testing.assert_allclose(
        loaded.localized_by_frame[2][manual_idx],
        [13.0, 8.25, 9.75, 3.0],
    )

    archived_edits = [
        edit
        for segment in loaded.audit_segments
        for edit in segment["edits"]
    ]
    assert {edit["action"] for edit in archived_edits} >= {
        "assign_detection",
        "append_manual_detection",
        "remove_assignment",
        "set_track_active",
        "set_track_status",
    }
    replayed = CorrectionSession(
        localized_by_frame=initial_localizations,
        tracks=initial_tracks,
    )
    replay_edits(replayed, archived_edits)

    assert replayed.state_fingerprint() == session.state_fingerprint()
