import numpy as np
import pytest

from ulm_track_correction_gui.core.correction_session import (
    TRACK_STATUS_FLAGGED,
    TRACK_STATUS_VERIFIED,
    CorrectionSession,
    DetectionAssignmentConflict,
)
from ulm_track_correction_gui.core.pala_contracts import (
    pala_to_scene_xy,
    scene_to_pala_zx,
)


def make_session() -> CorrectionSession:
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0], [11.0, 6.0, 7.0, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0]]),
    }
    tracks = [
        np.array([0.0, np.nan]),
        np.array([1.0, np.nan]),
    ]
    return CorrectionSession(localized_by_frame=localized, tracks=tracks)


def test_assignment_index_tracks_ownership():
    session = make_session()
    assert session.track_id_for_detection(0, 0) == 0
    assert session.track_id_for_detection(0, 1) == 1
    assert session.track_id_for_detection(1, 0) is None

    session.assign_detection(0, 1, 0)
    assert session.track_id_for_detection(1, 0) == 0

    session.remove_assignment(0, 1)
    assert session.track_id_for_detection(1, 0) is None


def test_assign_conflicting_detection_raises_then_steals():
    session = make_session()
    with pytest.raises(DetectionAssignmentConflict):
        session.assign_detection(0, 0, 1)

    session.assign_detection(0, 0, 1, steal=True)
    assert session.track_id_for_detection(0, 1) == 0
    assert np.isnan(session.tracks[1][0])
    # steal logs remove + assign with a shared group id
    remove_edit, assign_edit = session.edit_log[-2:]
    assert remove_edit.action == "remove_assignment"
    assert assign_edit.action == "assign_detection"
    assert remove_edit.group_id == assign_edit.group_id


def test_import_conflict_flags_tracks():
    localized = {0: np.array([[10.0, 2.5, 4.5, 1.0]])}
    tracks = [np.array([0.0]), np.array([0.0])]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    assert len(session.assignment_conflicts) == 1
    assert session.track_status[0] == TRACK_STATUS_FLAGGED
    assert session.track_status[1] == TRACK_STATUS_FLAGGED


def test_manual_detection_payload_is_replayable():
    session = make_session()
    session.append_manual_detection(1, intensity=20.0, z=3.25, x=5.25, track_id=0)
    append_edit, assign_edit = session.edit_log[-2:]
    assert append_edit.action == "append_manual_detection"
    assert append_edit.payload == {"intensity": 20.0, "z": 3.25, "x": 5.25}
    assert append_edit.group_id == assign_edit.group_id


def test_set_track_status_is_logged():
    session = make_session()
    session.set_track_status(0, TRACK_STATUS_VERIFIED)
    edit = session.edit_log[-1]
    assert edit.action == "set_track_status"
    assert edit.payload["after"] == TRACK_STATUS_VERIFIED


def test_modifying_verified_track_flags_it_for_rereview():
    session = make_session()
    session.set_track_status(0, TRACK_STATUS_VERIFIED)

    session.assign_detection(0, 1, 0)

    assert session.track_status[0] == TRACK_STATUS_FLAGGED
    assert session.integrity_issues(require_all_verified=True) == [
        "All active non-empty tracks must be verified; unresolved IDs: [0, 1]."
    ]


def test_finalization_requires_no_empty_drafts_and_all_nonempty_verified():
    session = make_session()
    draft_id = session.initialize_track()
    session.set_track_status(0, TRACK_STATUS_VERIFIED)
    session.set_track_status(1, TRACK_STATUS_VERIFIED)

    issues = session.integrity_issues(require_all_verified=True)

    assert issues == [f"Active empty track IDs must be resolved: [{draft_id}]."]


def test_track_summary_reports_gaps():
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0]]),
        2: np.array([[12.0, 8.0, 9.0, 3.0]]),
    }
    tracks = [np.array([0.0, np.nan, 0.0])]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    summary = session.track_summary(0)
    assert summary["n_points"] == 2
    assert summary["start_frame"] == 0
    assert summary["end_frame"] == 2
    assert summary["n_gaps"] == 1


def test_manual_detection_marker_tracks_append_and_remove():
    session = make_session()
    local_idx = session.append_manual_detection(1, intensity=20.0, z=3.0, x=4.0)
    assert session.is_manual_detection(1, local_idx)
    assert not session.is_manual_detection(0, 0)

    session.remove_manual_detection(1, local_idx)
    assert not session.is_manual_detection(1, local_idx)
    assert session.manual_detections == set()


def test_discard_manual_detection_keeps_indices_stable_and_protects_assignments():
    session = make_session()
    first = session.append_manual_detection(0, intensity=20.0, z=9.0, x=10.0)
    later = session.append_manual_detection(0, intensity=21.0, z=11.0, x=12.0)
    rows_before = session.localized_by_frame[0].copy()

    session.discard_manual_detection(0, first)

    np.testing.assert_array_equal(session.localized_by_frame[0], rows_before)
    assert later == first + 1
    assert session.is_manual_detection(0, first)
    assert session.is_discarded_manual_detection(0, first)
    with pytest.raises(ValueError, match="discarded"):
        session.assign_detection(0, 0, first)
    edit = session.edit_log[-1]
    assert edit.action == "set_manual_detection_discarded"
    assert edit.payload["before"] is False
    assert edit.payload["after"] is True

    session.set_manual_detection_discarded(0, first, False)
    assert not session.is_discarded_manual_detection(0, first)

    session.assign_detection(0, 0, later)
    with pytest.raises(ValueError, match="assigned to track 0"):
        session.discard_manual_detection(0, later)


def test_find_matching_detection():
    from ulm_track_correction_gui.core.pala_contracts import find_matching_detection

    localized = {0: np.array([[10.0, 2.5, 4.5, 1.0], [11.0, 6.0, 7.0, 1.0]])}
    assert find_matching_detection(localized, 0, 2.5, 4.5) == 0
    assert find_matching_detection(localized, 0, 6.1, 7.1) == 1
    assert find_matching_detection(localized, 0, 20.0, 20.0) is None
    assert find_matching_detection(localized, 5, 2.5, 4.5) is None
    assert (
        find_matching_detection(
            localized,
            0,
            2.5,
            4.5,
            excluded_local_indices={0},
        )
        is None
    )


def test_scene_coordinate_round_trip():
    # PALA (1,1) is the first pixel center -> scene (0.5, 0.5)
    assert pala_to_scene_xy(1.0, 1.0) == (0.5, 0.5)
    z, x = scene_to_pala_zx(*pala_to_scene_xy(3.25, 5.75))
    assert (z, x) == (3.25, 5.75)
