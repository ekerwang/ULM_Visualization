import numpy as np
import pytest

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.io.editlog import replay_edits


def make_session_with_undo() -> tuple[CorrectionSession, UndoManager]:
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0], [11.0, 6.0, 7.0, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0]]),
    }
    tracks = [
        np.array([0.0, np.nan]),
        np.array([1.0, np.nan]),
    ]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    manager = UndoManager(session)
    session.edit_listener = manager.record
    return session, manager


def test_undo_redo_assignment():
    session, manager = make_session_with_undo()
    session.assign_detection(0, 1, 0)
    assert session.tracks[0][1] == 0

    manager.undo()
    assert np.isnan(session.tracks[0][1])
    assert session.track_status[0] == "unreviewed"
    assert session.track_id_for_detection(1, 0) is None

    manager.redo()
    assert session.tracks[0][1] == 0
    assert session.track_id_for_detection(1, 0) == 0


def test_undo_steal_group_is_atomic():
    session, manager = make_session_with_undo()
    session.assign_detection(0, 0, 1, steal=True)
    assert session.track_id_for_detection(0, 1) == 0
    assert np.isnan(session.tracks[1][0])

    manager.undo()
    # both the steal and the implicit removal are rolled back together
    assert session.track_id_for_detection(0, 1) == 1
    assert session.tracks[1][0] == 1
    assert session.tracks[0][0] == 0
    assert session.track_status[0] == "unreviewed"
    assert session.track_status[1] == "unreviewed"


def test_undo_manual_detection_removes_row_and_assignment():
    session, manager = make_session_with_undo()
    local_idx = session.append_manual_detection(
        1, intensity=20.0, z=3.0, x=4.0, track_id=0
    )
    assert session.localized_by_frame[1].shape == (2, 4)
    assert session.tracks[0][1] == local_idx

    manager.undo()
    assert session.localized_by_frame[1].shape == (1, 4)
    assert np.isnan(session.tracks[0][1])
    assert session.track_id_for_detection(1, local_idx) is None

    manager.redo()
    assert session.localized_by_frame[1].shape == (2, 4)
    assert session.tracks[0][1] == local_idx


def test_remove_manual_detection_guards():
    session, _ = make_session_with_undo()
    # not the last row of frame 0
    with pytest.raises(ValueError):
        session.remove_manual_detection(0, 0)
    # last row of frame 0 is assigned to track 1
    with pytest.raises(ValueError):
        session.remove_manual_detection(0, 1)


def test_undo_redo_discarded_manual_detection_keeps_row_in_place():
    session, manager = make_session_with_undo()
    local_idx = session.append_manual_detection(
        1,
        intensity=20.0,
        z=3.0,
        x=4.0,
    )
    row_count = session.localized_by_frame[1].shape[0]

    session.discard_manual_detection(1, local_idx)
    assert session.is_discarded_manual_detection(1, local_idx)

    manager.undo()
    assert not session.is_discarded_manual_detection(1, local_idx)
    assert session.localized_by_frame[1].shape[0] == row_count

    manager.redo()
    assert session.is_discarded_manual_detection(1, local_idx)
    assert session.localized_by_frame[1].shape[0] == row_count


def test_undo_status_change():
    session, manager = make_session_with_undo()
    session.set_track_status(0, "verified")
    manager.undo()
    assert session.track_status[0] == "unreviewed"
    manager.redo()
    assert session.track_status[0] == "verified"


def make_merge_session() -> tuple[CorrectionSession, UndoManager]:
    localized = {
        0: np.array([[10.0, 2.0, 2.0, 1.0]]),
        1: np.array([[11.0, 3.0, 3.0, 2.0], [12.0, 30.0, 30.0, 2.0]]),
        2: np.array([[13.0, 4.0, 4.0, 3.0]]),
    }
    tracks = [
        np.array([0.0, 0.0, np.nan]),   # frames 0-1
        np.array([np.nan, 1.0, 0.0]),   # frames 1-2 (overlaps at frame 1)
    ]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    manager = UndoManager(session)
    session.edit_listener = manager.record
    return session, manager


def test_merge_tracks_moves_drops_and_empties_source():
    session, _ = make_merge_session()
    result = session.merge_tracks(0, 1)

    assert result == {"moved": 1, "dropped": 1}
    # frame 2 point moved to track 0; overlapping frame 1 kept track 0's point
    assert session.tracks[0][2] == 0
    assert session.tracks[0][1] == 0
    assert np.all(np.isnan(session.tracks[1]))
    assert session.track_id_for_detection(1, 1) is None  # dropped, unassigned
    assert "Merged into track 0" in session.track_notes[1]
    with_pytest_raises = pytest.raises(ValueError)
    with with_pytest_raises:
        session.merge_tracks(0, 0)


def test_merge_tracks_single_undo_restores_both():
    session, manager = make_merge_session()
    before_0 = session.tracks[0].copy()
    before_1 = session.tracks[1].copy()
    session.merge_tracks(0, 1)

    assert len(manager.undo_stack) == 1  # all merge edits share one group
    manager.undo()
    np.testing.assert_array_equal(
        np.nan_to_num(session.tracks[0], nan=-1), np.nan_to_num(before_0, nan=-1)
    )
    np.testing.assert_array_equal(
        np.nan_to_num(session.tracks[1], nan=-1), np.nan_to_num(before_1, nan=-1)
    )
    assert session.track_id_for_detection(1, 1) == 1


def test_edit_log_with_undo_history_replays_to_same_state():
    session, manager = make_session_with_undo()
    session.assign_detection(0, 1, 0)
    session.append_manual_detection(1, intensity=20.0, z=3.0, x=4.0, track_id=1)
    manager.undo()
    session.set_track_status(0, "verified")
    manager.undo()
    manager.redo()

    edits = [edit.to_dict() for edit in session.edit_log]
    recovered_session, _ = make_session_with_undo()
    recovered_session.edit_listener = None
    replay_edits(recovered_session, edits)

    for track_replayed, track_orig in zip(recovered_session.tracks, session.tracks):
        np.testing.assert_array_equal(
            np.nan_to_num(track_replayed, nan=-1),
            np.nan_to_num(track_orig, nan=-1),
        )
    for frame_idx, rows in session.localized_by_frame.items():
        np.testing.assert_allclose(
            recovered_session.localized_by_frame[frame_idx], rows
        )
    assert recovered_session.track_status == session.track_status
    assert recovered_session.assignment_index == session.assignment_index


def test_initialize_track_undo_redo_reuses_same_stable_id():
    session, manager = make_session_with_undo()

    track_id = session.initialize_track()

    assert track_id == 2
    assert session.n_tracks == 3
    assert session.is_track_active(track_id)
    assert session.track_summary(track_id)["n_points"] == 0
    assert session.source_track_id_for_track(track_id) is None

    manager.undo()
    assert session.n_tracks == 2

    manager.redo()
    assert session.n_tracks == 3
    assert session.is_track_active(track_id)
    assert session.source_track_id_for_track(track_id) is None


def test_split_tracks_single_undo_restores_parent_and_redo_reuses_child_ids():
    session, manager = make_merge_session()
    before_parent = session.tracks[0].copy()
    session.set_track_status(0, "verified", note="original note")

    result = session.split_track(0, split_after_frame=0)
    first = result["first_child_id"]
    second = result["second_child_id"]

    assert (first, second) == (2, 3)
    assert not session.is_track_active(0)
    assert session.track_status[first] == "flagged"
    assert session.track_status[second] == "flagged"
    np.testing.assert_allclose(session.tracks[first], [0.0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(session.tracks[second], [np.nan, 0.0, np.nan], equal_nan=True)
    assert session.source_track_id_for_track(first) is None
    assert session.source_track_id_for_track(second) is None

    manager.undo()
    assert session.n_tracks == 2
    assert session.is_track_active(0)
    np.testing.assert_allclose(session.tracks[0], before_parent, equal_nan=True)
    assert session.track_status[0] == "verified"
    assert session.track_notes[0] == "original note"

    manager.redo()
    assert session.n_tracks == 4
    assert not session.is_track_active(0)
    assert session.track_status[2] == "flagged"
    assert session.track_status[3] == "flagged"


def test_delete_track_undo_redo_preserves_localization_rows():
    session, manager = make_merge_session()
    rows_before = {frame: rows.copy() for frame, rows in session.localized_by_frame.items()}
    track_before = session.tracks[1].copy()

    result = session.delete_track(1)

    assert result == {"removed": 2}
    assert not session.is_track_active(1)
    assert session.track_id_for_detection(1, 1) is None
    assert session.track_id_for_detection(2, 0) is None
    for frame, rows in rows_before.items():
        np.testing.assert_array_equal(session.localized_by_frame[frame], rows)

    manager.undo()
    assert session.is_track_active(1)
    np.testing.assert_allclose(session.tracks[1], track_before, equal_nan=True)

    manager.redo()
    assert not session.is_track_active(1)
