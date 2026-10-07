import numpy as np

from ulm_track_correction_gui.core.commands import UndoManager
from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.edit_history import (
    describe_edit,
    track_history_entries,
)


def make_session() -> CorrectionSession:
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0], [11.0, 6.0, 7.0, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0], [13.0, 3.0, 3.0, 2.0]]),
        2: np.array([[14.0, 5.0, 5.0, 3.0]]),
    }
    tracks = [
        np.array([0.0, np.nan, np.nan]),
        np.array([1.0, 0.0, np.nan]),
    ]
    return CorrectionSession(localized_by_frame=localized, tracks=tracks)


def test_empty_history():
    session = make_session()
    assert track_history_entries(session, 0) == []


def test_single_edits_newest_first_with_1based_frames():
    session = make_session()
    session.assign_detection(0, 2, 0)
    session.remove_assignment(0, 2)
    entries = track_history_entries(session, 0)
    assert len(entries) == 2
    # newest first: the removal comes before the assignment
    assert "removed" in entries[0]["summary"]
    assert "added" in entries[1]["summary"].lower()
    assert entries[0]["frames"] == "3"
    assert entries[0]["jump_frame"] == 3
    assert entries[0]["details"] == []


def test_history_only_contains_own_track():
    session = make_session()
    session.assign_detection(0, 2, 0)
    session.set_track_status(1, "verified")
    entries = track_history_entries(session, 0)
    assert len(entries) == 1
    assert all("Status" not in e["summary"] for e in entries)


def test_status_change_description():
    session = make_session()
    session.set_track_status(0, "verified")
    entries = track_history_entries(session, 0)
    assert entries[0]["summary"] == "Status changed: unreviewed → verified"
    assert entries[0]["frames"] == ""
    assert entries[0]["jump_frame"] is None


def test_manual_point_with_assignment_groups_into_one_entry():
    session = make_session()
    session.append_manual_detection(
        2,
        intensity=20.0,
        z=3.0,
        x=4.0,
        track_id=0,
        extra_payload={"params": {"method": "radial"}},
    )
    entries = track_history_entries(session, 0)
    assert len(entries) == 1
    assert "Manual point localized" in entries[0]["summary"]
    assert "added to this track" in entries[0]["summary"]
    assert len(entries[0]["details"]) == 2
    assert "radial" in entries[0]["details"][0]["text"]
    assert entries[0]["details"][0]["frame"] == "3"


def test_discarded_manual_point_has_readable_description():
    session = make_session()
    local_idx = session.append_manual_detection(
        2,
        intensity=20.0,
        z=3.0,
        x=4.0,
    )
    session.discard_manual_detection(2, local_idx)
    assert describe_edit(session.edit_log[-1]) == "Manual point erased"

    session.set_manual_detection_discarded(2, local_idx, False)
    assert describe_edit(session.edit_log[-1]) == "Manual point restored"


def test_merge_summaries_for_both_tracks():
    session = make_session()
    session.merge_tracks(0, 1)
    target_entries = track_history_entries(session, 0)
    source_entries = track_history_entries(session, 1)
    # target got frame 1 (frame 0 overlapped and was dropped)
    assert target_entries[0]["summary"] == (
        "Merge: received 1 point(s) from track 1"
    )
    assert source_entries[0]["summary"] == (
        "Merge: gave 2 point(s) to track 0"
    )
    assert source_entries[0]["frames"] == "1–2"
    assert source_entries[0]["jump_frame"] == 1


def test_steal_shows_up_in_victim_history():
    session = make_session()
    session.assign_detection(0, 1, 0, steal=True)
    victim_entries = track_history_entries(session, 1)
    assert victim_entries[0]["summary"] == "Point stolen by track 0"


def test_undo_entries_are_grouped_and_labeled():
    session = make_session()
    manager = UndoManager(session)
    session.edit_listener = manager.record
    session.assign_detection(0, 2, 0)
    manager.undo()
    entries = track_history_entries(session, 0)
    assert len(entries) == 2
    assert entries[0]["summary"].startswith("Undo")
    assert len(entries[0]["details"]) == 2  # inverse remove + status restore


def test_describe_edit_replaced_detection():
    session = make_session()
    session.assign_detection(1, 1, 1, steal=False)  # replaces detection 0
    edit = session.edit_log[-1]
    assert describe_edit(edit) == "Point replaced (detection #0 → #1)"


def test_structural_edit_summaries_cover_initialize_split_and_delete():
    session = make_session()
    session.assign_detection(0, 2, 0)
    split = session.split_track(0, split_after_frame=0)

    parent_entries = track_history_entries(session, 0)
    first_entries = track_history_entries(session, split["first_child_id"])
    second_entries = track_history_entries(session, split["second_child_id"])

    assert parent_entries[0]["summary"].startswith("Split: became tracks")
    assert first_entries[0]["summary"] == "Split: received 1 point(s) from track 0"
    assert second_entries[0]["summary"] == "Split: received 1 point(s) from track 0"

    session.delete_track(1)
    deleted_entries = track_history_entries(session, 1)
    assert deleted_entries[0]["summary"] == (
        "Track deleted: 2 point(s) became unassigned"
    )

    new_id = session.initialize_track()
    assert track_history_entries(session, new_id)[0]["summary"] == (
        "New track initialized"
    )
