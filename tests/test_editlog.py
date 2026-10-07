import json

import numpy as np
import pytest

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.io.editlog import (
    EditLogBaselineMismatch,
    EditLogIntegrityError,
    EditLogLock,
    EditLogWriter,
    editlog_path_for,
    read_edit_log,
    read_edit_log_bundle,
    replay_edits,
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


def test_editlog_write_read_replay_round_trip(tmp_path):
    log_path = editlog_path_for(tmp_path / "input.mat")
    writer = EditLogWriter(log_path)

    session = make_session()
    session.edit_listener = writer.append
    session.assign_detection(0, 1, 0)
    session.append_manual_detection(1, intensity=20.0, z=3.0, x=4.0, track_id=1)
    discarded_idx = session.append_manual_detection(
        0,
        intensity=21.0,
        z=10.0,
        x=11.0,
    )
    session.discard_manual_detection(0, discarded_idx)
    session.assign_detection(1, 0, 0, steal=True)
    session.set_track_status(0, "verified")
    writer.close()

    edits = read_edit_log(log_path)
    assert len(edits) == len(session.edit_log)

    recovered = make_session()
    n_applied = replay_edits(recovered, edits)
    assert n_applied == len(edits)

    np.testing.assert_array_equal(
        np.nan_to_num(recovered.tracks[0], nan=-1),
        np.nan_to_num(session.tracks[0], nan=-1),
    )
    np.testing.assert_array_equal(
        np.nan_to_num(recovered.tracks[1], nan=-1),
        np.nan_to_num(session.tracks[1], nan=-1),
    )
    np.testing.assert_allclose(
        recovered.localized_by_frame[1],
        session.localized_by_frame[1],
    )
    assert recovered.track_status == session.track_status
    assert recovered.assignment_index == session.assignment_index
    assert (
        recovered.discarded_manual_detections
        == session.discarded_manual_detections
    )
    # replay keeps the original audit entries, not regenerated ones
    assert [e.created_at for e in recovered.edit_log] == [
        e["created_at"] for e in edits
    ]


def test_read_edit_log_tolerates_torn_last_line(tmp_path):
    log_path = tmp_path / "input.mat.editlog.jsonl"
    log_path.write_text(
        '{"action": "set_track_status", "track_id": 0, "frame_idx": null, '
        '"before_local_idx": null, "after_local_idx": null, "note": "", '
        '"payload": {"before": "unreviewed", "after": "verified"}, '
        '"group_id": "abc", "created_at": "2026-07-06T00:00:00+00:00"}\n'
        '{"action": "assign_det',
        encoding="utf-8",
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edits = read_edit_log(log_path)
    assert len(edits) == 1
    assert edits[0]["action"] == "set_track_status"


def test_structural_edits_replay_to_same_tracks_and_lifecycle():
    session = make_session()
    session.assign_detection(0, 1, 0)
    split = session.split_track(0, split_after_frame=0)
    session.delete_track(1)
    session.initialize_track()

    recovered = make_session()
    edits = [edit.to_dict() for edit in session.edit_log]
    replay_edits(recovered, edits)

    assert recovered.n_tracks == session.n_tracks
    assert recovered.retired_track_ids == session.retired_track_ids
    assert recovered.active_track_ids == session.active_track_ids
    assert recovered.source_track_ids == session.source_track_ids
    assert recovered.track_lineage == session.track_lineage
    assert split["first_child_id"] in recovered.active_track_ids
    assert split["second_child_id"] in recovered.active_track_ids
    for actual, expected in zip(recovered.tracks, session.tracks):
        np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_writer_repairs_torn_tail_before_future_appends(tmp_path):
    path = editlog_path_for(tmp_path / "source.mat")
    legacy = make_session()
    legacy.set_track_status(0, "verified")
    path.write_text(
        __import__("json").dumps(legacy.edit_log[-1].to_dict())
        + "\n"
        + '{"action":"assign_det',
        encoding="utf-8",
    )

    baseline = make_session()
    writer = EditLogWriter(
        path,
        baseline_fingerprint=baseline.state_fingerprint(),
        source_path=tmp_path / "source.mat",
    )
    baseline.set_track_status(1, "flagged")
    writer.append(baseline.edit_log[-1])
    writer.close()

    assert len(read_edit_log(path)) == 2
    assert read_edit_log_bundle(path).baseline_fingerprint == make_session().state_fingerprint()
    assert list(tmp_path.glob("*.corrupt-tail-*"))


def test_writer_rejects_different_baseline_fingerprint(tmp_path):
    path = editlog_path_for(tmp_path / "source.mat")
    session = make_session()
    writer = EditLogWriter(path, baseline_fingerprint=session.state_fingerprint())
    writer.close()

    with pytest.raises(EditLogBaselineMismatch):
        EditLogWriter(path, baseline_fingerprint="different")


def test_writer_upgrades_unbound_versioned_header(tmp_path):
    source = tmp_path / "source.mat"
    path = editlog_path_for(source)
    writer = EditLogWriter(path)
    writer.close()
    assert read_edit_log_bundle(path).baseline_fingerprint is None

    fingerprint = make_session().state_fingerprint()
    writer = EditLogWriter(
        path,
        baseline_fingerprint=fingerprint,
        source_path=source,
    )
    writer.close()

    bundle = read_edit_log_bundle(path)
    assert bundle.baseline_fingerprint == fingerprint
    assert bundle.source_path == str(source)


def test_hash_chain_rejects_modified_edit_record(tmp_path):
    path = editlog_path_for(tmp_path / "source.mat")
    session = make_session()
    writer = EditLogWriter(path, baseline_fingerprint=session.state_fingerprint())
    session.set_track_status(0, "verified")
    writer.append(session.edit_log[-1])
    writer.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["edit"]["payload"]["after"] = "flagged"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(EditLogIntegrityError, match="record hash mismatch"):
        read_edit_log_bundle(path)


def test_recovery_lease_blocks_a_second_writer(tmp_path):
    path = editlog_path_for(tmp_path / "source.mat")
    lock = EditLogLock(path)
    try:
        with pytest.raises(OSError, match="already using edit log"):
            EditLogWriter(path)
    finally:
        lock.close()

    writer = EditLogWriter(path)
    writer.close()


def test_failed_replay_is_transactional():
    session = make_session()
    original_tracks = [track.copy() for track in session.tracks]
    session.set_track_status(0, "verified")
    first = session.edit_log[-1].to_dict()
    fresh = make_session()
    bad = {
        "action": "assign_detection",
        "track_id": 999,
        "frame_idx": 1,
        "before_local_idx": None,
        "after_local_idx": 0,
        "payload": {},
        "note": "",
        "group_id": "bad",
    }

    with pytest.raises(IndexError):
        replay_edits(fresh, [first, bad])

    assert fresh.track_status[0] == "unreviewed"
    assert fresh.edit_log == []
    for actual, expected in zip(fresh.tracks, original_tracks):
        np.testing.assert_allclose(actual, expected, equal_nan=True)
