import json

import pytest

from ulm_track_correction_gui.core.dataset_queue import (
    DATASET_STATUS_COMPLETED,
    DATASET_STATUS_FLAGGED,
    DatasetQueue,
    DatasetQueueEntry,
)


def test_queue_adds_canonical_paths_in_order_and_deduplicates(tmp_path):
    first = tmp_path / "first_results.mat"
    second = tmp_path / "second_results.mat"

    queue = DatasetQueue()
    added, duplicates = queue.add_paths([first, second, first])

    assert (added, duplicates) == (2, 1)
    assert [entry.source_path for entry in queue.entries] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert queue.next_index(0) == 1
    assert queue.next_index(1) is None


def test_queue_json_round_trip_preserves_status_progress_and_order(tmp_path):
    first = DatasetQueueEntry(
        tmp_path / "first_results.mat",
        status=DATASET_STATUS_COMPLETED,
        verified_tracks=12,
        active_tracks=12,
        last_saved_at="2026-08-03T12:00:00+00:00",
    )
    second = DatasetQueueEntry(
        tmp_path / "second_results.mat",
        status=DATASET_STATUS_FLAGGED,
        verified_tracks=7,
        active_tracks=10,
    )

    restored = DatasetQueue.from_json(DatasetQueue([first, second]).to_json())

    assert restored.to_dict() == DatasetQueue([first, second]).to_dict()
    assert restored.completed_count() == 1
    assert restored.entries[0].corrected_path == (
        tmp_path / "corrected" / "first_results.mat"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps({"schema_version": 2, "entries": []}),
        json.dumps({"schema_version": 1, "entries": "not-a-list"}),
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"source_path": "x.mat", "status": "unknown"}],
            }
        ),
    ],
)
def test_queue_rejects_corrupt_or_unsupported_settings(payload):
    with pytest.raises(ValueError):
        DatasetQueue.from_json(payload)


def test_entry_progress_must_be_complete_and_bounded(tmp_path):
    with pytest.raises(ValueError):
        DatasetQueueEntry(tmp_path / "x.mat", verified_tracks=1)
    with pytest.raises(ValueError):
        DatasetQueueEntry(
            tmp_path / "x.mat",
            verified_tracks=3,
            active_tracks=2,
        )
