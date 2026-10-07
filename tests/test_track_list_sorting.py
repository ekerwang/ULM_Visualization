import pytest

pytest.importorskip("PySide6")

from ulm_track_correction_gui.gui.main_window_tracklist import (
    SORT_OPTIONS,
    _sort_track_summaries,
)


SUMMARIES = [
    {"track_id": 0, "n_points": 3, "n_gaps": 1, "start_frame": 4},
    {"track_id": 1, "n_points": 1, "n_gaps": 2, "start_frame": 2},
    {"track_id": 2, "n_points": 3, "n_gaps": 0, "start_frame": None},
    {"track_id": 3, "n_points": 2, "n_gaps": 2, "start_frame": 2},
]


def test_sort_options_pair_both_directions_for_every_field():
    assert SORT_OPTIONS == (
        "Track ID ↑",
        "Track ID ↓",
        "Points ↑",
        "Points ↓",
        "Gaps ↑",
        "Gaps ↓",
        "Start frame ↑",
        "Start frame ↓",
    )


@pytest.mark.parametrize(
    ("sort_mode", "expected_track_ids"),
    [
        ("Track ID ↑", [0, 1, 2, 3]),
        ("Track ID ↓", [3, 2, 1, 0]),
        ("Points ↑", [1, 3, 0, 2]),
        ("Points ↓", [0, 2, 3, 1]),
        ("Gaps ↑", [2, 0, 1, 3]),
        ("Gaps ↓", [1, 3, 0, 2]),
        ("Start frame ↑", [1, 3, 0, 2]),
        ("Start frame ↓", [0, 1, 3, 2]),
    ],
)
def test_track_summaries_sort_in_both_directions(sort_mode, expected_track_ids):
    ordered = _sort_track_summaries(SUMMARIES, sort_mode)

    assert [summary["track_id"] for summary in ordered] == expected_track_ids
