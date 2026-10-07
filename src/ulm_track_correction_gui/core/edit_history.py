"""Human-readable, per-track view of the session edit log (pure, no Qt).

The edit log is the faithful audit trail (including undo/redo, which are
logged as inverse operations). This module turns the raw entries for one
track into display-ready history entries: compound operations sharing a
``group_id`` collapse into one summarized entry with expandable steps.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .correction_session import CorrectionEdit, CorrectionSession

_MERGE_FROM = re.compile(r"Merged from track (\d+)")
_MERGE_INTO = re.compile(r"Merged? into track (\d+)")
_STOLEN_BY = re.compile(r"Reassigned to track (\d+)")
_SPLIT_FROM = re.compile(r"Split from track (\d+)")
_SPLIT_INTO = re.compile(r"Split into tracks (\d+) and (\d+)")
_DELETED_TRACK = re.compile(r"Deleted track (\d+)")


def format_edit_time(iso_timestamp: str) -> str:
    """UTC ISO timestamp -> short local-time display string."""

    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime("%m-%d %H:%M:%S")


def describe_edit(edit: CorrectionEdit) -> str:
    """One-line human description of a single logged edit."""

    action = edit.action
    if action == "assign_detection":
        text = f"Point added (detection #{edit.after_local_idx})"
        if edit.before_local_idx is not None:
            text = (
                f"Point replaced (detection #{edit.before_local_idx} → "
                f"#{edit.after_local_idx})"
            )
    elif action == "remove_assignment":
        text = f"Point removed (was detection #{edit.before_local_idx})"
    elif action == "append_manual_detection":
        payload = edit.payload
        method = (payload.get("params") or {}).get("method")
        z = payload.get("z")
        x = payload.get("x")
        where = ""
        if z is not None and x is not None:
            where = f" at z={float(z):.2f}, x={float(x):.2f}"
        text = f"Manual point localized{where}"
        if method:
            text += f" ({method})"
    elif action == "remove_manual_detection":
        text = "Manual point removed"
    elif action == "set_manual_detection_discarded":
        text = (
            "Manual point erased"
            if edit.payload.get("after")
            else "Manual point restored"
        )
    elif action == "initialize_track":
        text = "Track initialized"
    elif action == "remove_initialized_track":
        text = "Initialized track removed"
    elif action == "set_track_active":
        text = "Track activated" if edit.payload.get("after") else "Track retired"
    elif action == "set_track_status":
        before = edit.payload.get("before", "?")
        after = edit.payload.get("after", "?")
        text = f"Status changed: {before} → {after}"
    else:
        text = action
    note = edit.note.strip()
    if note in ("undo", "redo"):
        text += f" ({note})"
    elif note:
        text += f" — {note}"
    return text


def _frames_display(edits: list[CorrectionEdit]) -> tuple[str, int | None]:
    """(display string, 1-based jump frame) for a group of edits."""

    frames = sorted(
        {edit.frame_idx + 1 for edit in edits if edit.frame_idx is not None}
    )
    if not frames:
        return "", None
    if len(frames) == 1:
        return str(frames[0]), frames[0]
    return f"{frames[0]}–{frames[-1]}", frames[0]


def _summarize_group(edits: list[CorrectionEdit]) -> str:
    """Plain-language summary for one operation (one or more grouped edits).

    Undo/redo inverse edits carry an "undo"/"redo" note, except status
    restores (``set_track_status`` cannot take a note without overwriting the
    track's user note), hence the ``note in (..., "")`` checks.
    """

    notes = [edit.note for edit in edits]
    actions = [edit.action for edit in edits]

    if any(n == "undo" for n in notes) and all(n in ("undo", "") for n in notes):
        return f"Undo: reverted a previous operation ({len(edits)} step(s))"
    if any(n == "redo" for n in notes) and all(n in ("redo", "") for n in notes):
        return f"Redo: re-applied an operation ({len(edits)} step(s))"

    for note in notes:
        match = _MERGE_FROM.search(note)
        if match:
            n_in = sum(1 for a in actions if a == "assign_detection")
            return f"Merge: received {n_in} point(s) from track {match.group(1)}"
    for note in notes:
        match = _MERGE_INTO.search(note)
        if match:
            n_out = sum(1 for a in actions if a == "remove_assignment")
            return f"Merge: gave {n_out} point(s) to track {match.group(1)}"
    for note in notes:
        match = _STOLEN_BY.search(note)
        if match:
            return f"Point stolen by track {match.group(1)}"
    for note in notes:
        match = _SPLIT_FROM.search(note)
        if match:
            n_in = sum(1 for action in actions if action == "assign_detection")
            return f"Split: received {n_in} point(s) from track {match.group(1)}"
    for note in notes:
        match = _SPLIT_INTO.search(note)
        if match:
            n_out = sum(1 for action in actions if action == "remove_assignment")
            return (
                f"Split: became tracks {match.group(1)} and {match.group(2)} "
                f"({n_out} point(s))"
            )
    for note in notes:
        match = _DELETED_TRACK.search(note)
        if match:
            n_out = sum(1 for action in actions if action == "remove_assignment")
            return f"Track deleted: {n_out} point(s) became unassigned"

    if "append_manual_detection" in actions and "assign_detection" in actions:
        manual = next(e for e in edits if e.action == "append_manual_detection")
        return describe_edit(manual).split(" — ")[0] + " and added to this track"
    if any(a == "set_track_status" for a in actions):
        status_edit = next(e for e in edits if e.action == "set_track_status")
        if "Auto-flagged" in status_edit.note:
            return describe_edit(status_edit)
    if "initialize_track" in actions:
        return "New track initialized"
    if len(edits) == 1:
        return describe_edit(edits[0])
    return f"{len(edits)} linked edits"


def track_history_entries(
    session: CorrectionSession, track_id: int
) -> list[dict[str, Any]]:
    """Display-ready history for one track, newest operation first.

    Each entry: ``time`` / ``frames`` (display strings), ``summary``,
    ``jump_frame`` (1-based or None) and ``details`` — per-edit sub-rows,
    non-empty only for compound operations.
    """

    edits = session.edits_for_track_history(track_id)

    groups: list[list[CorrectionEdit]] = []
    for edit in edits:
        if (
            groups
            and edit.group_id is not None
            and groups[-1][-1].group_id == edit.group_id
        ):
            groups[-1].append(edit)
        else:
            groups.append([edit])

    entries: list[dict[str, Any]] = []
    for group in groups:
        frames, jump = _frames_display(group)
        entry: dict[str, Any] = {
            "time": format_edit_time(group[0].created_at),
            "frames": frames,
            "jump_frame": jump,
            "details": [],
        }
        entry["summary"] = _summarize_group(group)
        if len(group) > 1:
            for edit in group:
                frame = "" if edit.frame_idx is None else str(edit.frame_idx + 1)
                entry["details"].append(
                    {
                        "time": format_edit_time(edit.created_at),
                        "frame": frame,
                        "text": describe_edit(edit),
                        "jump_frame": None if not frame else int(frame),
                    }
                )
        entries.append(entry)
    entries.reverse()
    return entries
