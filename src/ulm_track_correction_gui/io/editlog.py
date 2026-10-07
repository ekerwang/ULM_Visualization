"""Incremental edit-log persistence and crash recovery.

Every edit is appended to a sidecar JSONL file next to the source MAT file
(``<source>.mat.editlog.jsonl``) and flushed immediately, so hours of manual
correction survive a crash. On reopening the source file, the log can be
replayed onto a freshly loaded session to reconstruct the corrected state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ulm_track_correction_gui.core.correction_session import (
    CorrectionEdit,
    CorrectionSession,
)

EDITLOG_SUFFIX = ".editlog.jsonl"
EDITLOG_SCHEMA_VERSION = 1

try:  # POSIX production targets; kept optional for importability elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


class EditLogIntegrityError(ValueError):
    """The sidecar is internally inconsistent or has been modified."""


class EditLogBaselineMismatch(EditLogIntegrityError):
    """The sidecar belongs to a different correction baseline."""


@dataclass(frozen=True)
class EditLogBundle:
    edits: list[dict[str, Any]]
    baseline_fingerprint: str | None = None
    source_path: str | None = None
    log_id: str | None = None
    is_legacy: bool = False
    corrupt_tail_offset: int | None = None
    last_sequence: int = 0
    last_hash: str | None = None


class EditLogLock:
    """Exclusive sidecar lease held across recovery decisions and writing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.with_name(f"{self.path.name}.lock").open("a+")
        if fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._fh.close()
                raise OSError(
                    f"Another process is already using edit log {self.path}."
                ) from exc

    def close(self) -> None:
        if self._fh.closed:
            return
        if fcntl is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def editlog_path_for(source_path: str | Path) -> Path:
    return Path(str(source_path) + EDITLOG_SUFFIX)


def _canonical_json(data: dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record_without_hash)).hexdigest()


def _header(
    baseline_fingerprint: str | None,
    source_path: str | None,
    log_id: str,
) -> dict[str, Any]:
    return {
        "record_type": "editlog_header",
        "schema_version": EDITLOG_SCHEMA_VERSION,
        "log_id": log_id,
        "baseline_fingerprint": baseline_fingerprint,
        "source_path": source_path,
        "created_at": datetime.now().astimezone().isoformat(),
    }


def _edit_record(
    edit: dict[str, Any],
    sequence: int,
    previous_hash: str | None,
) -> dict[str, Any]:
    body = {
        "record_type": "edit",
        "schema_version": EDITLOG_SCHEMA_VERSION,
        "sequence": int(sequence),
        "previous_hash": previous_hash,
        "edit": edit,
    }
    return {**body, "record_hash": _record_hash(body)}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_archive_path(path: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    candidate = path.with_name(f"{path.name}.{label}-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{label}-{stamp}-{suffix}")
        suffix += 1
    return candidate


def _write_versioned_log(
    path: Path,
    edits: list[dict[str, Any]],
    *,
    baseline_fingerprint: str | None,
    source_path: str | None,
    log_id: str,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    previous_hash: str | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_header(baseline_fingerprint, source_path, log_id)) + "\n")
            for sequence, edit in enumerate(edits, start=1):
                record = _edit_record(edit, sequence, previous_hash)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                previous_hash = record["record_hash"]
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _repair_corrupt_tail(path: Path, offset: int) -> Path:
    backup = _unique_archive_path(path, "corrupt-tail")
    with path.open("rb") as source:
        source.seek(offset)
        tail = source.read()
    with backup.open("xb") as archived:
        archived.write(tail)
        archived.flush()
        os.fsync(archived.fileno())
    with path.open("r+b") as fh:
        fh.truncate(offset)
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_directory(path.parent)
    return backup


class EditLogWriter:
    """Versioned, hash-chained JSONL writer with flush/fsync and file locking."""

    def __init__(
        self,
        path: str | Path,
        *,
        baseline_fingerprint: str | None = None,
        source_path: str | Path | None = None,
        lock: EditLogLock | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if lock is not None and lock.path.resolve(strict=False) != self.path.resolve(
            strict=False
        ):
            raise ValueError("Edit-log lock belongs to a different sidecar path.")
        self._lock = lock or EditLogLock(self.path)

        try:
            bundle = read_edit_log_bundle(self.path)
            if bundle.corrupt_tail_offset is not None:
                _repair_corrupt_tail(self.path, bundle.corrupt_tail_offset)
                bundle = read_edit_log_bundle(self.path)

            source_text = None if source_path is None else str(Path(source_path))
            if (
                bundle.baseline_fingerprint is not None
                and baseline_fingerprint is not None
                and bundle.baseline_fingerprint != baseline_fingerprint
            ):
                raise EditLogBaselineMismatch(
                    "Edit log baseline fingerprint does not match the loaded session."
                )

            needs_upgrade = (
                bundle.log_id is None
                or bundle.is_legacy
                or (
                    bundle.baseline_fingerprint is None
                    and baseline_fingerprint is not None
                )
                or (bundle.source_path is None and source_text is not None)
            )
            if needs_upgrade:
                log_id = bundle.log_id or hashlib.sha256(os.urandom(32)).hexdigest()[:24]
                _write_versioned_log(
                    self.path,
                    bundle.edits,
                    baseline_fingerprint=baseline_fingerprint,
                    source_path=source_text,
                    log_id=log_id,
                )
                bundle = read_edit_log_bundle(self.path)

            self._sequence = bundle.last_sequence
            self._previous_hash = bundle.last_hash
            self._fh = self.path.open("a", encoding="utf-8")
        except Exception:
            self._lock.close()
            raise

    def append(self, edit: CorrectionEdit) -> None:
        record = _edit_record(
            edit.to_dict(),
            self._sequence + 1,
            self._previous_hash,
        )
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._sequence += 1
        self._previous_hash = str(record["record_hash"])

    def close(self) -> None:
        if hasattr(self, "_fh") and not self._fh.closed:
            self._fh.close()
        if hasattr(self, "_lock"):
            self._lock.close()


def read_edit_log_bundle(path: str | Path) -> EditLogBundle:
    """Read and validate a versioned log, reporting a repairable corrupt tail."""

    log_path = Path(path)
    if not log_path.exists():
        return EditLogBundle(edits=[])
    edits: list[dict[str, Any]] = []
    baseline_fingerprint: str | None = None
    source_path: str | None = None
    log_id: str | None = None
    is_legacy = False
    expected_sequence = 1
    previous_hash: str | None = None
    valid_end = 0
    corrupt_tail_offset: int | None = None

    with log_path.open("rb") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line_start = valid_end
            line = raw_line.strip()
            if not line:
                valid_end = fh.tell()
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                warnings.warn(
                    f"Stopping at corrupt edit-log line {line_no} in {log_path}; "
                    f"keeping the {len(edits)} edit(s) parsed so far.",
                    stacklevel=2,
                )
                corrupt_tail_offset = line_start
                break

            record_type = data.get("record_type") if isinstance(data, dict) else None
            if record_type == "editlog_header":
                if line_no != 1 or log_id is not None:
                    raise EditLogIntegrityError("Edit-log header must be the first record.")
                if int(data.get("schema_version", -1)) != EDITLOG_SCHEMA_VERSION:
                    raise EditLogIntegrityError(
                        f"Unsupported edit-log schema version {data.get('schema_version')!r}."
                    )
                log_id = str(data.get("log_id") or "") or None
                baseline = data.get("baseline_fingerprint")
                baseline_fingerprint = None if baseline is None else str(baseline)
                source = data.get("source_path")
                source_path = None if source is None else str(source)
            elif record_type == "edit":
                if log_id is None:
                    raise EditLogIntegrityError("Versioned edit record has no header.")
                sequence = int(data.get("sequence", -1))
                if sequence != expected_sequence:
                    raise EditLogIntegrityError(
                        f"Edit-log sequence mismatch: expected {expected_sequence}, got {sequence}."
                    )
                if data.get("previous_hash") != previous_hash:
                    raise EditLogIntegrityError(
                        f"Edit-log hash chain breaks at sequence {sequence}."
                    )
                body = {key: value for key, value in data.items() if key != "record_hash"}
                calculated_hash = _record_hash(body)
                if data.get("record_hash") != calculated_hash:
                    raise EditLogIntegrityError(
                        f"Edit-log record hash mismatch at sequence {sequence}."
                    )
                edit = data.get("edit")
                if not isinstance(edit, dict):
                    raise EditLogIntegrityError(
                        f"Edit-log sequence {sequence} has no edit object."
                    )
                edits.append(dict(edit))
                previous_hash = calculated_hash
                expected_sequence += 1
            elif isinstance(data, dict) and "action" in data:
                if log_id is not None:
                    raise EditLogIntegrityError(
                        "Legacy edit records cannot follow a versioned header."
                    )
                is_legacy = True
                edits.append(dict(data))
            else:
                raise EditLogIntegrityError(
                    f"Unknown edit-log record at line {line_no}."
                )
            valid_end = fh.tell()

    return EditLogBundle(
        edits=edits,
        baseline_fingerprint=baseline_fingerprint,
        source_path=source_path,
        log_id=log_id,
        is_legacy=is_legacy,
        corrupt_tail_offset=corrupt_tail_offset,
        last_sequence=expected_sequence - 1,
        last_hash=previous_hash,
    )


def read_edit_log(path: str | Path) -> list[dict[str, Any]]:
    """Compatibility API returning only validated/unwrapped edit dictionaries."""

    return read_edit_log_bundle(path).edits


def apply_edit(session: CorrectionSession, edit: dict[str, Any]) -> None:
    """Re-apply one logged edit to a session.

    ``append_manual_detection`` entries are applied without their optional
    track assignment: when the original call also assigned the new detection,
    that assignment was logged as a separate ``assign_detection`` entry in the
    same group and will be replayed on its own.
    """

    action = edit.get("action")
    note = str(edit.get("note", ""))
    group_id = edit.get("group_id")

    def expect_equal(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            raise ValueError(
                f"Edit log diverges before {action}: expected {label}={expected!r}, "
                f"found {actual!r}."
            )

    if action == "assign_detection":
        track_id = int(edit["track_id"])
        frame_idx = int(edit["frame_idx"])
        before = edit.get("before_local_idx")
        before = None if before is None else int(before)
        expect_equal(
            "before_local_idx",
            session.local_idx_for_track_frame(track_id, frame_idx),
            before,
        )
        owner = session.track_id_for_detection(frame_idx, int(edit["after_local_idx"]))
        if owner is not None and owner != track_id:
            raise ValueError(
                f"Edit log diverges before assign_detection: target detection is "
                f"still owned by track {owner}."
            )
        session.assign_detection(
            track_id,
            frame_idx,
            int(edit["after_local_idx"]),
            note=note,
            steal=False,
            group_id=group_id,
        )
    elif action == "remove_assignment":
        track_id = int(edit["track_id"])
        frame_idx = int(edit["frame_idx"])
        before = edit.get("before_local_idx")
        before = None if before is None else int(before)
        expect_equal(
            "before_local_idx",
            session.local_idx_for_track_frame(track_id, frame_idx),
            before,
        )
        session.remove_assignment(
            track_id,
            frame_idx,
            note=note,
            group_id=group_id,
        )
    elif action == "append_manual_detection":
        payload = edit.get("payload") or {}
        local_idx = session.append_manual_detection(
            int(edit["frame_idx"]),
            float(payload["intensity"]),
            float(payload["z"]),
            float(payload["x"]),
            track_id=None,
            note=note,
            group_id=group_id,
        )
        expected = edit.get("after_local_idx")
        if expected is not None and int(expected) != int(local_idx):
            raise ValueError(
                f"Edit log does not match the loaded data: manual detection "
                f"replayed to local_idx {local_idx}, expected {expected}. "
                f"Was the source file modified after the log was written?"
            )
    elif action == "remove_manual_detection":
        session.remove_manual_detection(
            int(edit["frame_idx"]),
            int(edit["before_local_idx"]),
            note=note,
            group_id=group_id,
        )
    elif action == "set_manual_detection_discarded":
        payload = edit.get("payload") or {}
        frame_idx = int(edit["frame_idx"])
        local_idx = int(edit["after_local_idx"])
        expect_equal(
            "discarded",
            session.is_discarded_manual_detection(frame_idx, local_idx),
            bool(payload["before"]),
        )
        session.set_manual_detection_discarded(
            frame_idx,
            local_idx,
            bool(payload["after"]),
            note=note,
            group_id=group_id,
        )
    elif action == "initialize_track":
        payload = edit.get("payload") or {}
        session.initialize_track(
            track_id=int(edit["track_id"]),
            status=str(payload.get("status", "edited")),
            track_note=str(payload.get("track_note", "")),
            lineage=payload.get("lineage"),
            note=note,
            group_id=group_id,
        )
    elif action == "remove_initialized_track":
        session.remove_initialized_track(
            int(edit["track_id"]),
            note=note,
            group_id=group_id,
        )
    elif action == "set_track_active":
        payload = edit.get("payload") or {}
        track_id = int(edit["track_id"])
        expect_equal("active", session.is_track_active(track_id), bool(payload["before"]))
        session.set_track_active(
            track_id,
            bool(payload["after"]),
            note=note,
            group_id=group_id,
        )
    elif action == "set_track_status":
        payload = edit.get("payload") or {}
        track_id = int(edit["track_id"])
        expect_equal(
            "status",
            session.track_status.get(track_id, "unreviewed"),
            str(payload["before"]),
        )
        if "note_before" in payload:
            expect_equal(
                "track_note",
                session.track_notes.get(track_id, ""),
                str(payload.get("note_before") or ""),
            )
        session.set_track_status(
            track_id,
            str(payload["after"]),
            note=(payload.get("note_after") if "note_after" in payload else note or None),
            audit_note=note,
            group_id=group_id,
        )
    else:
        raise ValueError(f"Unknown edit-log action: {action!r}")


def replay_edits(session: CorrectionSession, edits: list[dict[str, Any]]) -> int:
    """Replay logged edits onto a freshly loaded session.

    On success the session's in-memory edit log is replaced with the original
    logged entries (original timestamps and group ids), not the freshly
    generated ones, so the audit trail stays faithful.
    """

    working = CorrectionSession(
        localized_by_frame={
            frame_idx: rows.copy()
            for frame_idx, rows in session.localized_by_frame.items()
        },
        tracks=[track.copy() for track in session.tracks],
        image_stack=session.image_stack,
        metadata=deepcopy(session.metadata),
        track_status=dict(session.track_status),
        track_notes=dict(session.track_notes),
        edit_log=deepcopy(session.edit_log),
        audit_segments=deepcopy(session.audit_segments),
        manual_detections=set(session.manual_detections),
        discarded_manual_detections=set(session.discarded_manual_detections),
        retired_track_ids=set(session.retired_track_ids),
        source_track_ids=dict(session.source_track_ids),
        track_lineage=deepcopy(session.track_lineage),
    )
    baseline = len(working.edit_log)
    for edit in edits:
        apply_edit(working, edit)

    del working.edit_log[baseline:]
    working.edit_log.extend(CorrectionEdit.from_dict(e) for e in edits)
    lifecycle_ids = {
        int(edit["track_id"])
        for edit in edits
        if edit.get("action") == "set_track_active" and edit.get("track_id") is not None
    }
    legacy_merged_ids = {
        int(edit["track_id"])
        for edit in edits
        if edit.get("track_id") is not None
        and "Merged into track" in str(edit.get("note", ""))
    }
    for track_id in legacy_merged_ids - lifecycle_ids:
        if (
            0 <= track_id < working.n_tracks
            and not any(~np.isnan(working.tracks[track_id]))
        ):
            working.retired_track_ids.add(track_id)

    working.validate_integrity()
    listener = session.edit_listener
    session.localized_by_frame = working.localized_by_frame
    session.tracks = working.tracks
    session.metadata = working.metadata
    session.track_status = working.track_status
    session.track_notes = working.track_notes
    session.edit_log = working.edit_log
    session.audit_segments = working.audit_segments
    session.manual_detections = working.manual_detections
    session.discarded_manual_detections = working.discarded_manual_detections
    session.retired_track_ids = working.retired_track_ids
    session.source_track_ids = working.source_track_ids
    session.track_lineage = working.track_lineage
    session.assignment_index = working.assignment_index
    session.assignment_conflicts = working.assignment_conflicts
    session.edit_listener = listener
    return len(edits)
