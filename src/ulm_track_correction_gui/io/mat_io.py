"""MAT import/export helpers for correction sessions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from ulm_track_correction_gui.core.correction_session import CorrectionSession
from ulm_track_correction_gui.core.legacy_adapters import infer_n_frames
from ulm_track_correction_gui.core.pala_contracts import (
    ensure_mat_tracking,
    export_array_to_localized_by_frame,
    localized_by_frame_to_export_array,
    matrix_to_tracks,
    tracks_from_track_lines_array,
    tracks_to_track_lines_array,
)


STACK_KEYS = (
    "filtered",
    "absIQ_SVDfiltered",
    "IQ_Data",
    "IQ",
    "StackData",
    "image_stack",
    "stack",
)

LOCALIZATION_KEYS = (
    "localized_points",
    "MatTracking",
    "localized_mat_tracking",
)

AUDIT_SEGMENT_SCHEMA_VERSION = 1


def _first_array(mat: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in mat and mat[key] is not None:
            value = np.asarray(mat[key])
            if value.size:
                return value
    return None


def _json_from_mat(value: Any, default: Any) -> Any:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    text = str(arr.item() if arr.shape == () else arr.flat[0])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _int_key_dict(value: Any) -> dict[int, Any]:
    data = value if isinstance(value, dict) else {}
    return {int(key): item for key, item in data.items()}


def _track_slot_count_from_metadata(metadata: dict[str, Any]) -> int | None:
    raw_count = metadata.get("track_slot_count")
    if raw_count is None:
        return None
    if isinstance(raw_count, bool):
        raise ValueError("metadata track_slot_count must be a non-negative integer.")
    count = int(raw_count)
    if count < 0 or count != raw_count:
        raise ValueError("metadata track_slot_count must be a non-negative integer.")
    return count


def _audit_segments_from_mat(
    mat: dict[str, Any], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    segments = _json_from_mat(mat.get("audit_log_segments_json"), [])
    if isinstance(segments, list):
        valid_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
        if valid_segments:
            return valid_segments

    legacy_edits = _json_from_mat(mat.get("edit_log_json"), [])
    if not isinstance(legacy_edits, list) or not legacy_edits:
        return []
    return [
        {
            "schema_version": AUDIT_SEGMENT_SCHEMA_VERSION,
            "segment_id": uuid.uuid4().hex,
            "kind": "legacy_embedded_audit",
            "session_track_id_to_export_id": dict(
                metadata.get("session_track_id_to_export_id") or {}
            ),
            "session_local_idx_to_export_idx": dict(
                metadata.get("session_local_idx_to_export_idx") or {}
            ),
            "edits": [dict(edit) for edit in legacy_edits if isinstance(edit, dict)],
        }
    ]


def _audit_fingerprint(segments: list[dict[str, Any]]) -> str:
    json_safe = json.loads(json.dumps(segments, ensure_ascii=False, default=str))
    encoded = json.dumps(
        json_safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_correction_session(path: str | Path) -> CorrectionSession:
    """Load a correction session from a PALA/current-GUI style MAT file."""

    mat_path = Path(path)
    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise ValueError(
            f"{mat_path.name} looks like a MATLAB v7.3 (HDF5) file, which is "
            "not supported yet. Re-save it in MATLAB with save(..., '-v7')."
        ) from exc

    stack = _first_array(mat, STACK_KEYS)

    localized_source = _first_array(mat, LOCALIZATION_KEYS)
    if localized_source is None:
        localized_by_frame = {}
    else:
        localized_by_frame = export_array_to_localized_by_frame(localized_source)

    n_frames = infer_n_frames(localized_by_frame=localized_by_frame, stack=stack)
    if "n_frames" in mat:
        n_frames = max(n_frames, int(np.asarray(mat["n_frames"]).item()))

    metadata = _json_from_mat(mat.get("metadata_json"), {})
    metadata.setdefault("source_path", str(mat_path))

    source_track_ids: dict[int, int] | None = None
    if "track_lines" in mat:
        tracks = tracks_from_track_lines_array(
            mat["track_lines"],
            localized_by_frame,
            n_frames,
            n_track_slots=_track_slot_count_from_metadata(metadata),
        )
        if metadata.get("track_storage_format") != "track_lines_nx5":
            source_track_ids = {
                track_id: track_id
                for track_id, track in enumerate(tracks)
                if np.any(~np.isnan(track))
            }
    elif "corrected_tracks_matrix" in mat:
        tracks = matrix_to_tracks(
            np.asarray(mat["corrected_tracks_matrix"], dtype=float),
            n_frames if n_frames > 0 else None,
        )
    else:
        legacy_keys = (
            "tracks",
            "track_points",
            "corrected_track_points",
            "corrected_track_paths",
        )
        found_legacy = [key for key in legacy_keys if key in mat]
        legacy_detail = (
            f" Found unsupported legacy key(s): {', '.join(found_legacy)}."
            if found_legacy
            else ""
        )
        raise ValueError(
            f"{mat_path.name} has no supported tracking source.{legacy_detail} "
            "Current inputs must contain track_lines shaped (N, 5) with columns "
            "[track_id, frame_idx_0based, z_1based, x_1based, local_idx_0based]. "
            "Legacy Correction GUI files containing corrected_tracks_matrix remain "
            "readable. Rerun legacy Batch data with the current Batch Processing."
        )

    retired_track_ids: set[int] | None = None
    if "retired_track_ids" in metadata:
        raw_retired_track_ids = metadata["retired_track_ids"]
        if not isinstance(raw_retired_track_ids, list) or any(
            isinstance(track_id, bool) or not isinstance(track_id, int)
            for track_id in raw_retired_track_ids
        ):
            raise ValueError("metadata retired_track_ids must be a list of integers.")
        retired_track_ids = set(raw_retired_track_ids)

    manual_detections: set[tuple[int, int]] = set()
    if "manual_detections" in mat:
        manual_arr = np.asarray(mat["manual_detections"], dtype=float)
        if manual_arr.size:
            manual_arr = manual_arr.reshape(-1, 2)
            manual_detections = {
                (int(frame_idx), int(local_idx))
                for frame_idx, local_idx in manual_arr
            }

    session = CorrectionSession(
        localized_by_frame=localized_by_frame,
        tracks=tracks,
        image_stack=stack,
        metadata=metadata,
        track_status=_int_key_dict(_json_from_mat(mat.get("track_status_json"), {})),
        track_notes=_int_key_dict(_json_from_mat(mat.get("track_notes_json"), {})),
        manual_detections=manual_detections,
        audit_segments=_audit_segments_from_mat(mat, metadata),
        retired_track_ids=retired_track_ids,
        source_track_ids=source_track_ids,
    )
    stored_state_fingerprint = metadata.get("export_state_fingerprint")
    if (
        stored_state_fingerprint is not None
        and stored_state_fingerprint != session.state_fingerprint()
    ):
        raise ValueError(
            f"{mat_path.name} failed its correction-state fingerprint check. "
            "The corrected MAT may be incomplete, corrupted, or modified."
        )
    stored_audit_fingerprint = metadata.get("audit_log_fingerprint")
    if (
        stored_audit_fingerprint is not None
        and stored_audit_fingerprint != _audit_fingerprint(session.audit_segments)
    ):
        raise ValueError(
            f"{mat_path.name} failed its audit-log fingerprint check. "
            "Do not use it as a reproducible correction result."
        )
    if session.assignment_conflicts:
        session.metadata.setdefault("import_warnings", []).append(
            f"{len(session.assignment_conflicts)} detection(s) are referenced "
            "by more than one track; the involved tracks were flagged."
        )
    return session


def _localization_state_for_export(
    session: CorrectionSession,
    track_ids: list[int],
) -> tuple[
    dict[int, np.ndarray],
    list[np.ndarray],
    set[tuple[int, int]],
    dict[str, dict[str, int]],
]:
    """Omit discarded manual rows and remap retained frame-local indices."""

    if not session.discarded_manual_detections:
        return (
            session.localized_by_frame,
            [session.tracks[track_id] for track_id in track_ids],
            set(session.manual_detections),
            {},
        )

    remap_by_frame: dict[int, dict[int, int]] = {}
    localized_by_frame: dict[int, np.ndarray] = {}
    for frame_idx, source_rows in session.localized_by_frame.items():
        rows = ensure_mat_tracking(source_rows)
        discarded = {
            local_idx
            for discarded_frame, local_idx in session.discarded_manual_detections
            if discarded_frame == frame_idx
        }
        retained = [
            local_idx for local_idx in range(rows.shape[0]) if local_idx not in discarded
        ]
        remap_by_frame[frame_idx] = {
            old_local_idx: new_local_idx
            for new_local_idx, old_local_idx in enumerate(retained)
        }
        if retained:
            localized_by_frame[frame_idx] = rows[retained].copy()

    tracks: list[np.ndarray] = []
    for track_id in track_ids:
        track = np.asarray(session.tracks[track_id], dtype=float).copy()
        for frame_idx in np.flatnonzero(~np.isnan(track)):
            old_local_idx = int(track[frame_idx])
            try:
                track[frame_idx] = remap_by_frame[int(frame_idx)][old_local_idx]
            except KeyError as exc:
                raise ValueError(
                    "Cannot export a track that references a discarded manual "
                    f"detection: track {track_id}, frame {int(frame_idx)}, "
                    f"local_idx {old_local_idx}."
                ) from exc
        tracks.append(track)

    manual_detections = {
        (frame_idx, remap_by_frame[frame_idx][local_idx])
        for frame_idx, local_idx in session.manual_detections
        if (frame_idx, local_idx) not in session.discarded_manual_detections
    }
    session_local_idx_to_export_idx = {
        str(frame_idx): {
            str(old_local_idx): new_local_idx
            for old_local_idx, new_local_idx in remap.items()
        }
        for frame_idx, remap in remap_by_frame.items()
        if any(
            discarded_frame == frame_idx
            for discarded_frame, _local_idx in session.discarded_manual_detections
        )
    }
    return (
        localized_by_frame,
        tracks,
        manual_detections,
        session_local_idx_to_export_idx,
    )


def _prepare_for_export(session: CorrectionSession) -> CorrectionSession:
    """Prepare a verified-only export without changing any session track ID.

    Structural operations retain stable session slots for undo/replay, and the
    corrected MAT retains those same slots in correction metadata. Only active,
    non-empty tracks whose status is ``verified`` produce rows in ``track_lines``;
    every exported Track ID therefore remains identical to the ID shown in the
    GUI, even when verified IDs are sparse.

    ``track_id_remap`` keeps its established checkpoint meaning of original Batch
    source ID -> exported corrected ID. Because export IDs now equal session IDs,
    both that mapping and ``session_track_id_to_export_id`` preserve IDs instead
    of compacting them.
    """

    track_ids = list(range(len(session.tracks)))
    verified_track_ids = [
        track_id
        for track_id in session.active_nonempty_track_ids
        if session.track_status.get(track_id) == "verified"
    ]
    verified_track_id_set = set(verified_track_ids)
    metadata = dict(session.metadata)
    export_remap = {
        str(session.source_track_ids[track_id]): track_id
        for track_id in verified_track_ids
        if track_id in session.source_track_ids
    }
    metadata["track_id_remap"] = export_remap
    metadata["session_track_id_to_export_id"] = {
        str(track_id): track_id for track_id in track_ids
    }
    exported_lineage = {
        str(track_id): dict(session.track_lineage.get(track_id, {}))
        for track_id in verified_track_ids
    }
    metadata["track_lineage"] = exported_lineage
    metadata["retired_track_ids"] = sorted(set(track_ids) - verified_track_id_set)
    metadata["active_empty_track_ids"] = []
    metadata["exported_verified_track_ids"] = verified_track_ids
    metadata["track_id_export_policy"] = "verified_only_preserve_session_ids"
    metadata["track_storage_format"] = "track_lines_nx5"
    metadata["track_slot_count"] = len(track_ids)
    metadata.pop("n_tracks_before_renumber", None)
    metadata.pop("omitted_active_empty_track_ids", None)
    (
        localized_by_frame,
        tracks,
        manual_detections,
        local_idx_remap,
    ) = _localization_state_for_export(session, track_ids)
    for track_id in set(track_ids) - verified_track_id_set:
        tracks[track_id] = np.full(session.n_frames, np.nan, dtype=float)
    metadata["session_local_idx_to_export_idx"] = local_idx_remap
    return CorrectionSession(
        localized_by_frame=localized_by_frame,
        tracks=tracks,
        image_stack=session.image_stack,
        metadata=metadata,
        track_status={track_id: "verified" for track_id in verified_track_ids},
        track_notes={
            track_id: session.track_notes[track_id]
            for track_id in verified_track_ids
            if track_id in session.track_notes
        },
        manual_detections=manual_detections,
        retired_track_ids=set(track_ids) - verified_track_id_set,
        source_track_ids={
            track_id: session.source_track_ids[track_id]
            for track_id in verified_track_ids
            if track_id in session.source_track_ids
        },
        track_lineage={
            track_id: dict(session.track_lineage.get(track_id, {}))
            for track_id in verified_track_ids
        },
    )


def _audit_segments_for_export(
    original: CorrectionSession,
    exported: CorrectionSession,
) -> list[dict[str, Any]]:
    """Compose archived ID mappings and append this session's audit segment."""

    def local_idx_mapping(value: Any) -> dict[int, dict[int, int]]:
        return {
            int(frame_idx): {
                int(old_local_idx): int(new_local_idx)
                for old_local_idx, new_local_idx in dict(frame_mapping).items()
            }
            for frame_idx, frame_mapping in dict(value or {}).items()
        }

    def compose_local_idx_mappings(
        previous: dict[int, dict[int, int]],
        current: dict[int, dict[int, int]],
    ) -> dict[str, dict[str, int]]:
        composed: dict[int, dict[int, int]] = {
            frame_idx: dict(frame_mapping)
            for frame_idx, frame_mapping in previous.items()
        }
        for frame_idx, current_frame_mapping in current.items():
            previous_frame_mapping = previous.get(frame_idx)
            if previous_frame_mapping is None:
                composed[frame_idx] = dict(current_frame_mapping)
                continue
            composed[frame_idx] = {
                old_local_idx: current_frame_mapping[current_local_idx]
                for old_local_idx, current_local_idx in previous_frame_mapping.items()
                if current_local_idx in current_frame_mapping
            }
        return {
            str(frame_idx): {
                str(old_local_idx): new_local_idx
                for old_local_idx, new_local_idx in frame_mapping.items()
            }
            for frame_idx, frame_mapping in composed.items()
        }

    current_to_export = {
        int(old): int(new)
        for old, new in dict(
            exported.metadata.get("session_track_id_to_export_id") or {}
        ).items()
    }
    current_local_idx_to_export = local_idx_mapping(
        exported.metadata.get("session_local_idx_to_export_idx")
    )
    segments: list[dict[str, Any]] = []
    for archived in original.audit_segments:
        segment = deepcopy(archived)
        previous_mapping = {
            int(old): int(current)
            for old, current in dict(
                segment.get("session_track_id_to_export_id") or {}
            ).items()
        }
        segment["session_track_id_to_export_id"] = {
            str(old): current_to_export[current]
            for old, current in previous_mapping.items()
            if current in current_to_export
        }
        segment["session_local_idx_to_export_idx"] = compose_local_idx_mappings(
            local_idx_mapping(segment.get("session_local_idx_to_export_idx")),
            current_local_idx_to_export,
        )
        segments.append(segment)

    if original.edit_log:
        segments.append(
            {
                "schema_version": AUDIT_SEGMENT_SCHEMA_VERSION,
                "segment_id": uuid.uuid4().hex,
                "kind": "correction_session",
                "baseline_fingerprint": original.metadata.get(
                    "active_editlog_baseline_fingerprint",
                    original.metadata.get("source_correction_fingerprint"),
                ),
                "session_track_id_to_export_id": {
                    str(old): new for old, new in current_to_export.items()
                },
                "session_local_idx_to_export_idx": compose_local_idx_mappings(
                    {},
                    current_local_idx_to_export,
                ),
                "source_track_ids": {
                    str(track_id): source_id
                    for track_id, source_id in original.source_track_ids.items()
                },
                "edits": original.edit_log_dicts(),
            }
        )
    return segments


def _flatten_audit_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(edit)
        for segment in segments
        for edit in segment.get("edits", [])
        if isinstance(edit, dict)
    ]


def _reject_source_overwrite(session: CorrectionSession, path: str | Path) -> None:
    source_path = session.metadata.get("source_path")
    if not source_path:
        return
    destination = Path(path).expanduser().resolve(strict=False)
    source = Path(str(source_path)).expanduser().resolve(strict=False)
    if destination == source:
        raise ValueError(
            "Refusing to overwrite the source MAT. Save corrected data to a "
            "different path or use the corrected/ dataset workflow."
        )


def save_correction_session(session: CorrectionSession, path: str | Path) -> None:
    """Save corrected data without changing the PALA localization/track contract.

    The image stack is intentionally NOT written: corrected results stay small
    and the bubble movie is referenced through ``metadata["source_path"]``
    instead of being duplicated on every save.
    Only verified tracks are exported. Track IDs remain identical to their
    in-session/GUI IDs in canonical five-column ``track_lines``. Empty, retired,
    and non-verified slots have no rows; correction metadata preserves the total
    slot count for reopening and audit traceability.
    """

    _reject_source_overwrite(session, path)
    session.validate_integrity()
    mat_path = Path(path)
    original = session
    session = _prepare_for_export(session)
    session.validate_integrity()
    audit_segments = _audit_segments_for_export(original, session)
    verification_issues = original.integrity_issues(require_all_verified=True)
    review_counts = {
        status: sum(
            original.track_status.get(track_id) == status
            for track_id in original.active_nonempty_track_ids
        )
        for status in ("verified", "unreviewed", "edited", "flagged")
    }
    session.metadata["review_summary"] = {
        "active_nonempty_tracks": len(original.active_nonempty_track_ids),
        "exported_verified_tracks": len(session.active_nonempty_track_ids),
        "verified_tracks": review_counts.get("verified", 0),
        "unreviewed_tracks": review_counts.get("unreviewed", 0),
        "edited_tracks": review_counts.get("edited", 0),
        "flagged_tracks": review_counts.get("flagged", 0),
    }
    session.metadata["verification_complete"] = not verification_issues
    session.metadata["completion_override"] = bool(
        original.metadata.get("dataset_review_state") == "completed"
        and verification_issues
    )
    session.metadata["export_state_fingerprint"] = session.state_fingerprint()
    session.metadata["audit_log_fingerprint"] = _audit_fingerprint(audit_segments)

    data: dict[str, Any] = {
        "localized_points": localized_by_frame_to_export_array(session.localized_by_frame),
        "track_lines": tracks_to_track_lines_array(
            session.tracks, session.localized_by_frame
        ),
        # Human-added localization rows: [frame_idx_0based, local_idx]. These
        # rows also exist inside localized_points; this array marks which of
        # them changed the original localization result.
        "manual_detections": (
            np.asarray(sorted(session.manual_detections), dtype=float).reshape(-1, 2)
            if session.manual_detections
            else np.empty((0, 2), dtype=float)
        ),
        "n_frames": float(session.n_frames),
        "metadata_json": json.dumps(session.metadata, ensure_ascii=False),
        "track_status_json": json.dumps(session.track_status, ensure_ascii=False),
        "track_notes_json": json.dumps(session.track_notes, ensure_ascii=False),
        # Flat legacy field remains for MATLAB/external readers. The segmented
        # form preserves each session track-ID namespace across re-saves.
        "edit_log_json": json.dumps(
            _flatten_audit_segments(audit_segments), ensure_ascii=False
        ),
        "audit_log_segments_json": json.dumps(audit_segments, ensure_ascii=False),
    }

    savemat(str(mat_path), data, do_compression=True)


def verify_correction_export(session: CorrectionSession, path: str | Path) -> None:
    """Reload a written MAT and prove semantic equality before publication."""

    expected = _prepare_for_export(session)
    expected.validate_integrity()
    actual = load_correction_session(path)
    actual.validate_integrity()

    if expected.n_frames != actual.n_frames or expected.n_tracks != actual.n_tracks:
        raise ValueError(
            "Saved MAT verification failed: frame/track counts changed during round trip."
        )
    if set(expected.localized_by_frame) != set(actual.localized_by_frame):
        raise ValueError("Saved MAT verification failed: localization frames changed.")
    for frame_idx in expected.localized_by_frame:
        if not np.array_equal(
            expected.localized_by_frame[frame_idx],
            actual.localized_by_frame[frame_idx],
            equal_nan=True,
        ):
            raise ValueError(
                f"Saved MAT verification failed: localizations changed at frame {frame_idx}."
            )
    for track_id, (expected_track, actual_track) in enumerate(
        zip(expected.tracks, actual.tracks, strict=True)
    ):
        if not np.array_equal(expected_track, actual_track, equal_nan=True):
            raise ValueError(
                f"Saved MAT verification failed: track {track_id} changed during round trip."
            )
    if expected.track_status != actual.track_status:
        raise ValueError("Saved MAT verification failed: track statuses changed.")
    if expected.track_notes != actual.track_notes:
        raise ValueError("Saved MAT verification failed: track notes changed.")
    if expected.manual_detections != actual.manual_detections:
        raise ValueError("Saved MAT verification failed: manual detections changed.")
    expected_segments = _audit_segments_for_export(session, expected)

    def without_segment_ids(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = deepcopy(segments)
        for segment in normalized:
            segment.pop("segment_id", None)
        return normalized

    if without_segment_ids(expected_segments) != without_segment_ids(
        actual.audit_segments
    ):
        raise ValueError("Saved MAT verification failed: audit history changed.")
    for key in (
        "source_path",
        "track_id_remap",
        "session_track_id_to_export_id",
        "session_local_idx_to_export_idx",
        "track_lineage",
        "retired_track_ids",
        "active_empty_track_ids",
        "exported_verified_track_ids",
        "track_id_export_policy",
        "track_storage_format",
        "track_slot_count",
        "dataset_review_state",
    ):
        if key in expected.metadata and actual.metadata.get(key) != expected.metadata[key]:
            raise ValueError(
                f"Saved MAT verification failed: metadata field {key!r} changed."
            )
    expected_verification_complete = not session.integrity_issues(
        require_all_verified=True
    )
    if actual.metadata.get("verification_complete") is not expected_verification_complete:
        raise ValueError(
            "Saved MAT verification failed: verification classification changed."
        )
    expected_completion_override = bool(
        session.metadata.get("dataset_review_state") == "completed"
        and not expected_verification_complete
    )
    if actual.metadata.get("completion_override") is not expected_completion_override:
        raise ValueError(
            "Saved MAT verification failed: completion override changed."
        )
    stored_fingerprint = actual.metadata.get("export_state_fingerprint")
    if stored_fingerprint != actual.state_fingerprint():
        raise ValueError("Saved MAT verification failed: state fingerprint mismatch.")


def save_correction_session_atomic(
    session: CorrectionSession,
    path: str | Path,
) -> None:
    """Write a corrected MAT completely before replacing its destination.

    The temporary file lives beside the destination so ``os.replace`` stays
    on one filesystem. A failed export removes only the temporary file and
    leaves any previous corrected result untouched.
    """

    _reject_source_overwrite(session, path)
    mat_path = Path(path)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{mat_path.stem}-",
        suffix=".tmp.mat",
        dir=mat_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_correction_session(session, temporary_path)
        with temporary_path.open("rb") as fh:
            os.fsync(fh.fileno())
        verify_correction_export(session, temporary_path)
        os.replace(temporary_path, mat_path)
        directory_fd = os.open(mat_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
