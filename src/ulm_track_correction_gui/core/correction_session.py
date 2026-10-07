"""Track-correction session state built around PALA-compatible arrays."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .legacy_adapters import (
    infer_n_frames,
    track_paths_from_tracks,
    track_points_from_tracks,
)
from .pala_contracts import (
    LocalizedByFrame,
    PalaPointRef,
    TrackList,
    append_localization_row,
    ensure_mat_tracking,
    ensure_track_vector,
    get_localization_row,
    iter_track_point_refs,
    normalize_localized_by_frame,
    set_track_assignment,
    tracks_to_matrix,
)


TRACK_STATUS_UNREVIEWED = "unreviewed"
TRACK_STATUS_EDITED = "edited"
TRACK_STATUS_VERIFIED = "verified"
TRACK_STATUS_FLAGGED = "flagged"

VALID_TRACK_STATUSES = {
    TRACK_STATUS_UNREVIEWED,
    TRACK_STATUS_EDITED,
    TRACK_STATUS_VERIFIED,
    TRACK_STATUS_FLAGGED,
}


def new_group_id() -> str:
    return uuid.uuid4().hex[:12]


class DetectionAssignmentConflict(ValueError):
    """Raised when assigning a detection that already belongs to another track.

    One detection belongs to at most one track. Callers must either cancel or retry with
    ``steal=True``, which removes the detection from its current owner first.
    """

    def __init__(
        self,
        frame_idx: int,
        local_idx: int,
        owner_track_id: int,
        requested_track_id: int,
    ) -> None:
        self.frame_idx = frame_idx
        self.local_idx = local_idx
        self.owner_track_id = owner_track_id
        self.requested_track_id = requested_track_id
        super().__init__(
            f"Detection (frame {frame_idx}, local_idx {local_idx}) already "
            f"belongs to track {owner_track_id}; refusing to assign it to "
            f"track {requested_track_id} without steal=True."
        )


@dataclass
class CorrectionEdit:
    """One atomic correction operation.

    The edit log is the replay/audit record (autosave recovery replays it), so
    every entry must be self-contained: ``payload`` carries whatever the action
    needs to be re-applied (e.g. z/x/intensity for manual detections), and
    ``group_id`` ties compound operations together so undo/replay treat them
    as one unit.
    """

    action: str
    track_id: int | None
    frame_idx: int | None
    before_local_idx: int | None
    after_local_idx: int | None
    note: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "before_local_idx": self.before_local_idx,
            "after_local_idx": self.after_local_idx,
            "note": self.note,
            "payload": self.payload,
            "group_id": self.group_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectionEdit":
        return cls(
            action=str(data["action"]),
            track_id=None if data.get("track_id") is None else int(data["track_id"]),
            frame_idx=None if data.get("frame_idx") is None else int(data["frame_idx"]),
            before_local_idx=(
                None
                if data.get("before_local_idx") is None
                else int(data["before_local_idx"])
            ),
            after_local_idx=(
                None
                if data.get("after_local_idx") is None
                else int(data["after_local_idx"])
            ),
            note=str(data.get("note", "")),
            payload=dict(data.get("payload") or {}),
            group_id=data.get("group_id"),
            created_at=str(
                data.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
        )


@dataclass
class CorrectionSession:
    """In-memory state for one correction job."""

    localized_by_frame: LocalizedByFrame
    tracks: TrackList
    image_stack: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    track_status: dict[int, str] = field(default_factory=dict)
    track_notes: dict[int, str] = field(default_factory=dict)
    edit_log: list[CorrectionEdit] = field(default_factory=list)
    audit_segments: list[dict[str, Any]] = field(default_factory=list)
    manual_detections: set[tuple[int, int]] = field(default_factory=set)
    discarded_manual_detections: set[tuple[int, int]] = field(default_factory=set)
    retired_track_ids: set[int] | None = None
    source_track_ids: dict[int, int] | None = None
    track_lineage: dict[int, dict[str, Any]] = field(default_factory=dict)
    edit_listener: Callable[[CorrectionEdit], None] | None = field(
        default=None, repr=False, compare=False
    )
    assignment_index: dict[tuple[int, int], int] = field(
        init=False, default_factory=dict, repr=False
    )
    assignment_conflicts: list[dict[str, Any]] = field(
        init=False, default_factory=list, repr=False
    )

    def __post_init__(self) -> None:
        self.localized_by_frame = normalize_localized_by_frame(self.localized_by_frame)
        n_frames = self.n_frames
        self.tracks = [
            ensure_track_vector(track, n_frames).copy()
            for track in self.tracks
        ]
        self.audit_segments = [dict(segment) for segment in self.audit_segments]
        self.track_status = {
            int(track_id): str(status)
            for track_id, status in self.track_status.items()
        }
        self.track_notes = {
            int(track_id): str(note) for track_id, note in self.track_notes.items()
        }
        for track_id in range(len(self.tracks)):
            self.track_status.setdefault(track_id, TRACK_STATUS_UNREVIEWED)
        invalid_statuses = sorted(
            (int(track_id), status)
            for track_id, status in self.track_status.items()
            if int(track_id) < 0
            or int(track_id) >= len(self.tracks)
            or status not in VALID_TRACK_STATUSES
        )
        if invalid_statuses:
            raise ValueError(f"Invalid track status entries: {invalid_statuses}")
        invalid_note_ids = sorted(
            int(track_id)
            for track_id in self.track_notes
            if int(track_id) < 0 or int(track_id) >= len(self.tracks)
        )
        if invalid_note_ids:
            raise ValueError(f"track_notes contains invalid IDs: {invalid_note_ids}")
        if self.retired_track_ids is None:
            self.retired_track_ids = {
                track_id
                for track_id, track in enumerate(self.tracks)
                if not np.any(~np.isnan(track))
            }
        else:
            self.retired_track_ids = {int(track_id) for track_id in self.retired_track_ids}
            invalid = sorted(
                track_id
                for track_id in self.retired_track_ids
                if track_id < 0 or track_id >= len(self.tracks)
            )
            if invalid:
                raise ValueError(f"retired_track_ids contains invalid IDs: {invalid}")
            nonempty = sorted(
                track_id
                for track_id in self.retired_track_ids
                if np.any(~np.isnan(self.tracks[track_id]))
            )
            if nonempty:
                raise ValueError(
                    "Retired tracks must be empty; non-empty retired IDs: "
                    f"{nonempty}"
                )

        if self.source_track_ids is None:
            remap = self.metadata.get("track_id_remap")
            if isinstance(remap, dict):
                current_ids = [int(current_id) for current_id in remap.values()]
                if len(set(current_ids)) != len(current_ids):
                    raise ValueError("metadata track_id_remap must be one-to-one.")
                source_track_ids = {
                    int(current_id): int(source_id)
                    for source_id, current_id in remap.items()
                    if 0 <= int(current_id) < len(self.tracks)
                }
            else:
                source_track_ids = {
                    track_id: track_id for track_id in range(len(self.tracks))
                }
            self.source_track_ids = source_track_ids
        else:
            self.source_track_ids = {
                int(track_id): int(source_id)
                for track_id, source_id in self.source_track_ids.items()
            }
            invalid = sorted(
                track_id
                for track_id in self.source_track_ids
                if track_id < 0 or track_id >= len(self.tracks)
            )
            if invalid:
                raise ValueError(f"source_track_ids contains invalid IDs: {invalid}")
        if len(set(self.source_track_ids.values())) != len(self.source_track_ids):
            raise ValueError("source_track_ids must map current tracks one-to-one.")

        metadata_lineage = self.metadata.get("track_lineage")
        if not self.track_lineage and isinstance(metadata_lineage, dict):
            self.track_lineage = {
                int(track_id): dict(lineage)
                for track_id, lineage in metadata_lineage.items()
                if 0 <= int(track_id) < len(self.tracks) and isinstance(lineage, dict)
            }
        else:
            self.track_lineage = {
                int(track_id): dict(lineage)
                for track_id, lineage in self.track_lineage.items()
            }
        for track_id, source_id in self.source_track_ids.items():
            self.track_lineage.setdefault(
                track_id,
                {"kind": "source", "source_track_id": int(source_id)},
            )
        self.manual_detections = {
            (int(frame_idx), int(local_idx))
            for frame_idx, local_idx in self.manual_detections
        }
        self.discarded_manual_detections = {
            (int(frame_idx), int(local_idx))
            for frame_idx, local_idx in self.discarded_manual_detections
        }
        self._validate_track_references()
        self._validate_manual_detection_references()
        self._rebuild_assignment_index()
        self._validate_discarded_manual_detection_references()

    def _validate_track_references(self) -> None:
        """Reject invalid local indices before any derived view can index rows."""

        for track_id, track in enumerate(self.tracks):
            for frame_idx in np.flatnonzero(~np.isnan(track)):
                local_idx = int(track[frame_idx])
                rows = ensure_mat_tracking(self.localized_by_frame.get(int(frame_idx), []))
                if local_idx >= rows.shape[0]:
                    raise ValueError(
                        f"Track {track_id} references local_idx={local_idx} at frame "
                        f"{int(frame_idx)}, but that frame has only {rows.shape[0]} "
                        "localization row(s)."
                    )
                if (int(frame_idx), local_idx) in self.discarded_manual_detections:
                    raise ValueError(
                        f"Track {track_id} references discarded manual detection "
                        f"(frame_idx={int(frame_idx)}, local_idx={local_idx})."
                    )

    def _validate_manual_detection_references(self) -> None:
        for frame_idx, local_idx in sorted(self.manual_detections):
            if frame_idx < 0 or frame_idx >= self.n_frames:
                raise ValueError(
                    f"manual_detections references invalid frame_idx={frame_idx}."
                )
            rows = ensure_mat_tracking(self.localized_by_frame.get(frame_idx, []))
            if local_idx < 0 or local_idx >= rows.shape[0]:
                raise ValueError(
                    "manual_detections references invalid "
                    f"(frame_idx={frame_idx}, local_idx={local_idx})."
                )

    def _validate_discarded_manual_detection_references(self) -> None:
        invalid = sorted(
            ref
            for ref in self.discarded_manual_detections
            if ref not in self.manual_detections
        )
        if invalid:
            raise ValueError(
                "discarded_manual_detections must reference manual_detections; "
                f"invalid entries: {invalid}."
            )
        assigned = sorted(
            ref
            for ref in self.discarded_manual_detections
            if ref in self.assignment_index
        )
        if assigned:
            raise ValueError(
                "Discarded manual detections must be unassigned; assigned entries: "
                f"{assigned}."
            )

    def _rebuild_assignment_index(self) -> None:
        """Build the (frame_idx, local_idx) -> track_id reverse index.

        Detections referenced by more than one track violate the uniqueness
        invariant; the first referencing track keeps the assignment in the
        index and all involved tracks are flagged for review.
        """

        self.assignment_index = {}
        self.assignment_conflicts = []
        for track_id, track in enumerate(self.tracks):
            for frame_idx in np.flatnonzero(~np.isnan(track)):
                key = (int(frame_idx), int(track[frame_idx]))
                owner = self.assignment_index.get(key)
                if owner is None:
                    self.assignment_index[key] = track_id
                    continue
                self.assignment_conflicts.append(
                    {
                        "frame_idx": key[0],
                        "local_idx": key[1],
                        "owner_track_id": owner,
                        "track_id": track_id,
                    }
                )
                for tid in (owner, track_id):
                    self.track_status[tid] = TRACK_STATUS_FLAGGED
                    if not self.track_notes.get(tid):
                        self.track_notes[tid] = (
                            "Import conflict: detection shared with another track."
                        )

    @property
    def n_frames(self) -> int:
        return infer_n_frames(
            localized_by_frame=self.localized_by_frame,
            tracks=self.tracks,
            stack=self.image_stack,
        )

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def active_track_ids(self) -> list[int]:
        return [
            track_id
            for track_id in range(len(self.tracks))
            if track_id not in self.retired_track_ids
        ]

    @property
    def active_nonempty_track_ids(self) -> list[int]:
        return [
            track_id
            for track_id in self.active_track_ids
            if np.any(~np.isnan(self.tracks[track_id]))
        ]

    @property
    def active_empty_track_ids(self) -> list[int]:
        return [
            track_id
            for track_id in self.active_track_ids
            if not np.any(~np.isnan(self.tracks[track_id]))
        ]

    def _log(self, edit: CorrectionEdit) -> None:
        self.edit_log.append(edit)
        if self.edit_listener is not None:
            self.edit_listener(edit)

    def state_fingerprint(self) -> str:
        """Stable SHA-256 of correction-relevant baseline state.

        The image stack and filesystem path are intentionally excluded: edit-log
        replay mutates localization/track/review state, not display pixels.
        """

        digest = hashlib.sha256()

        def update_array(label: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            digest.update(label.encode("utf-8"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))

        digest.update(f"n_frames:{self.n_frames}\n".encode("ascii"))
        for frame_idx in sorted(self.localized_by_frame):
            update_array(
                f"localizations:{frame_idx}",
                ensure_mat_tracking(self.localized_by_frame[frame_idx]),
            )
        for track_id, track in enumerate(self.tracks):
            update_array(f"track:{track_id}", track)
        structured = {
            "track_status": self.track_status,
            "track_notes": self.track_notes,
            "manual_detections": sorted(self.manual_detections),
            "discarded_manual_detections": sorted(
                self.discarded_manual_detections
            ),
            "retired_track_ids": sorted(self.retired_track_ids),
            "source_track_ids": self.source_track_ids,
            "track_lineage": self.track_lineage,
        }
        digest.update(
            json.dumps(
                structured,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def integrity_issues(self, *, require_all_verified: bool = False) -> list[str]:
        """Return human-readable violations without mutating the session."""

        issues: list[str] = []
        try:
            self._validate_track_references()
            self._validate_manual_detection_references()
            self._validate_discarded_manual_detection_references()
        except (IndexError, ValueError) as exc:
            issues.append(str(exc))

        owners: dict[tuple[int, int], int] = {}
        for track_id, track in enumerate(self.tracks):
            for frame_idx in np.flatnonzero(~np.isnan(track)):
                key = (int(frame_idx), int(track[frame_idx]))
                previous = owners.get(key)
                if previous is not None:
                    issues.append(
                        f"Detection {key} is assigned to both track {previous} "
                        f"and track {track_id}."
                    )
                else:
                    owners[key] = track_id

        invalid_statuses = sorted(
            (track_id, status)
            for track_id, status in self.track_status.items()
            if track_id < 0
            or track_id >= self.n_tracks
            or status not in VALID_TRACK_STATUSES
        )
        if invalid_statuses:
            issues.append(f"Invalid track status entries: {invalid_statuses}.")

        nonempty_retired = sorted(
            track_id
            for track_id in self.retired_track_ids
            if np.any(~np.isnan(self.tracks[track_id]))
        )
        if nonempty_retired:
            issues.append(f"Retired tracks are not empty: {nonempty_retired}.")

        if require_all_verified:
            empty = self.active_empty_track_ids
            if empty:
                issues.append(f"Active empty track IDs must be resolved: {empty}.")
            unverified = [
                track_id
                for track_id in self.active_nonempty_track_ids
                if self.track_status.get(track_id) != TRACK_STATUS_VERIFIED
            ]
            if unverified:
                issues.append(
                    "All active non-empty tracks must be verified; unresolved IDs: "
                    f"{unverified}."
                )
        return issues

    def validate_integrity(self, *, require_all_verified: bool = False) -> None:
        issues = self.integrity_issues(require_all_verified=require_all_verified)
        if issues:
            raise ValueError("Session integrity validation failed:\n- " + "\n- ".join(issues))

    def edits_for_track_history(self, track_id: int) -> list[CorrectionEdit]:
        """Return archived plus current edits mapped into the current ID space."""

        self.validate_track_id(track_id)
        edits: list[CorrectionEdit] = []
        for segment_index, segment in enumerate(self.audit_segments):
            mapping = {
                int(old): int(current)
                for old, current in dict(
                    segment.get("session_track_id_to_export_id") or {}
                ).items()
            }
            for raw_edit in segment.get("edits", []):
                edit = CorrectionEdit.from_dict(raw_edit)
                if edit.track_id is None or mapping.get(edit.track_id) != track_id:
                    continue
                edit.track_id = track_id
                if edit.group_id is not None:
                    edit.group_id = f"archive-{segment_index}-{edit.group_id}"
                edits.append(edit)
        edits.extend(edit for edit in self.edit_log if edit.track_id == track_id)
        return edits

    def validate_track_id(self, track_id: int) -> None:
        if track_id < 0 or track_id >= len(self.tracks):
            raise IndexError(f"track_id {track_id} is outside this session.")

    def validate_active_track_id(self, track_id: int) -> None:
        self.validate_track_id(track_id)
        if track_id in self.retired_track_ids:
            raise ValueError(f"track_id {track_id} is retired in this session.")

    def is_track_active(self, track_id: int) -> bool:
        self.validate_track_id(track_id)
        return track_id not in self.retired_track_ids

    def source_track_id_for_track(self, track_id: int) -> int | None:
        self.validate_track_id(track_id)
        return self.source_track_ids.get(track_id)

    def current_to_source_track_ids(self) -> dict[int, int]:
        return {
            track_id: source_id
            for track_id, source_id in self.source_track_ids.items()
            if track_id not in self.retired_track_ids
        }

    def validate_detection(
        self,
        frame_idx: int,
        local_idx: int,
        *,
        allow_discarded: bool = False,
    ) -> None:
        rows = ensure_mat_tracking(self.localized_by_frame.get(frame_idx, []))
        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} is outside this session.")
        if local_idx < 0 or local_idx >= rows.shape[0]:
            raise IndexError(
                f"local_idx {local_idx} is outside frame {frame_idx}."
            )
        if (
            not allow_discarded
            and (int(frame_idx), int(local_idx))
            in self.discarded_manual_detections
        ):
            raise ValueError(
                "This manual detection was discarded and cannot be assigned. "
                "Undo the erase operation before using it again."
            )

    def point_refs_for_track(self, track_id: int) -> list[PalaPointRef]:
        self.validate_track_id(track_id)
        return iter_track_point_refs(self.tracks[track_id])

    def localization_row_for_ref(self, point_ref: PalaPointRef) -> np.ndarray:
        return get_localization_row(self.localized_by_frame, point_ref)

    def local_idx_for_track_frame(self, track_id: int, frame_idx: int) -> int | None:
        self.validate_track_id(track_id)
        value = float(self.tracks[track_id][frame_idx])
        return None if np.isnan(value) else int(value)

    def track_id_for_detection(self, frame_idx: int, local_idx: int) -> int | None:
        """Return the track owning a detection, or ``None`` if unassigned."""

        return self.assignment_index.get((int(frame_idx), int(local_idx)))

    def is_manual_detection(self, frame_idx: int, local_idx: int) -> bool:
        """True when this detection was added by a human, not the original run.

        Human-added rows change the localization result; this marker keeps
        them distinguishable in the GUI and in exports.
        """

        return (int(frame_idx), int(local_idx)) in self.manual_detections

    def is_discarded_manual_detection(self, frame_idx: int, local_idx: int) -> bool:
        """True when an unassigned Add Point row is hidden and omitted on export."""

        return (
            int(frame_idx),
            int(local_idx),
        ) in self.discarded_manual_detections

    def set_manual_detection_discarded(
        self,
        frame_idx: int,
        local_idx: int,
        discarded: bool,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Discard or restore one unassigned row created through Add Point.

        The localization table remains append-only so every existing
        ``local_idx`` stays stable. Discarded rows are tombstones: GUI picking
        and rendering ignore them, and MAT export removes them while remapping
        retained frame-local indices.
        """

        self.validate_detection(frame_idx, local_idx, allow_discarded=True)
        key = (int(frame_idx), int(local_idx))
        if key not in self.manual_detections:
            raise ValueError(
                "Only detections created through Add Point can be erased."
            )
        owner = self.assignment_index.get(key)
        if owner is not None:
            raise ValueError(
                f"Detection is assigned to track {owner}; remove that assignment "
                "before erasing it."
            )

        before = key in self.discarded_manual_detections
        after = bool(discarded)
        if before == after:
            return
        if after:
            self.discarded_manual_detections.add(key)
        else:
            self.discarded_manual_detections.discard(key)

        row = ensure_mat_tracking(self.localized_by_frame[frame_idx])[local_idx]
        self._log(
            CorrectionEdit(
                action="set_manual_detection_discarded",
                track_id=None,
                frame_idx=frame_idx,
                before_local_idx=local_idx,
                after_local_idx=local_idx,
                note=note,
                payload={
                    "before": before,
                    "after": after,
                    "intensity": float(row[0]),
                    "z": float(row[1]),
                    "x": float(row[2]),
                },
                group_id=group_id or new_group_id(),
            )
        )

    def discard_manual_detection(
        self,
        frame_idx: int,
        local_idx: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Erase one unassigned Add Point detection without reindexing rows."""

        self.set_manual_detection_discarded(
            frame_idx,
            local_idx,
            True,
            note=note,
            group_id=group_id,
        )

    def assign_detection(
        self,
        track_id: int,
        frame_idx: int,
        local_idx: int,
        *,
        note: str = "",
        steal: bool = False,
        group_id: str | None = None,
    ) -> None:
        """Assign an existing localization row to a track frame.

        Enforces the one-detection-one-track invariant: if the detection is
        owned by another track, raises ``DetectionAssignmentConflict`` unless
        ``steal=True``, in which case the previous owner loses the frame first
        (both edits share one ``group_id``).
        """

        self.validate_active_track_id(track_id)
        self.validate_detection(frame_idx, local_idx)
        gid = group_id or new_group_id()

        key = (int(frame_idx), int(local_idx))
        owner = self.assignment_index.get(key)
        if owner is not None and owner != track_id:
            if not steal:
                raise DetectionAssignmentConflict(frame_idx, local_idx, owner, track_id)
            self.remove_assignment(
                owner,
                frame_idx,
                note=f"Reassigned to track {track_id}.",
                group_id=gid,
            )

        before = self.local_idx_for_track_frame(track_id, frame_idx)
        if before == int(local_idx):
            return
        status_before = self.track_status.get(track_id, TRACK_STATUS_UNREVIEWED)
        self.tracks[track_id] = set_track_assignment(
            self.tracks[track_id],
            frame_idx,
            local_idx,
        )
        if before is not None:
            self.assignment_index.pop((int(frame_idx), int(before)), None)
        self.assignment_index[key] = track_id
        self.track_status[track_id] = (
            TRACK_STATUS_FLAGGED
            if status_before == TRACK_STATUS_VERIFIED
            else TRACK_STATUS_EDITED
        )
        self._log(
            CorrectionEdit(
                action="assign_detection",
                track_id=track_id,
                frame_idx=frame_idx,
                before_local_idx=before,
                after_local_idx=local_idx,
                note=note,
                payload={"status_before": status_before},
                group_id=gid,
            )
        )

    def remove_assignment(
        self,
        track_id: int,
        frame_idx: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Remove a track assignment for one frame, leaving a PALA ``NaN`` gap."""

        self.validate_active_track_id(track_id)
        before = self.local_idx_for_track_frame(track_id, frame_idx)
        if before is None:
            return
        status_before = self.track_status.get(track_id, TRACK_STATUS_UNREVIEWED)
        self.tracks[track_id] = set_track_assignment(
            self.tracks[track_id],
            frame_idx,
            None,
        )
        if before is not None:
            key = (int(frame_idx), int(before))
            if self.assignment_index.get(key) == track_id:
                self.assignment_index.pop(key, None)
        self.track_status[track_id] = (
            TRACK_STATUS_FLAGGED
            if status_before == TRACK_STATUS_VERIFIED
            else TRACK_STATUS_EDITED
        )
        self._log(
            CorrectionEdit(
                action="remove_assignment",
                track_id=track_id,
                frame_idx=frame_idx,
                before_local_idx=before,
                after_local_idx=None,
                note=note,
                payload={"status_before": status_before},
                group_id=group_id or new_group_id(),
            )
        )

    def append_manual_detection(
        self,
        frame_idx: int,
        intensity: float,
        z: float,
        x: float,
        *,
        track_id: int | None = None,
        note: str = "",
        group_id: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a manually localized PALA row, optionally assigning it.

        The payload records z/x/intensity so the edit log alone is enough to
        replay this operation during autosave recovery. ``extra_payload`` can
        carry provenance (e.g. the localization method/parameters used to
        produce the candidate). The new row is marked in
        ``manual_detections``: it modifies the localization result.
        """

        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} is outside this session.")

        gid = group_id or new_group_id()
        local_idx = append_localization_row(
            self.localized_by_frame,
            frame_idx,
            intensity,
            z,
            x,
        )
        self.manual_detections.add((int(frame_idx), int(local_idx)))
        payload: dict[str, Any] = {
            "intensity": float(intensity),
            "z": float(z),
            "x": float(x),
        }
        if extra_payload:
            payload.update(extra_payload)
        self._log(
            CorrectionEdit(
                action="append_manual_detection",
                track_id=track_id,
                frame_idx=frame_idx,
                before_local_idx=None,
                after_local_idx=local_idx,
                note=note,
                payload=payload,
                group_id=gid,
            )
        )
        if track_id is not None:
            if track_id in self.retired_track_ids:
                self.set_track_active(
                    track_id,
                    True,
                    note="Reactivated empty track for manual detection.",
                    group_id=gid,
                )
            self.assign_detection(
                track_id,
                frame_idx,
                local_idx,
                note="Assign manual detection. " + note if note else "Assign manual detection.",
                group_id=gid,
            )
        return local_idx

    def remove_manual_detection(
        self,
        frame_idx: int,
        local_idx: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Remove a manually appended detection (undo support only).

        The append-only invariant means arbitrary rows can never be deleted,
        because every stored ``local_idx`` would shift. Removing strictly the
        LAST row of a frame is the one safe exception, which is exactly what
        undoing an ``append_manual_detection`` needs (undo is LIFO, so the
        appended row is still last when it is undone).
        """

        rows = ensure_mat_tracking(self.localized_by_frame.get(frame_idx, []))
        local_idx = int(local_idx)
        if local_idx != rows.shape[0] - 1 or local_idx < 0:
            raise ValueError(
                "Only the most recently appended detection of a frame can be "
                "removed (append-only invariant)."
            )
        owner = self.assignment_index.get((int(frame_idx), local_idx))
        if owner is not None:
            raise ValueError(
                f"Detection is assigned to track {owner}; remove that "
                "assignment first."
            )
        if (int(frame_idx), local_idx) in self.discarded_manual_detections:
            raise ValueError(
                "Restore the discarded manual detection before removing its row."
            )
        row = rows[local_idx]
        self.localized_by_frame[frame_idx] = rows[:local_idx].copy()
        self.manual_detections.discard((int(frame_idx), local_idx))
        self._log(
            CorrectionEdit(
                action="remove_manual_detection",
                track_id=None,
                frame_idx=frame_idx,
                before_local_idx=local_idx,
                after_local_idx=None,
                note=note,
                payload={
                    "intensity": float(row[0]),
                    "z": float(row[1]),
                    "x": float(row[2]),
                },
                group_id=group_id or new_group_id(),
            )
        )

    def initialize_track(
        self,
        *,
        track_id: int | None = None,
        status: str = TRACK_STATUS_EDITED,
        track_note: str = "",
        lineage: dict[str, Any] | None = None,
        note: str = "Initialized new track.",
        group_id: str | None = None,
    ) -> int:
        """Append one active empty track slot and return its stable session ID.

        ``track_id`` is accepted only for deterministic redo/replay and must be
        exactly the next list index. New tracks deliberately have no Batch
        source-track mapping; selected-track checkpoint plotting therefore
        reports them as derived tracks instead of showing unrelated data.
        """

        if status not in VALID_TRACK_STATUSES:
            raise ValueError(f"Unknown track status: {status}")
        next_id = len(self.tracks)
        requested_id = next_id if track_id is None else int(track_id)
        if requested_id != next_id:
            raise ValueError(
                f"A new track must use the next stable session ID {next_id}, "
                f"got {requested_id}."
            )

        self.tracks.append(np.full(self.n_frames, np.nan, dtype=float))
        self.retired_track_ids.discard(requested_id)
        self.track_status[requested_id] = status
        if track_note:
            self.track_notes[requested_id] = track_note
        else:
            self.track_notes.pop(requested_id, None)
        lineage_payload = dict(lineage or {"kind": "initialized"})
        self.track_lineage[requested_id] = lineage_payload
        self.source_track_ids.pop(requested_id, None)
        self._log(
            CorrectionEdit(
                action="initialize_track",
                track_id=requested_id,
                frame_idx=None,
                before_local_idx=None,
                after_local_idx=None,
                note=note,
                payload={
                    "status": status,
                    "track_note": track_note,
                    "lineage": lineage_payload,
                },
                group_id=group_id or new_group_id(),
            )
        )
        return requested_id

    def remove_initialized_track(
        self,
        track_id: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Pop an empty, last-created track slot (undo/replay support only)."""

        self.validate_track_id(track_id)
        if track_id != len(self.tracks) - 1:
            raise ValueError(
                "Only the last track slot can be removed while undoing initialization."
            )
        if np.any(~np.isnan(self.tracks[track_id])):
            raise ValueError("An initialized track must be empty before its slot is removed.")

        payload = {
            "status": self.track_status.get(track_id, TRACK_STATUS_EDITED),
            "track_note": self.track_notes.get(track_id, ""),
            "lineage": dict(self.track_lineage.get(track_id, {"kind": "initialized"})),
        }
        self.tracks.pop()
        self.retired_track_ids.discard(track_id)
        self.source_track_ids.pop(track_id, None)
        self.track_lineage.pop(track_id, None)
        self.track_status.pop(track_id, None)
        self.track_notes.pop(track_id, None)
        self._log(
            CorrectionEdit(
                action="remove_initialized_track",
                track_id=track_id,
                frame_idx=None,
                before_local_idx=None,
                after_local_idx=None,
                note=note,
                payload=payload,
                group_id=group_id or new_group_id(),
            )
        )

    def set_track_active(
        self,
        track_id: int,
        active: bool,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> None:
        """Activate or retire an existing stable track slot.

        Retired tracks must be empty. They remain addressable by their session
        ID for undo and edit-log replay, but are hidden from the track list and
        omitted from corrected MAT export.
        """

        self.validate_track_id(track_id)
        active = bool(active)
        before = track_id not in self.retired_track_ids
        if before == active:
            return
        if not active and np.any(~np.isnan(self.tracks[track_id])):
            raise ValueError("A track must be empty before it can be retired.")
        if active:
            self.retired_track_ids.discard(track_id)
        else:
            self.retired_track_ids.add(track_id)
        self._log(
            CorrectionEdit(
                action="set_track_active",
                track_id=track_id,
                frame_idx=None,
                before_local_idx=None,
                after_local_idx=None,
                note=note,
                payload={"before": before, "after": active},
                group_id=group_id or new_group_id(),
            )
        )

    def delete_track(
        self,
        track_id: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> dict[str, int]:
        """Retire a whole track while preserving every localization row."""

        self.validate_active_track_id(track_id)
        gid = group_id or new_group_id()
        frames = [int(frame_idx) for frame_idx in np.flatnonzero(~np.isnan(self.tracks[track_id]))]
        for frame_idx in frames:
            self.remove_assignment(
                track_id,
                frame_idx,
                note=f"Deleted track {track_id}.",
                group_id=gid,
            )
        self.set_track_active(
            track_id,
            False,
            note=f"Deleted track {track_id}." + (f" {note}" if note else ""),
            group_id=gid,
        )
        return {"removed": len(frames)}

    def split_track(
        self,
        parent_id: int,
        split_after_frame: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> dict[str, int]:
        """Replace one parent with two newly-IDed, automatically flagged tracks.

        Points at frames ``<= split_after_frame`` move to the first child and
        later points move to the second. Both sides must contain at least one
        assigned detection. The parent becomes an empty retired shell so its
        stable session ID remains available to undo/replay.
        """

        self.validate_active_track_id(parent_id)
        boundary = int(split_after_frame)
        if boundary < 0 or boundary >= self.n_frames - 1:
            raise ValueError(
                f"split_after_frame must satisfy 0 <= frame < {self.n_frames - 1}."
            )
        frames = [int(frame_idx) for frame_idx in np.flatnonzero(~np.isnan(self.tracks[parent_id]))]
        before_frames = [frame_idx for frame_idx in frames if frame_idx <= boundary]
        after_frames = [frame_idx for frame_idx in frames if frame_idx > boundary]
        if not before_frames or not after_frames:
            raise ValueError(
                "Split must leave at least one assigned detection on each side."
            )

        gid = group_id or new_group_id()
        child_a = self.initialize_track(
            status=TRACK_STATUS_EDITED,
            lineage={
                "kind": "split",
                "parent_session_track_id": parent_id,
                "segment": "before",
                "split_after_frame": boundary,
            },
            note=f"Initialized first child from track {parent_id}.",
            group_id=gid,
        )
        child_b = self.initialize_track(
            status=TRACK_STATUS_EDITED,
            lineage={
                "kind": "split",
                "parent_session_track_id": parent_id,
                "segment": "after",
                "split_after_frame": boundary,
            },
            note=f"Initialized second child from track {parent_id}.",
            group_id=gid,
        )

        for frame_idx in frames:
            local_idx = int(self.tracks[parent_id][frame_idx])
            child_id = child_a if frame_idx <= boundary else child_b
            self.remove_assignment(
                parent_id,
                frame_idx,
                note=f"Split into tracks {child_a} and {child_b}.",
                group_id=gid,
            )
            self.assign_detection(
                child_id,
                frame_idx,
                local_idx,
                note=f"Split from track {parent_id}.",
                group_id=gid,
            )

        self.set_track_active(
            parent_id,
            False,
            note=f"Split into tracks {child_a} and {child_b}.",
            group_id=gid,
        )
        split_note = (
            f"Auto-flagged: split from track {parent_id} after frame {boundary + 1}."
        )
        if note:
            split_note += f" {note}"
        self.set_track_status(
            child_a,
            TRACK_STATUS_FLAGGED,
            note=split_note,
            group_id=gid,
        )
        self.set_track_status(
            child_b,
            TRACK_STATUS_FLAGGED,
            note=split_note,
            group_id=gid,
        )
        return {
            "parent_id": parent_id,
            "first_child_id": child_a,
            "second_child_id": child_b,
            "first_points": len(before_frames),
            "second_points": len(after_frames),
            "split_after_frame": boundary,
        }

    def merge_tracks(
        self,
        target_id: int,
        source_id: int,
        *,
        note: str = "",
        group_id: str | None = None,
    ) -> dict[str, int]:
        """Merge ``source_id``'s assignments into ``target_id``.

        Every source assignment is removed; those at frames where the target
        has no detection are assigned to the target ("moved"), those at frames
        the target already covers are simply dropped from the source
        ("dropped" — the detections stay in ``localized_by_frame``,
        unassigned). The source track ends up empty, but no track ID is ever
        renumbered. Saved corrected MAT files preserve the same stable ID slots
        used by the GUI and undo/replay.

        All resulting edits share one ``group_id``: a single undo restores
        both tracks completely.
        """

        self.validate_active_track_id(target_id)
        self.validate_active_track_id(source_id)
        if target_id == source_id:
            raise ValueError("Cannot merge a track into itself.")

        gid = group_id or new_group_id()
        source = self.tracks[source_id]
        moved = 0
        dropped = 0
        for frame_idx in np.flatnonzero(~np.isnan(source)):
            frame_idx = int(frame_idx)
            local_idx = int(source[frame_idx])
            self.remove_assignment(
                source_id,
                frame_idx,
                note=f"Merge into track {target_id}.",
                group_id=gid,
            )
            if np.isnan(self.tracks[target_id][frame_idx]):
                self.assign_detection(
                    target_id,
                    frame_idx,
                    local_idx,
                    note=f"Merged from track {source_id}.",
                    group_id=gid,
                )
                moved += 1
            else:
                dropped += 1

        merge_note = f"Merged into track {target_id}." + (f" {note}" if note else "")
        self.set_track_status(
            source_id,
            TRACK_STATUS_EDITED,
            note=merge_note,
            group_id=gid,
        )
        self.set_track_active(
            source_id,
            False,
            note=merge_note,
            group_id=gid,
        )
        return {"moved": moved, "dropped": dropped}

    def set_track_status(
        self,
        track_id: int,
        status: str,
        *,
        note: str | None = None,
        audit_note: str | None = None,
        group_id: str | None = None,
    ) -> None:
        self.validate_track_id(track_id)
        if status not in VALID_TRACK_STATUSES:
            raise ValueError(f"Unknown track status: {status}")
        before = self.track_status.get(track_id, TRACK_STATUS_UNREVIEWED)
        note_before = self.track_notes.get(track_id, "")
        self.track_status[track_id] = status
        if note is not None:
            if note:
                self.track_notes[track_id] = note
            else:
                self.track_notes.pop(track_id, None)
        note_after = self.track_notes.get(track_id, "")
        self._log(
            CorrectionEdit(
                action="set_track_status",
                track_id=track_id,
                frame_idx=None,
                before_local_idx=None,
                after_local_idx=None,
                note=(note or "") if audit_note is None else audit_note,
                payload={
                    "before": before,
                    "after": status,
                    "note_before": note_before,
                    "note_after": note_after,
                },
                group_id=group_id or new_group_id(),
            )
        )

    def track_summary(self, track_id: int) -> dict[str, Any]:
        self.validate_track_id(track_id)
        track = self.tracks[track_id]
        mask = ~np.isnan(track)
        n_points = int(mask.sum())
        if n_points:
            frames = np.flatnonzero(mask)
            start_frame = int(frames[0])
            end_frame = int(frames[-1])
            n_gaps = int(end_frame - start_frame + 1 - n_points)
        else:
            start_frame = None
            end_frame = None
            n_gaps = 0
        return {
            "track_id": track_id,
            "active": track_id not in self.retired_track_ids,
            "status": self.track_status.get(track_id, TRACK_STATUS_UNREVIEWED),
            "n_points": n_points,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_gaps": n_gaps,
            "note": self.track_notes.get(track_id, ""),
        }

    def review_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in VALID_TRACK_STATUSES}
        for track_id in self.active_track_ids:
            status = self.track_status.get(track_id, TRACK_STATUS_UNREVIEWED)
            counts[status] = counts.get(status, 0) + 1
        return counts

    def to_track_matrix(self) -> np.ndarray:
        return tracks_to_matrix(self.tracks, self.n_frames)

    def to_track_points(self) -> list[dict]:
        return track_points_from_tracks(self.tracks, self.localized_by_frame)

    def to_track_paths(self) -> list[dict]:
        return track_paths_from_tracks(self.tracks, self.localized_by_frame)

    def edit_log_dicts(self) -> list[dict[str, Any]]:
        return [edit.to_dict() for edit in self.edit_log]
