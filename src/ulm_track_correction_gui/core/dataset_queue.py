"""Qt-free dataset queue state for multi-file correction workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_QUEUE_SCHEMA_VERSION = 1

DATASET_STATUS_PENDING = "pending"
DATASET_STATUS_IN_PROGRESS = "in_progress"
DATASET_STATUS_COMPLETED = "completed"
DATASET_STATUS_FLAGGED = "flagged"

VALID_DATASET_STATUSES = {
    DATASET_STATUS_PENDING,
    DATASET_STATUS_IN_PROGRESS,
    DATASET_STATUS_COMPLETED,
    DATASET_STATUS_FLAGGED,
}


def canonical_dataset_path(path: str | Path) -> str:
    """Return one stable absolute key without requiring the file to exist."""

    return str(Path(path).expanduser().resolve(strict=False))


@dataclass
class DatasetQueueEntry:
    """One source MAT and its lightweight, manually controlled review state."""

    source_path: str
    status: str = DATASET_STATUS_PENDING
    verified_tracks: int | None = None
    active_tracks: int | None = None
    last_saved_at: str | None = None

    def __post_init__(self) -> None:
        self.source_path = canonical_dataset_path(self.source_path)
        if self.status not in VALID_DATASET_STATUSES:
            raise ValueError(f"Unsupported dataset status: {self.status!r}.")
        if (self.verified_tracks is None) != (self.active_tracks is None):
            raise ValueError(
                "verified_tracks and active_tracks must either both be known or both be None."
            )
        if self.verified_tracks is not None:
            self.verified_tracks = int(self.verified_tracks)
            self.active_tracks = int(self.active_tracks)
            if not 0 <= self.verified_tracks <= self.active_tracks:
                raise ValueError(
                    "Dataset progress must satisfy 0 <= verified_tracks <= active_tracks."
                )

    @property
    def source(self) -> Path:
        return Path(self.source_path)

    @property
    def corrected_path(self) -> Path:
        return self.source.parent / "corrected" / self.source.name

    def set_progress(self, verified_tracks: int, active_tracks: int) -> None:
        verified = int(verified_tracks)
        active = int(active_tracks)
        if not 0 <= verified <= active:
            raise ValueError(
                "Dataset progress must satisfy 0 <= verified_tracks <= active_tracks."
            )
        self.verified_tracks = verified
        self.active_tracks = active

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "status": self.status,
            "verified_tracks": self.verified_tracks,
            "active_tracks": self.active_tracks,
            "last_saved_at": self.last_saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetQueueEntry:
        if not isinstance(data, dict):
            raise ValueError("Each dataset queue entry must be a JSON object.")
        source_path = data.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("Dataset queue entry is missing source_path.")
        last_saved_at = data.get("last_saved_at")
        if last_saved_at is not None and not isinstance(last_saved_at, str):
            raise ValueError("last_saved_at must be a string or null.")
        return cls(
            source_path=source_path,
            status=str(data.get("status", DATASET_STATUS_PENDING)),
            verified_tracks=data.get("verified_tracks"),
            active_tracks=data.get("active_tracks"),
            last_saved_at=last_saved_at,
        )


@dataclass
class DatasetQueue:
    """Ordered, de-duplicated collection of dataset review entries."""

    entries: list[DatasetQueueEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        unique: list[DatasetQueueEntry] = []
        seen: set[str] = set()
        for entry in self.entries:
            if entry.source_path in seen:
                continue
            seen.add(entry.source_path)
            unique.append(entry)
        self.entries = unique

    def __len__(self) -> int:
        return len(self.entries)

    def index_for_path(self, path: str | Path) -> int | None:
        key = canonical_dataset_path(path)
        for index, entry in enumerate(self.entries):
            if entry.source_path == key:
                return index
        return None

    def entry_for_path(self, path: str | Path) -> DatasetQueueEntry | None:
        index = self.index_for_path(path)
        return None if index is None else self.entries[index]

    def add_paths(self, paths: list[str | Path]) -> tuple[int, int]:
        """Append new paths in order; return ``(added, duplicates)``."""

        known = {entry.source_path for entry in self.entries}
        added = 0
        duplicates = 0
        for path in paths:
            key = canonical_dataset_path(path)
            if key in known:
                duplicates += 1
                continue
            self.entries.append(DatasetQueueEntry(source_path=key))
            known.add(key)
            added += 1
        return added, duplicates

    def remove_at(self, index: int) -> DatasetQueueEntry:
        return self.entries.pop(index)

    def next_index(self, current_index: int) -> int | None:
        """Return the literal next row without filtering or wraparound."""

        next_index = int(current_index) + 1
        return next_index if 0 <= next_index < len(self.entries) else None

    def completed_count(self) -> int:
        return sum(
            entry.status == DATASET_STATUS_COMPLETED for entry in self.entries
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATASET_QUEUE_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetQueue:
        if not isinstance(data, dict):
            raise ValueError("Dataset queue settings must be a JSON object.")
        if data.get("schema_version") != DATASET_QUEUE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported dataset queue settings version; expected schema version 1."
            )
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("Dataset queue settings are missing the entries list.")
        return cls([DatasetQueueEntry.from_dict(item) for item in raw_entries])

    @classmethod
    def from_json(cls, text: str) -> DatasetQueue:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Dataset queue settings contain invalid JSON.") from exc
        return cls.from_dict(data)
