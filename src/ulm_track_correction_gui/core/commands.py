"""Undo/redo manager built on inverse operations of logged edits.

Undo and redo are executed through the normal ``CorrectionSession`` methods,
so they are themselves logged (and therefore autosaved and replayable): the
edit log stays a faithful, append-only history, while the undo stack tracks
which edits are user-reversible. Compound operations (steal-assign, manual
detection with assignment) share a ``group_id`` and are undone atomically.
"""

from __future__ import annotations

from .correction_session import CorrectionEdit, CorrectionSession, new_group_id


class UndoManager:
    """LIFO undo/redo over edit groups.

    ``record`` must be called for every session edit (wire it into the
    session's edit listener). Edits recorded while an undo/redo is executing
    are suppressed from the stacks automatically.
    """

    def __init__(self, session: CorrectionSession) -> None:
        self.session = session
        self.undo_stack: list[list[CorrectionEdit]] = []
        self.redo_stack: list[list[CorrectionEdit]] = []
        self._suspended = False

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def record(self, edit: CorrectionEdit) -> None:
        if self._suspended:
            return
        self.redo_stack.clear()
        if (
            self.undo_stack
            and edit.group_id is not None
            and self.undo_stack[-1][-1].group_id == edit.group_id
        ):
            self.undo_stack[-1].append(edit)
        else:
            self.undo_stack.append([edit])

    def undo(self) -> list[CorrectionEdit] | None:
        if not self.undo_stack:
            return None
        group = self.undo_stack.pop()
        # the inverse edits of one undo form one compound operation themselves
        gid = new_group_id()
        self._suspended = True
        try:
            for edit in reversed(group):
                self._invert(edit, gid)
        finally:
            self._suspended = False
        self.redo_stack.append(group)
        return group

    def redo(self) -> list[CorrectionEdit] | None:
        if not self.redo_stack:
            return None
        group = self.redo_stack.pop()
        gid = new_group_id()
        self._suspended = True
        try:
            for edit in group:
                self._reapply(edit, gid)
        finally:
            self._suspended = False
        self.undo_stack.append(group)
        return group

    # ------------------------------------------------------------ internals

    def _invert(self, edit: CorrectionEdit, gid: str) -> None:
        session = self.session
        action = edit.action
        if action == "assign_detection":
            if edit.before_local_idx is None:
                session.remove_assignment(
                    edit.track_id, edit.frame_idx, note="undo", group_id=gid
                )
            else:
                session.assign_detection(
                    edit.track_id,
                    edit.frame_idx,
                    edit.before_local_idx,
                    steal=True,
                    note="undo",
                    group_id=gid,
                )
            self._restore_status(edit, gid)
        elif action == "remove_assignment":
            if edit.before_local_idx is not None:
                session.assign_detection(
                    edit.track_id,
                    edit.frame_idx,
                    edit.before_local_idx,
                    steal=True,
                    note="undo",
                    group_id=gid,
                )
            self._restore_status(edit, gid)
        elif action == "append_manual_detection":
            session.remove_manual_detection(
                edit.frame_idx,
                edit.after_local_idx,
                note="undo",
                group_id=gid,
            )
        elif action == "remove_manual_detection":
            payload = edit.payload
            session.append_manual_detection(
                edit.frame_idx,
                payload["intensity"],
                payload["z"],
                payload["x"],
                note="undo",
                group_id=gid,
            )
        elif action == "set_manual_detection_discarded":
            session.set_manual_detection_discarded(
                edit.frame_idx,
                edit.before_local_idx,
                bool(edit.payload["before"]),
                note="undo",
                group_id=gid,
            )
        elif action == "initialize_track":
            session.remove_initialized_track(
                edit.track_id,
                note="undo",
                group_id=gid,
            )
        elif action == "remove_initialized_track":
            payload = edit.payload
            session.initialize_track(
                track_id=edit.track_id,
                status=payload["status"],
                track_note=payload.get("track_note", ""),
                lineage=payload.get("lineage"),
                note="undo",
                group_id=gid,
            )
        elif action == "set_track_active":
            session.set_track_active(
                edit.track_id,
                bool(edit.payload["before"]),
                note="undo",
                group_id=gid,
            )
        elif action == "set_track_status":
            session.set_track_status(
                edit.track_id,
                edit.payload["before"],
                note=edit.payload.get("note_before"),
                audit_note="",
                group_id=gid,
            )
        else:
            raise ValueError(f"Cannot invert edit action: {action!r}")

    def _reapply(self, edit: CorrectionEdit, gid: str) -> None:
        session = self.session
        action = edit.action
        if action == "assign_detection":
            session.assign_detection(
                edit.track_id,
                edit.frame_idx,
                edit.after_local_idx,
                steal=True,
                note="redo",
                group_id=gid,
            )
        elif action == "remove_assignment":
            session.remove_assignment(
                edit.track_id, edit.frame_idx, note="redo", group_id=gid
            )
        elif action == "append_manual_detection":
            payload = edit.payload
            session.append_manual_detection(
                edit.frame_idx,
                payload["intensity"],
                payload["z"],
                payload["x"],
                note="redo",
                group_id=gid,
            )
        elif action == "remove_manual_detection":
            session.remove_manual_detection(
                edit.frame_idx,
                edit.before_local_idx,
                note="redo",
                group_id=gid,
            )
        elif action == "set_manual_detection_discarded":
            session.set_manual_detection_discarded(
                edit.frame_idx,
                edit.after_local_idx,
                bool(edit.payload["after"]),
                note="redo",
                group_id=gid,
            )
        elif action == "initialize_track":
            payload = edit.payload
            session.initialize_track(
                track_id=edit.track_id,
                status=payload["status"],
                track_note=payload.get("track_note", ""),
                lineage=payload.get("lineage"),
                note="redo",
                group_id=gid,
            )
        elif action == "remove_initialized_track":
            session.remove_initialized_track(
                edit.track_id,
                note="redo",
                group_id=gid,
            )
        elif action == "set_track_active":
            session.set_track_active(
                edit.track_id,
                bool(edit.payload["after"]),
                note="redo",
                group_id=gid,
            )
        elif action == "set_track_status":
            session.set_track_status(
                edit.track_id,
                edit.payload["after"],
                note=edit.payload.get("note_after"),
                audit_note="",
                group_id=gid,
            )
        else:
            raise ValueError(f"Cannot reapply edit action: {action!r}")

    def _restore_status(self, edit: CorrectionEdit, gid: str) -> None:
        before = edit.payload.get("status_before")
        # skip no-op restores so the edit log/history stays free of
        # "edited -> edited" noise when undoing multi-frame groups
        if before and self.session.track_status.get(edit.track_id) != before:
            self.session.set_track_status(edit.track_id, before, group_id=gid)
