from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ulm_track_correction_gui.core.correction_session import CorrectionSession


def main() -> int:
    localized = {
        0: np.array([[10.0, 2.5, 4.5, 1.0]]),
        1: np.array([[12.0, 8.0, 9.0, 2.0]]),
    }
    tracks = [np.array([0.0, np.nan])]
    session = CorrectionSession(localized_by_frame=localized, tracks=tracks)
    session.assign_detection(track_id=0, frame_idx=1, local_idx=0)

    assert session.n_frames == 2
    assert session.n_tracks == 1
    assert session.tracks[0][1] == 0
    print("Smoke check passed: PALA localization and track structures are usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
