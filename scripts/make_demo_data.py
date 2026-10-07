"""Generate a synthetic demo dataset in the current Batch MAT format.

Creates a MAT file with the current Batch input keys
(``filtered`` + ``localized_points`` + five-column ``track_lines``). The scene
contains moving Gaussian
"bubbles" with occasional missed detections (track gaps) and a few clutter
detections that belong to no track.

Usage:
    python scripts/make_demo_data.py [--out PATH] [--n-frames 60] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

NZ, NX = 128, 128
PSF_SIGMA = 1.5
DETECTION_PROB = 0.85
CLUTTER_PER_FRAME = 1.5


def simulate(n_frames: int, rng: np.random.Generator):
    n_bubbles = 10
    z0 = rng.uniform(15, NZ - 15, n_bubbles)
    x0 = rng.uniform(15, NX - 15, n_bubbles)
    vz = rng.uniform(-0.8, 0.8, n_bubbles)
    vx = rng.uniform(-0.8, 0.8, n_bubbles)
    amp = rng.uniform(0.6, 1.0, n_bubbles)
    t_start = rng.integers(0, n_frames // 3, n_bubbles)
    t_end = rng.integers(2 * n_frames // 3, n_frames, n_bubbles)

    zz, xx = np.meshgrid(
        np.arange(1, NZ + 1, dtype=float),
        np.arange(1, NX + 1, dtype=float),
        indexing="ij",
    )

    stack = rng.normal(0.0, 0.02, size=(NZ, NX, n_frames)).astype(np.float32)
    localized_rows: list[list[float]] = []
    track_line_rows: list[list[float]] = []
    n_local_in_frame = np.zeros(n_frames, dtype=int)

    for f in range(n_frames):
        for b in range(n_bubbles):
            if not (t_start[b] <= f <= t_end[b]):
                continue
            t = f - t_start[b]
            z = z0[b] + vz[b] * t + 1.5 * np.sin(t / 7.0 + b)
            x = x0[b] + vx[b] * t + 1.5 * np.cos(t / 9.0 + b)
            if not (3 < z < NZ - 2 and 3 < x < NX - 2):
                continue
            stack[:, :, f] += (
                amp[b]
                * np.exp(-((zz - z) ** 2 + (xx - x) ** 2) / (2 * PSF_SIGMA**2))
            ).astype(np.float32)

            if rng.random() < DETECTION_PROB:
                z_det = z + rng.normal(0, 0.1)
                x_det = x + rng.normal(0, 0.1)
                local_idx = n_local_in_frame[f]
                localized_rows.append([f, amp[b], z_det, x_det, f + 1])
                track_line_rows.append([b, f, z_det, x_det, local_idx])
                n_local_in_frame[f] += 1

        for _ in range(rng.poisson(CLUTTER_PER_FRAME)):
            z_c = rng.uniform(3, NZ - 2)
            x_c = rng.uniform(3, NX - 2)
            localized_rows.append([f, rng.uniform(0.1, 0.3), z_c, x_c, f + 1])
            n_local_in_frame[f] += 1

    return (
        stack,
        np.asarray(localized_rows, dtype=float),
        np.asarray(track_line_rows, dtype=float),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo/demo_results.mat"),
    )
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    stack, localized_points, track_lines = simulate(args.n_frames, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(args.out),
        {
            "filtered": stack,
            "localized_points": localized_points,
            "track_lines": track_lines,
        },
        do_compression=True,
    )
    n_tracks = len(np.unique(track_lines[:, 0]))
    print(
        f"Wrote {args.out}\n"
        f"  stack: {stack.shape}, localizations: {localized_points.shape[0]}, "
        f"tracks: {n_tracks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
