# ULM Visualization

A desktop application for viewing ultrasound localization microscopy (ULM)
results and manually reviewing and correcting microbubble tracks.

Overlay detections and trajectories on ultrasound frames, browse tracks,
adjust display layers, correct assignments, add points, merge or split tracks,
and export reviewed results or a movie of the current view. Edits support
undo/redo, per-track histories, and crash recovery.

## Install and launch

Use Python 3.11 or newer in a dedicated environment. From the repository root:

```bash
python -m pip install -e .
python -m ulm_track_correction_gui
```

The installed command `ulm-track-correction-gui` also launches the application.
The Python package retains its name `ulm_track_correction_gui`.

## Try the demo data

After installation, run these commands from the repository root:

```bash
python scripts/make_demo_data.py --out demo/demo_results.mat
python -m ulm_track_correction_gui
```

The demo contains 60 synthetic ultrasound frames with moving bubbles, track
gaps, and unassigned detections; no experimental data is included.

1. **Load:** click **Add dataset**, select `demo/demo_results.mat`, then
   double-click its row in the Datasets list.
2. **Explore:** select a track in the Tracks list. Use **Left/Right** to step
   through frames, **Up/Down** to select tracks, and **Space** to play or pause.
   Scroll over the image to zoom; press **F** to fit the view again.
3. **Try a correction:** press **R** and click a point on the selected track to
   remove its assignment. Press **Esc** to return to Inspect, or use the
   **Undo** button to restore the assignment.
4. **Review and save:** press **V** to verify an accepted track or **X** to flag
   it. Click **Save progress** and choose a dataset status. The corrected file
   is saved as `demo/corrected/demo_results.mat`; only verified track geometry
   is exported. Keep the original demo file for its ultrasound movie.

The generated demo has no accumulation checkpoint, so checkpoint-dependent
layers are unavailable. Movie playback, track overlays, and correction tools
can still be explored.

## Input data

The application reads compatible MATLAB `-v7` MAT files containing:

- `filtered`: an ultrasound image stack shaped `(z, x, frames)`.
- `localized_points`: per-frame localization data.
- `track_lines`: an `N × 5` table with columns
  `[track_id, frame_idx_0based, z_1based, x_1based, local_idx_0based]`.

Basic browsing and correction work without accumulation checkpoints. The
checkpoint-dependent accumulation layers require a compatible companion
`*_accumulation_checkpoint.mat` (schema v2) from the upstream processing pipeline.
MATLAB v7.3/HDF5 input is not supported; resave those files with `-v7`.
Data files are opened through the GUI; a command-line `--open` option is not provided.

Corrected exports reference their source movie instead of copying the image
stack. Keep the source file available when reopening corrected results.

## Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_check.py
python scripts/gui_smoke.py
```

The GUI smoke test uses offscreen rendering and an isolated synthetic-data copy.
It writes artifacts to a temporary directory by default. `DEMO_MAT` optionally
selects a synthetic input, and `SMOKE_OUT_DIR` selects the artifact directory.
Use dedicated demo data for this check.

## Algorithm attribution

The `reference_processing` package contains migrated localization and tracking
implementations used by Add Point and reference tests. Original author credits
and algorithm references remain in their source files.

The Python adaptations
[`localizeRadialSymmetry.py`](src/ulm_track_correction_gui/reference_processing/localization_funcs/localizeRadialSymmetry.py)
and
[`ULM_localization2D_NonNCC.py`](src/ulm_track_correction_gui/reference_processing/localization_funcs/ULM_localization2D_NonNCC.py)
retain their upstream **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 (CC BY-NC-SA 4.0)** notices. See the
[license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Their headers preserve the original authors and publication reference and
identify the Python adaptation. Reuse must follow the attribution,
noncommercial, and share-alike terms, including identifying modifications.
This notice applies to these files; it does not establish a license for the
rest of the repository.

## Disclaimer

This project was developed with assistance from Codex, an AI coding tool, throughout the development process.
