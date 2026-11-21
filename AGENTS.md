# AGENTS.md - Gilly Lab Squid Chromatophores

Guidance for AI coding agents working in this repository.

## What this is

A Panel/HoloViews dashboard for analysing chromatophores in cephalopod skin video with CHROMAS (run, export, explore). A separate **Signal** add-on measures the brightness of boxes on the whole animal (flashing and flickering) and has its own database; it does not use CHROMAS. The only video shipped for the Signal add-on is a synthetic demo with known truth (plus four short real clips); treat its thresholds as placeholders until they have been checked against hand-labeled real footage.

## Components

1. **chroma_analysis.py** - pure numpy/pandas analysis: `window_features`, `classify`, `pair_phase`, `episodes`, `flash_cycles` (cycle-by-cycle frequency and amplitude), `find_pauses`, `grid_features` (spatial synchrony), `in_phase_fraction`. No video or database code here.
2. **main.py** - orchestrator: reads a video with OpenCV, extracts mean grayscale per ROI and frame, calls the analysis, and writes `processed/gl_chromatophore.duckdb` atomically (per-video DELETE + INSERT in one transaction).
3. **chromatophore_holoviews.py** - read-only dashboard. Reads the database; plays `videos/*.mp4` through `--static-dirs videos=videos`.
4. **roi_tracker.py** - `BoxTracker` (optical-flow ROI tracking) used by `main.extract_signals` for the ROIs named in a video's `[tracking] rois`. Frames it flags as lost make `lost_frac` windows labeled `untracked` in `chroma_analysis.classify`.
5. **chromas_bridge.py / chromas_export.py / chromas_view.py / make_fake_chromas.py** - optional CHROMAS integration. CHROMAS is installed in the sibling project `gillylab_chromas` (Python 3.12, own `.venv`) and is never imported here. The bridge starts `chromas.exe` as a subprocess, starts `chromas_export.py` with the CHROMAS Python (zarr and xarray exist only there), and loads CSV into `processed/gl_chromas.duckdb` (tables `chromas_runs`, `chromas_areas`, `chromas_focus`). `chromas_view.py` is the dashboard's CHROMAS tab. `make_fake_chromas.py` runs in the CHROMAS environment and writes a synthetic dataset.
   **rf_training_view.py** is the Datasets tab's step-by-step form for training a random forest (frames, masks, crops, train, score). **lookup_picker.py** is the dashboard's colour picker for lookup segmentation (Datasets tab): click on a video frame, see the result live, then start CHROMAS with the table as its model file so its own clicking window never opens. It builds the table itself (every Background click is class 0, every Chromatophore click is class 1; CHROMAS counts segmentation > 0 as chromatophore) and does not import or modify CHROMAS.
6. **chromas_activation.py** - analysis of CHROMAS results: relative-area activation events (rate above `onset_z` robust sigmas, hysteresis onset), Delaunay neighbours, a permutation test of neighbour coupling (rate correlation and event coincidence), and waves (groups larger than chance, plane fit of onset time against position). Reads `processed/chromas_export/<dataset>/` (needs `centroids.csv`), writes `processed/chromas_activation/<dataset>/`. `python chromas_activation.py example` must keep passing after changes (synthetic data with known waves and a no-coupling control). A folder `examples/<name>/` may hold a video, its lookup table and an `example.toml` with the scale and region (none ship yet).
7. **make_demo_video.py** - synthetic video with a truth file (`videos/demo_two_squid_truth.json`), shown in the dashboard as "Show known truth".

## Rules

- Settings live in `config.defaults.toml`; a video's `videos/<id>.toml` may override them. Do not hard-code thresholds in the analysis code.
- Each video needs `videos/<id>.toml` with a `[rois]` table (`name = [x, y, width, height]` in pixels). An optional `[grids]` table (`name = [x, y, width, height, rows, columns]`) gives a spatial synchrony index. A `background` ROI in open water is a useful reference for camera or lighting changes.
- Tables of `gl_chromatophore.duckdb`: `videos`, `rois`, `roi_signal`, `roi_windows`, `pair_windows`, `episodes` (also n_cycles, cycle_freq_hz, cycle_freq_sd_hz), `flash_cycles` (with amplitude), `pauses`, `grid_windows`, `build_metadata`, all keyed by `video_id`.
- After changing the analysis, run `python main.py --demo` and compare the detected states, flash frequencies and phases with the known truth in the dashboard (tick **Show known truth**). There is no automated test.
- The video must be H.264 `.mp4` for browsers. `make_demo_video.py` writes H.264 through the ffmpeg bundled with `imageio-ffmpeg`.
- Videos in `videos/` are tracked in git: the synthetic demo and the small real clips (Movies 1 to 4 of Rosen et al. 2015, 35 KB to 950 KB each). Keep clips small, or use Git LFS. Rights to the real clips stay with the authors and publisher; keep the attribution in the README.
- Python 3.13 (`requires-python = ">=3.13,<3.14"`). Dashboard port 5009 (CTD 5006, flow-through 5007, echogram 5008).
- Dependencies are pinned in `pyproject.toml` with `~=` (compatible patch releases). DuckDB is pinned to 2.0: a database file written by 2.0 cannot be opened by 1.x. Do not add chromas, zarr or xarray to `pyproject.toml`: CHROMAS needs OpenCV 4.x and Python 3.12, so it stays in `gillylab_chromas`. CHROMAS results go to their own file, `gl_chromas.duckdb`, because the dashboard keeps `gl_chromatophore.duckdb` open read-only and DuckDB does not allow a second process to write to an open file.
- Be honest in the docs about what the synthetic demo does and does not show.
