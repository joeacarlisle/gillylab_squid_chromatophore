"""Bridge between this project and CHROMAS (runs in its own environment, see gillylab_chromas).

CHROMAS needs a different Python (3.12) and OpenCV than this project, so it is never imported here.  This module
starts it as a separate process and moves its results into DuckDB:

    run     start `chromas run` on a video         ->  <chromas home>/output/<name>.zarr
    export  start chromas_export.py in the CHROMAS Python  ->  processed/chromas_export/<name>/*.csv
    load    read those files into processed/gl_chromas.duckdb

The CHROMAS results go to their own database file, not to gl_chromatophore.duckdb: the dashboard keeps that one
open read-only, and DuckDB does not let another process write to a file that is open.

Command line (run in this project's environment):
    python chromas_bridge.py check                        # are the CHROMAS paths right?
    python chromas_bridge.py run data/clip.mp4 [--method lookup] [--name clip] [--weights file.pth] [--gpu]
    python chromas_bridge.py import output/clip.zarr [--name clip]   # export + load an existing dataset
    python chromas_bridge.py load [name ...] [--method lookup]   # (re)load exported folders; no names = all of them
    python chromas_bridge.py list

Settings: [chromas] in config.defaults.toml (home folder of the CHROMAS project; environment variable
GL_CHROMAS_HOME overrides it).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / 'processed'
EXPORT_DIR = PROCESSED_DIR / 'chromas_export'
CHROMAS_DB_PATH = PROCESSED_DIR / 'gl_chromas.duckdb'
EXPORT_SCRIPT = BASE_DIR / 'chromas_export.py'
METHODS = ['neuralnet', 'lookup', 'randomforest']
# `chromas run` prints a screenful of harmless click warnings (options it declares twice) that would bury real errors.
_NOISE = re.compile(r'UserWarning: The parameter .* is used more than once|^\s+(self\.|parser = |pieces = )|'
                    r'SyntaxWarning|^\s+weights = |'
                    # chromas generate overlay-video also tries to write a registered copy and cannot (see make_video):
                    # a warning for every frame, which does not affect the video this dashboard plays
                    r'cap_ffmpeg\.cpp:\d+ write FFmpeg: Failed to write frame')

# The lookup method labels pixels by colour from a few clicked points.  CHROMAS treats the FIRST clicked colour as
# background and every later one as chromatophore.  Clicking a chromatophore first (or background again later) labels
# almost the whole frame as chromatophore (89-99.7% in the failed runs), and the tracking step then finds none.
# How much of a frame is really chromatophore depends on the skin: about 3% on the Gilly lab's squid skin, about a
# third on the CHROMAS cuttlefish example.  `chromas run` still prints "completed successfully" after that,
# so the run is stopped as soon as the coverage is known to be too high.
LOOKUP_COVERAGE_LIMIT = 75.0
LOOKUP_SANE_LIMIT = 60.0      # after a failed run: above this share the clicked table is the likely cause
_TOO_FEW = re.compile(r'Not enough chromatophores detected \((\d+) < (\d+)\)')

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS chromas_runs (
        dataset_id VARCHAR PRIMARY KEY, video_path VARCHAR, zarr_path VARCHAR, method VARCHAR,
        fps DOUBLE, fps_source VARCHAR, n_chunks INTEGER, n_chromatophores INTEGER, n_frames INTEGER,
        synthetic BOOLEAN, overlay_path VARCHAR, imported_utc VARCHAR, info_json VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS chromas_areas (
        dataset_id VARCHAR, chunk INTEGER, frame INTEGER, orig_frame INTEGER, t_s DOUBLE,
        chromatophore INTEGER, area DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS chromas_focus (
        dataset_id VARCHAR, frame INTEGER, t_s DOUBLE, in_focus BOOLEAN, statistic DOUBLE)""",
]


# ── Settings ─────────────────────────────────────────────────────────────────────
def _is_real_weights(path):
    """True for a weights file that is really there: not missing, and not a Git LFS pointer (a small text file that
    stands in for the real file when the repository was cloned without Git LFS)."""
    try:
        return path.is_file() and path.stat().st_size > 1_000_000
    except OSError:
        return False


def settings():
    """Paths of the CHROMAS project: home folder, its chromas and python executables, data and output folders."""
    cfg = tomllib.loads((BASE_DIR / 'config.defaults.toml').read_text(encoding='utf-8')).get('chromas', {})
    home = Path(os.environ.get('GL_CHROMAS_HOME') or cfg.get('home', '../gillylab_chromas'))
    if not home.is_absolute():
        home = (BASE_DIR / home).resolve()
    scripts = home / '.venv' / ('Scripts' if os.name == 'nt' else 'bin')
    ext = '.exe' if os.name == 'nt' else ''
    # Network weights: [chromas] weights in the config (relative to this project), by default the copy in models/;
    # if that file is missing (e.g. a clone without Git LFS), the copy in the CHROMAS folder.
    weights = Path(cfg.get('weights', 'models/sepia_2classes_fcn_resnet50.pth'))
    if not weights.is_absolute():
        weights = BASE_DIR / weights
    if not _is_real_weights(weights):
        weights = home / 'model_weights' / 'sepia_2classes_fcn_resnet50.pth'
    return {'home': home, 'chromas_exe': scripts / f'chromas{ext}', 'python_exe': scripts / f'python{ext}',
            'data_dir': home / 'data', 'output_dir': home / 'output',
            'weights': weights,
            'method': cfg.get('method', 'neuralnet'), 'focus_threshold': float(cfg.get('focus_threshold', 0.01)),
            'n_workers': int(cfg.get('n_workers', 2))}


def check_install(st=None):
    """List of problems with the CHROMAS setup (empty list = ready)."""
    st = st or settings()
    problems = []
    if not st['home'].is_dir():
        problems.append(f"CHROMAS folder not found: {st['home']} (set [chromas] home in config.defaults.toml)")
    elif not st['python_exe'].exists():
        problems.append(f"No virtual environment at {st['home'] / '.venv'}: run setup.bat in that folder")
    elif not st['chromas_exe'].exists():
        problems.append(f"chromas is not installed in {st['home'] / '.venv'}: run setup.bat in that folder")
    return problems


# ── Run CHROMAS and export ───────────────────────────────────────────────────────
def _kill_tree(proc):
    """Stop a process and the worker processes it started."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
    else:
        proc.kill()


def lookup_run_problem(line, limit=LOOKUP_COVERAGE_LIMIT):
    """A message when this line of `chromas run` output shows the lookup segmentation has failed, else None."""
    m = re.search(r'has ([\d.]+)% segmentation coverage', line)
    if m and float(m.group(1)) > limit:
        return (f'The lookup table labelled {float(m.group(1)):g}% of the frame as chromatophore (a usable table gives '
                f'well under half), so no individual chromatophores can be tracked. CHROMAS treats the FIRST point '
                f'you click as background and every later point as chromatophore. Run it again: click the background '
                f'first, then one or two chromatophores, then press Q. Or use the neuralnet method, which needs no '
                f'clicking. The run was stopped here; the output folder holds a partial dataset that can be ignored.')
    if 'marked as failed because' in line and 'segment_frames' in line:
        return ('CHROMAS ran out of memory while segmenting the video: its Dask workers were killed one after another. '
                'The frames it was working on were not saved, so the next step would find an empty frame. Each Dask worker gets '
                'the computer\'s memory divided by the number of workers. Set n_workers = 1 under [chromas] in '
                'config.defaults.toml (this project) so one worker gets all of it (the default is 2), close other programs and orphaned python.exe '
                'processes (Task Manager), or crop the video in the Video tab. The run was stopped here.')
    if 'low segmentation coverage (less than 5%)' in line:
        return ('CHROMAS registration found the first frame almost empty (under 1% chromatophore; with the sparse-skin '
                'patch it only skips frames below that). Look earlier in the output (the whole run is in '
                'processed/chromas_last_run.log) for segmentation errors such as "exceeded 95% memory budget" or "marked '
                'as failed": a segmentation that did not finish looks like this. If there were none, the colours are the '
                'cause: check that the first click was the background and that chromatophores were clicked in their dark '
                'centres. The run was stopped here.')
    m = _TOO_FEW.search(line)
    if m:
        return (f'Only {m.group(1)} chromatophores were found to track (at least {m.group(2)} are needed). The '
                f'segmentation (lookup table) is probably poor: run it again with different clicks, or use the '
                f'neuralnet method. The run was stopped here.')
    return None


class Stopped(RuntimeError):
    """The person pressed Stop in the dashboard."""


_stop = {'flag': False, 'proc': None}
_stop_lock = threading.Lock()
STOPPED_TEXT = ('stopped by you. What it printed so far is in the output box; a partial result may be left in the '
                'CHROMAS output folder and can be ignored or run again.')


def request_stop():
    """Stop the CHROMAS process that is running now (and the steps still waiting behind it)."""
    with _stop_lock:
        _stop['flag'] = True
        proc = _stop['proc']
    if proc is not None and proc.poll() is None:
        _kill_tree(proc)


def clear_stop():
    """Call when a new job starts, so an earlier Stop does not stop it."""
    with _stop_lock:
        _stop['flag'] = False


def _stream(cmd, cwd, log, stop_on=None, extra_env=None):
    """Run cmd, send each output line to log, raise if it fails.

    ``stop_on(line)`` may return a message: the process is then stopped and that message is raised.
    request_stop() stops it from outside: Stopped is raised.
    """
    with _stop_lock:
        if _stop['flag']:
            raise Stopped(STOPPED_TEXT)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', **(extra_env or {}))
    log('> ' + ' '.join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    with _stop_lock:
        _stop['proc'] = proc
        already = _stop['flag']
    if already:                                           # Stop was pressed while the process was starting
        _kill_tree(proc)
    full = PROCESSED_DIR / 'chromas_last_run.log'      # complete unfiltered output, for troubleshooting
    try:
        with open(full, 'a', encoding='utf-8') as fh:
            fh.write('\n> ' + ' '.join(str(c) for c in cmd) + '\n')
            for line in proc.stdout:
                line = line.rstrip()
                fh.write(line + '\n')
                if line and not _NOISE.search(line):
                    log(line)
                problem = stop_on(line) if stop_on else None
                if problem:
                    fh.write('STOPPED: ' + problem + '\n')
                    _kill_tree(proc)
                    proc.wait()
                    raise RuntimeError(problem)
            code = proc.wait()
            with _stop_lock:
                stopped = _stop['flag']
            if stopped:
                fh.write('STOPPED by the person\n')
    finally:
        with _stop_lock:
            _stop['proc'] = None
    if stopped:
        raise Stopped(STOPPED_TEXT)
    if code != 0:
        raise RuntimeError(f'{Path(str(cmd[0])).name} stopped with exit code {proc.returncode}')


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')


def find_video(video, st=None):
    """The video file for a name typed in the dashboard or on the command line.  A name without a folder is looked
    for in the CHROMAS data folder, then in this project's videos/ and examples/<name>/ folders, then in the folder the
    command was started from; a name without an extension also tries .mp4, .avi, .mov, .mkv and .m4v.  Raises FileNotFoundError naming
    every place it looked and the videos the data folder has."""
    st = st or settings()
    p = Path(str(video).strip().strip('"'))
    names = [p] if p.suffix.lower() in VIDEO_EXTS else [p] + [p.with_name(p.name + e) for e in VIDEO_EXTS]
    examples = sorted(d for d in (BASE_DIR / 'examples').iterdir() if d.is_dir()) if (BASE_DIR / 'examples').is_dir() else []
    folders = [None] if p.is_absolute() else [st['data_dir'], BASE_DIR / 'videos', *examples, Path.cwd()]
    tried = []
    for folder in folders:
        for n in names:
            cand = n if folder is None else folder / n
            tried.append(cand)
            if cand.is_file():
                return cand.resolve()
    have = sorted(q.name for q in st['data_dir'].iterdir() if q.suffix.lower() in VIDEO_EXTS) \
        if st['data_dir'].is_dir() else []
    where = ', '.join(dict.fromkeys(str(c.parent) for c in tried))
    raise FileNotFoundError(f'Video not found: "{video}". Looked in {where}. '
                            + (f'Videos in the CHROMAS data folder: {", ".join(have)}.' if have else
                               f'The CHROMAS data folder ({st["data_dir"]}) has no videos.'))


def saved_lookup_table(dataset_id, st=None):
    """A lookup table already made for this video: examples/<id>/lookup_table.npz (made by hand for the examples),
    else <CHROMAS output>/<id>.lookup_table.npz (kept after an earlier run whose clicks worked), else None."""
    st = st or settings()
    for p in (BASE_DIR / 'examples' / dataset_id / 'lookup_table.npz', st['output_dir'] / f'{dataset_id}.lookup_table.npz'):
        if p.is_file():
            return p
    return None


def lookup_coverage(table, video, frame=None):
    """Percent of the video's first frame (or of the given frame) that a CHROMAS lookup table labels as chromatophore (labels above 0), the
    way CHROMAS applies it (RGB divided by 16 indexes the table), or None if it cannot be computed."""
    try:
        import cv2
        import numpy as np
        with np.load(table, allow_pickle=False) as d:
            w = d['weights']
        cap = cv2.VideoCapture(str(video))
        if frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) >> 4
        return 100.0 * float((w[rgb[..., 0], rgb[..., 1], rgb[..., 2]] > 0).mean())
    except Exception:
        return None


def lookup_table_problem(table, video, frame=None):
    """A message when a clicked lookup table cannot work on this video, else None."""
    cov = lookup_coverage(table, video, frame)
    if cov is None:
        return None
    clicks = 'In the window, click the BACKGROUND first, then one or more CHROMATOPHORES, then press Q.'
    if cov < 0.2:
        return (f'The lookup table from your clicks labels only {cov:.1f}% of the frame as chromatophore, so CHROMAS '
                f'finds none. Usually only background was clicked, or the chromatophore click was on a pale spot. {clicks}')
    if cov > LOOKUP_SANE_LIMIT:
        return (f'The lookup table from your clicks labels {cov:.0f}% of the frame as chromatophore, so the pigment '
                'merges into a few huge patches and CHROMAS finds no separate chromatophores. Usually the second click '
                'landed on background (or a chromatophore was clicked first, which CHROMAS takes as background). '
                + clicks)
    return None


def run_pipeline(video, name=None, method=None, focus_threshold=None, gpu=False, weights=None, log=print,
                 new_lookup=False):
    """Run `chromas run` on a video.  Returns the dataset (.zarr folder), its id and the video path used.

    Lookup method: a table saved for this video (saved_lookup_table) is used without clicking unless new_lookup is
    set; after clicking, the table is kept as <id>.lookup_table.npz for the next run.  When a run with clicked colours
    fails, the table is checked on the first frame and a message says if the clicks were the cause."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    video = find_video(video, st)
    method = method or st['method']
    if name and not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise RuntimeError(f'The dataset name "{name}" may use only letters, digits, "_", "-" and "." (no spaces: they '
                           'break ffmpeg and CHROMAS).')
    weights_file = None
    if method == 'neuralnet':
        # The chromas package ships without trained weights (its default path points to a file that is not installed).
        weights_file = Path(weights) if weights else st['weights']
        if not weights_file.exists():
            raise RuntimeError(f'The neuralnet method needs a weights file, not found: {weights_file}. Copy '
                               'model_weights/sepia_2classes_fcn_resnet50.pth from Chromas2025.zip (Edmond, '
                               'doi 10.17617/3.MJI3HA) to that place, or use the lookup method (interactive).')
    elif method == 'randomforest':
        # A random forest has no supplied model: it is trained on the lab's own painted frames
        # (`chromas train ... -m rf`, or the training form in the Datasets tab) and saved as rf_segmentation_model_wrapper.pkl.
        if not weights:
            raise RuntimeError('The randomforest method needs the model file you trained (rf_segmentation_model_wrapper.pkl); '
                               'there is no supplied one. Give it in the weights field, or train one first '
                               '(the training form in the Datasets tab).')
        weights_file = Path(weights)
        if not weights_file.exists():
            raise RuntimeError(f'Random forest model not found: {weights_file}')
    elif method == 'lookup' and weights:
        weights_file = Path(weights)       # a lookup table saved by an earlier calibration (lookup_table.npz)
        if not weights_file.exists():
            raise RuntimeError(f'Lookup table not found: {weights_file}')
    elif method == 'lookup' and not new_lookup and (saved_lookup_table(name or video.stem, st)
                                                    or saved_lookup_table(video.stem, st)):
        # the table saved for this dataset name, else the one saved for the video (a new name alone does not mean new colours)
        weights_file = saved_lookup_table(name or video.stem, st) or saved_lookup_table(video.stem, st)
        log(f'Using the lookup table saved for this video ({weights_file}), so no clicking is needed. To click new '
            'colours, tick "Click new colours" in the form.')
    elif method == 'lookup':
        log('NOTE: the lookup method without a saved lookup table opens a WINDOW on your screen (look in the taskbar '
            "for 'Figure 1'). Click the BACKGROUND FIRST, then one or two chromatophores, then press Q. More windows "
            'follow; close each one. The run waits until you do. The table is saved as lookup_table.npz next to the '
            'dataset; pass it as the weights file to skip this next time.')
    dataset_id = name or video.stem
    st['output_dir'].mkdir(parents=True, exist_ok=True)
    zarr_path = st['output_dir'] / f'{dataset_id}.zarr'
    if zarr_path.exists():
        log(f'NOTE: a dataset named "{dataset_id}" already exists and will be replaced by this run. Give the run another '
            'dataset name to keep it.')
    # In `chromas run` the flags of the single steps become options that take a value, so a bare
    # `--force-overwrite` swallows the next argument and fails with exit code 2: they need `true`.
    cmd = [st['chromas_exe'], 'run', video, '--output-name', zarr_path, '--force-overwrite', 'true',
           '--method', method,
           '--focus-threshold', focus_threshold if focus_threshold is not None else st['focus_threshold']]
    if weights_file:
        cmd += ['--weights', weights_file]
    if st['n_workers'] > 0:
        # Dask gives each worker (total memory / n_workers). CHROMAS's own default of 4 leaves ~4 GB each, too little to segment
        # 150 frames of 1920x1080 at once (frames fail silently and registration then sees an empty frame).
        cmd += ['--n-workers', st['n_workers']]
    if gpu:
        cmd += ['--gpu', 'true']
    clicked = method == 'lookup' and weights_file is None
    table = st['output_dir'] / 'lookup_table.npz'           # where CHROMAS saves the clicked table
    started = datetime.now().timestamp()
    try:
        _stream(cmd, st['home'], log, stop_on=lookup_run_problem if method == 'lookup' else None)
    except RuntimeError as exc:
        if clicked and table.exists() and table.stat().st_mtime >= started - 1:
            problem = lookup_table_problem(table, video)
            if problem:
                raise RuntimeError(problem) from None
        raise
    if clicked and table.exists() and table.stat().st_mtime >= started - 1:
        keep = st['output_dir'] / f'{dataset_id}.lookup_table.npz'
        import shutil
        shutil.copy2(table, keep)
        log(f'Lookup table kept as {keep.name}: the next run of this video uses it without clicking.')
    # `chromas run` can print an error from one of its steps and still exit normally, so check what it wrote.
    if not zarr_path.exists():
        raise RuntimeError(f'CHROMAS finished but wrote no dataset ({zarr_path.name}). Read its output above for the '
                           'step that failed. Typical causes: no in-focus stretch of at least 100 frames in the video '
                           '(try a lower focus threshold), or video too distant or low in contrast.')
    return zarr_path, dataset_id, video


def export_dataset(zarr_path, dataset_id, log=print):
    """Run chromas_export.py in the CHROMAS Python.  Returns the folder holding the exported files."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    out = EXPORT_DIR / dataset_id
    try:
        _stream([st['python_exe'], EXPORT_SCRIPT, zarr_path, out], st['home'], log)
    except Stopped:
        raise
    except RuntimeError as exc:
        raise RuntimeError(f'Reading the CHROMAS results failed ({exc}). The line above this message says why. If it says '
                           "No chunk has 'areas', CHROMAS stopped before its area step and found no chromatophores "
                           'to measure.') from None
    return out


def export_extras(zarr_path, dataset_id, log=print):
    """Write processed/chromas_export/<dataset_id>/extras/ (quality checks, motion markers, slices, CHROMAS groups)
    without touching the other exported files.  Runs chromas_export.py --extras-only in the CHROMAS Python."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    out = EXPORT_DIR / dataset_id
    _stream([st['python_exe'], EXPORT_SCRIPT, zarr_path, out, '--extras-only'], st['home'], log)
    return out


EXTRA_STEPS = {
    # name: (what it does, chromas command after the executable)
    'slice': ('cut each chromatophore into 36 wedges and follow the area of each (for the flower plots)',
              ['slice', '{zarr}', '-n', '1', '-t', '2']),
    'cluster': ('group chromatophores whose areas change together (CHROMAS\'s own grouping)',
                ['analyse', 'cluster', 'areas', '{zarr}']),
    # The CHROMAS papers' motor-unit analysis.  Run once per chunk that has wedge data: PCA then ICA on the 36 wedge
    # traces of each chromatophore (each independent component is a putative motor neuron), then clustering of the
    # components of all chromatophores (each cluster is a putative motor unit).
    'motor_units': ('find putative motor units from the wedges (PCA, ICA, clustering of components); needs the slice step',
                    [['ica', '{zarr}', '-c', '{chunk}'], ['analyse', 'cluster', 'ics', '{zarr}', '-c', '{chunk}']]),
}


# settings tried in turn for `chromas analyse cluster ics` (the first is CHROMAS's default)
ICS_CLUSTER_SETTINGS = [[], ['--damping', '0.9', '--max-iter', '1000', '--convergence-iter', '50'], ['-m', 'hdbscan']]


def sliced_chunks(zarr_path):
    """Numbers of the chunks of a dataset that have wedge data (the slice step has run on them)."""
    out = []
    for p in Path(zarr_path).glob('chunk_*'):
        n = p.name.split('_')[1]
        if n.isdigit() and (p / 'slice_areas').exists():
            out.append(int(n))
    return sorted(out)


def chunk_summary(zarr_path):
    """[{'chunk', 'frames', 'chromatophores'}] for each chunk of a CHROMAS result, read from the shape of its `areas`
    array (no zarr package needed), and the chunk numbers the slice step can take together: the biggest group of chunks
    with the same number of chromatophores ('' when they all match or nothing could be read)."""
    rows = []
    for p in Path(zarr_path).glob('chunk_*'):
        n = p.name.split('_')[1]
        try:
            shape = json.loads((p / 'areas' / '.zarray').read_text())['shape']
            rows.append({'chunk': int(n), 'frames': int(shape[0]), 'chromatophores': int(shape[1])})
        except (ValueError, KeyError, IndexError, OSError):
            continue
    rows.sort(key=lambda r: r['chunk'])
    groups = {}
    for r in rows:
        groups.setdefault(r['chromatophores'], []).append(r['chunk'])
    suggestion = ''
    if len(groups) > 1:
        best = max(groups.values(), key=len)                 # ties: the group that contains the lowest chunk number
        ties = [g for g in groups.values() if len(g) == len(best)]
        best = min(ties, key=lambda g: g[0])
        suggestion = ','.join(str(c) for c in best)
    return rows, suggestion


def parse_chunk_list(text):
    """'0,2' -> '0,2' (CHROMAS's --chunk-selection); empty -> ''.  Raises ValueError for anything but numbers and commas."""
    t = (text or '').replace(' ', '')
    if t and not all(p.isdigit() for p in t.split(',')):
        raise ValueError('Chunks: type chunk numbers separated by commas, for example 0,2 (or leave it empty for all).')
    return t


def run_extra_step(zarr_path, dataset_id, step, log=print, chunks=''):
    """Run an optional CHROMAS step on a finished dataset, then export the extras again.  `chunks` (slice and grouping steps,
    e.g. '0,2') limits the step to those chunks: `chromas slice --chunk-selection`, or one `chromas analyse cluster areas -c N` per chunk."""
    st = settings()
    chunks = parse_chunk_list(chunks)
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    if step == 'motor_units':
        chunks = sliced_chunks(zarr_path)
        if not chunks:
            raise RuntimeError('This dataset has no wedge data yet. Run the slice step on it first (Datasets tab, '
                               '"Run slice step"), then the motor-unit analysis.')
        commands = [[a.replace('{zarr}', str(zarr_path)).replace('{chunk}', str(c)) for a in cmd]
                    for c in chunks for cmd in EXTRA_STEPS[step][1]]
    elif step == 'cluster':
        # chromas analyse cluster areas takes ONE chunk (-c N) or the whole dataset, which fails when the chunks have
        # different numbers of chromatophores.  So: the chunks typed in the Chunks box, else the biggest group of chunks
        # that match, else (all match) the whole dataset.
        suggested = chunk_summary(zarr_path)[1]
        todo = chunks.split(',') if chunks else (suggested.split(',') if suggested else [])
        base = [a.replace('{zarr}', str(zarr_path)) for a in EXTRA_STEPS[step][1]]
        commands = [base + ['-c', c] for c in todo] if todo else [base]
    else:
        commands = [[a.replace('{zarr}', str(zarr_path)) for a in EXTRA_STEPS[step][1]]]
        if chunks and step == 'slice':
            commands[0] += ['--chunk-selection', chunks]
    # chromas prints "Slicing failed!" but still exits with code 0, so the output is watched
    seen = {'cross': False}

    def stop_on(line):
        if '3-dimensional vectors' in line:
            seen['cross'] = True
        return 'chromas reported a failure' if line.strip().lower().endswith('failed!') else None
    try:
        for args in commands:
            # Agg: these steps can draw figures; without a display backend no window opens and waits
            if step == 'motor_units' and 'ics' in args:
                # affinity propagation can fail to converge on the independent components, and CHROMAS's 10 tries all use
                # the same settings; so retry with a stronger damping and more iterations, then with HDBSCAN
                for i, extra in enumerate(ICS_CLUSTER_SETTINGS):
                    try:
                        _stream([st['chromas_exe'], *args, *extra], st['home'], log, stop_on=stop_on,
                                extra_env={'MPLBACKEND': 'Agg'})
                        break
                    except RuntimeError:
                        if i == len(ICS_CLUSTER_SETTINGS) - 1:
                            raise
                        log(f'Clustering the components did not converge; trying again with: {" ".join(ICS_CLUSTER_SETTINGS[i + 1])}')
                continue
            _stream([st['chromas_exe'], *args], st['home'], log, stop_on=stop_on, extra_env={'MPLBACKEND': 'Agg'})
    except Stopped:
        raise
    except RuntimeError as exc:
        hint = ''
        if seen['cross']:
            hint = (' This is the NumPy 2 problem: run extras\\apply_patches.py in the CHROMAS folder again '
                    '(setup.bat does it), then try again.')
        elif step == 'slice':
            hint = (' If the output above says too few motion markers were found, the skin has too few small, steady '
                    'chromatophores to slice. If it says "conflicting dimension sizes", the chunks have different numbers '
                    'of chromatophore labels: type the chunks to slice in the Chunks box (for example 0,2) and try again.')
        raise RuntimeError(f'chromas {step} failed ({exc}).{hint}') from None
    return export_extras(zarr_path, dataset_id, log)


# ── Random forest training ───────────────────────────────────────────────────────
RF_MODEL_NAME = 'rf_segmentation_model_wrapper.pkl'


def training_runs_dir(st=None):
    return (st or settings())['home'] / 'training_runs'


def train_forest(input_dir, name, log=print):
    """Train a random forest segmentation model with `chromas train -m rf` on a folder with images/ and masks/.

    The model is saved in <CHROMAS folder>/training_runs/<name>/; returns the path of the .pkl file."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    input_dir = Path(input_dir)
    for sub in ('images', 'masks'):
        if not any((input_dir / sub).glob('*.png')):
            raise RuntimeError(f'No images in {input_dir / sub}.')
    out = training_runs_dir(st) / name
    if (out / RF_MODEL_NAME).exists():
        raise RuntimeError(f'A model called "{name}" already exists. Choose another name.')
    _stream([st['chromas_exe'], 'train', input_dir, '-m', 'rf', '-o', out], st['home'], log,
            extra_env={'MPLBACKEND': 'Agg'})
    if not (out / RF_MODEL_NAME).exists():
        raise RuntimeError('chromas finished but did not write the model file; see the output above.')
    return out / RF_MODEL_NAME


def score_models(folder, lookup=None, forests=(), log=print):
    """Score a lookup table and/or random forest models on a folder of painted frames (images/ and masks/)."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    if not lookup and not forests:
        raise RuntimeError('Choose a model or a lookup table to score.')
    cmd = [st['python_exe'], BASE_DIR / 'chromas_score_segmentation.py', folder]
    if lookup:
        cmd += ['--lookup', lookup]
    if forests:
        cmd += ['--forest', *forests]
    _stream(cmd, st['home'], log)


# ── Quality-control videos ───────────────────────────────────────────────────────
VIDEO_KINDS = {
    # name: (what it shows, chromas command after "generate")
    'segmentation': ('Segmentation overlay: what CHROMAS took for pigment, drawn on the footage', 'overlay-video'),
    'territories': ('Territories: each chromatophore\'s territory drawn on the footage', 'cleanqueen-video'),
}
VIDEO_MAX_WIDTH = 960          # pixels; the dashboard plays the file in the page, so it is shrunk
# What each video needs in the dataset's chunk group (a folder in the .zarr); CHROMAS draws both in the registered frame,
# so a dataset whose run stopped before registration cannot make them.
VIDEO_NEEDS = {'segmentation': ('pullbacks', 'registration'), 'territories': ('warped_cleanqueen', 'registration')}


def list_videos(dataset_id):
    """Quality-control videos made for a dataset: processed/chromas_export/<dataset>/extras/video_<kind>_chunk<n>.mp4."""
    d = EXPORT_DIR / dataset_id / 'extras'
    return sorted(d.glob('video_*.mp4')) if d.is_dir() else []


def _chunk_seconds(dataset_id, chunk):
    """Length of a chunk in seconds from the export's info.json, or None."""
    try:
        info = json.loads((EXPORT_DIR / dataset_id / 'info.json').read_text(encoding='utf-8'))
        c = next(c for c in info['chunks'] if int(c['chunk']) == int(chunk))
        return float(c['n_frames']) / float(info['fps'])
    except (OSError, ValueError, KeyError, StopIteration, TypeError, ZeroDivisionError):
        return None


def make_video(zarr_path, dataset_id, kind='segmentation', chunk=0, seconds=20, log=print):
    """Make a quality-control video of one chunk with `chromas generate`, then convert it to a small H.264 file the
    dashboard (and any browser) can play.  seconds: how much of the chunk to draw, from its start (0 = all of it).
    Returns the path of the video."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    if kind not in VIDEO_KINDS:
        raise ValueError(f'unknown video kind {kind!r} (use one of {", ".join(VIDEO_KINDS)})')
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f'the bundled ffmpeg (imageio-ffmpeg) is needed to convert the video: {exc}') from None
    out_dir = EXPORT_DIR / dataset_id / 'extras'
    out_dir.mkdir(parents=True, exist_ok=True)
    need, step = VIDEO_NEEDS[kind]
    group = Path(zarr_path) / f'chunk_{int(chunk)}'
    if group.is_dir() and not (group / need).exists():
        raise RuntimeError(
            f'{Path(zarr_path).name} has no {step} for chunk {int(chunk)} (its "{need}" is missing), so CHROMAS cannot '
            f'draw this video. The run that made it stopped before the {step} step (a partial result, for example '
            f'after the lookup segmentation failed). Run CHROMAS on the video again, in full, then make the video.')
    raw = PROCESSED_DIR / f'_raw_{dataset_id}_{kind}_{int(chunk)}.mp4'
    reg = raw.with_suffix('.registered.mp4')      # `overlay-video` also writes a second, registered copy
    raw.unlink(missing_ok=True)                  # chromas refuses to overwrite a file
    reg.unlink(missing_ok=True)
    cmd = [st['chromas_exe'], 'generate', VIDEO_KINDS[kind][1], zarr_path, '-c', int(chunk), '-o', raw]
    length = _chunk_seconds(dataset_id, chunk)
    if seconds and int(seconds) > 0 and (length is None or int(seconds) < int(length)):
        cmd += ['-e', int(seconds)]
    PROCESSED_DIR.mkdir(exist_ok=True)
    seen = {'missing': None}

    def stop_on(line):
        # chromas prints "... failed!" with the Python error above it, but still exits with code 0
        m = re.search(r"object has no attribute '(\w+)'", line)
        if m:
            seen['missing'] = m.group(1)
        return 'chromas reported a failure' if line.strip().lower().endswith('failed!') else None
    try:
        try:
            _stream(cmd, st['home'], log, stop_on=stop_on)
        except Stopped:
            raise
        except RuntimeError as exc:
            miss = seen['missing']
            why = (f' The dataset has no "{miss}" (the registration step did not finish for it): run CHROMAS on the '
                   f'video again, in full.' if miss else ' Read the CHROMAS output above for the error.')
            raise RuntimeError(f'chromas could not make the {kind} video ({exc}).{why}') from None
        if not raw.exists() or raw.stat().st_size < 1000:
            raise RuntimeError(f'chromas finished but wrote no video; read its output above. (Chunk {chunk} may not '
                               f'exist in this dataset.)')
        return _finish_video(raw, ffmpeg, out_dir / f'video_{kind}_chunk{int(chunk)}.mp4', log)
    finally:
        raw.unlink(missing_ok=True)
        reg.unlink(missing_ok=True)


def _finish_video(raw, ffmpeg, out, log):
    """Convert chromas' mp4v file to a small H.264 file the browser can play."""
    _stream([ffmpeg, '-y', '-loglevel', 'error', '-i', raw, '-an', '-c:v', 'libx264', '-preset', 'veryfast',
             '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             '-vf', f"scale='min({VIDEO_MAX_WIDTH},iw)':-2,setsar=1", out], BASE_DIR, log)
    log(f'Video saved: {out.name} ({out.stat().st_size / 1e6:.1f} MB)')
    return out


# ── DuckDB ───────────────────────────────────────────────────────────────────────
def connect(read_only=False):
    PROCESSED_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(CHROMAS_DB_PATH), read_only=read_only)
    if not read_only:
        for stmt in SCHEMA:
            con.execute(stmt)
    return con


def load_into_duckdb(dataset_id, export_dir, video_path=None, zarr_path=None, method=None):
    """Replace this dataset's rows in gl_chromas.duckdb with the exported files (one transaction)."""
    export_dir = Path(export_dir)
    info = json.loads((export_dir / 'info.json').read_text(encoding='utf-8'))
    overlay = export_dir / info['overlay'] if info.get('overlay') else None
    con = connect()
    try:
        con.execute('BEGIN')
        for table in ('chromas_runs', 'chromas_areas', 'chromas_focus'):
            con.execute(f'DELETE FROM {table} WHERE dataset_id = ?', (dataset_id,))
        con.execute(
            'INSERT INTO chromas_areas SELECT ?, chunk, frame, orig_frame, t_s, chromatophore, area '
            "FROM read_csv(?, header = true, columns = {'chunk': 'INTEGER', 'frame': 'INTEGER', 'orig_frame': 'INTEGER', "
            "'t_s': 'DOUBLE', 'chromatophore': 'INTEGER', 'area': 'DOUBLE'})",
            (dataset_id, str(export_dir / 'areas.csv')))
        con.execute(
            'INSERT INTO chromas_focus SELECT ?, frame, t_s, in_focus, statistic '
            "FROM read_csv(?, header = true, columns = {'frame': 'INTEGER', 't_s': 'DOUBLE', 'in_focus': 'BOOLEAN', "
            "'statistic': 'DOUBLE'})", (dataset_id, str(export_dir / 'focus.csv')))
        con.execute('INSERT INTO chromas_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
            dataset_id, str(video_path or info['attrs'].get('original_video', '')), str(zarr_path or info['dataset']),
            method or '', info['fps'], info['fps_source'], info['n_chunks'], info['n_chromatophores'], info['n_frames'],
            info['synthetic'], str(overlay) if overlay else None, datetime.now(timezone.utc).isoformat(timespec='seconds'),
            json.dumps(info)))
        con.execute('COMMIT')
        n = con.execute('SELECT count(*) FROM chromas_areas WHERE dataset_id = ?', (dataset_id,)).fetchone()[0]
    except Exception:
        con.execute('ROLLBACK')
        raise
    finally:
        con.close()
    return n


def import_dataset(zarr_path, dataset_id=None, video_path=None, method=None, log=print):
    """Export a finished CHROMAS dataset and load it into DuckDB."""
    zarr_path = Path(zarr_path)
    dataset_id = dataset_id or zarr_path.stem
    out = export_dataset(zarr_path, dataset_id, log)
    n = load_into_duckdb(dataset_id, out, video_path=video_path, zarr_path=zarr_path, method=method)
    log(f'Loaded {n} area values of "{dataset_id}" into {CHROMAS_DB_PATH.name}')
    return dataset_id


def rerun_areas(zarr_path, dataset_id, min_area=200, border=40, log=print):
    """Measure the areas of a finished dataset again with another minimum territory size (`chromas area -cma`, CHROMAS's
    default is 200 px) and another border (`-rb`, default 40 px), then export and load the result.  CHROMAS removes the
    old areas and territories first and numbers the territories again, so wedge, grouping and motor-unit results made
    before no longer match: run those steps again."""
    st = settings()
    bad = check_install(st)
    if bad:
        raise RuntimeError('; '.join(bad))
    cmd = [st['chromas_exe'], 'area', str(zarr_path), '-n', str(st['n_workers'] if st['n_workers'] > 0 else 1), '-t', '2',
           '-cma', str(int(min_area)), '-rb', str(int(border))]
    log(f'Measuring areas again: minimum territory {int(min_area)} px, border {int(border)} px')
    _stream(cmd, st['home'], log, stop_on=lambda line: 'chromas reported a failure' if line.strip().lower().endswith('failed!') else None,
            extra_env={'MPLBACKEND': 'Agg'})
    method = None
    try:
        con = connect()
        try:
            row = con.execute('SELECT method FROM chromas_runs WHERE dataset_id = ?', (dataset_id,)).fetchone()
        finally:
            con.close()
        method = row[0] if row else None
    except Exception:
        pass
    return import_dataset(zarr_path, dataset_id, method=method, log=log)


def run_and_import(video, name=None, method=None, focus_threshold=None, gpu=False, weights=None, log=print,
                   new_lookup=False):
    """The whole chain: chromas run, export, load.  Returns the dataset id."""
    st = settings()
    zarr_path, dataset_id, video = run_pipeline(video, name, method, focus_threshold, gpu, weights, log, new_lookup)
    done = import_dataset(zarr_path, dataset_id, video_path=video, method=method or st['method'], log=log)
    if (method or st['method']) == 'lookup':
        # keep the colours that made this result with the result, so it can be told later which colours were used
        src = Path(weights) if weights else (saved_lookup_table(dataset_id, st) or saved_lookup_table(Path(video).stem, st))
        if src and Path(src).is_file():
            try:
                shutil.copy2(src, EXPORT_DIR / dataset_id / 'lookup_table.npz')
                log(f'The lookup table used (colours) is kept with the result: processed/chromas_export/{dataset_id}/lookup_table.npz')
            except OSError as exc:
                log(f'NOTE: could not keep a copy of the lookup table with the result ({exc}).')
    return done


def load_exports(names=None, method=None, log=print):
    """(Re)load export folders (processed/chromas_export/<name>/) into gl_chromas.duckdb without running CHROMAS.
    names: folder names; None = every folder that has an info.json.  The segmentation method already recorded
    for a dataset is kept unless method is given.  Returns the dataset ids loaded."""
    if names:
        dirs = [EXPORT_DIR / n for n in names]
        missing = [d.name for d in dirs if not (d / 'info.json').exists()]
        if missing:
            raise FileNotFoundError(f'No exported dataset (info.json) in processed/chromas_export/ for: {", ".join(missing)}')
    else:
        dirs = sorted(d for d in EXPORT_DIR.iterdir() if (d / 'info.json').exists()) if EXPORT_DIR.exists() else []
    known = {}
    if CHROMAS_DB_PATH.exists():
        con = connect(read_only=True)
        try:
            known = dict(con.execute('SELECT dataset_id, method FROM chromas_runs').fetchall())
        finally:
            con.close()
    loaded = []
    for d in dirs:
        n = load_into_duckdb(d.name, d, method=method or known.get(d.name) or None)
        log(f'Loaded {n} area values of "{d.name}" into {CHROMAS_DB_PATH.name}')
        loaded.append(d.name)
    if not loaded:
        log('No exported datasets found in processed/chromas_export/.')
    return loaded


# ── Removing datasets ────────────────────────────────────────────────────────────
def protected_names():
    """Datasets named after a folder in examples/ (the example datasets): never removed from the table."""
    d = BASE_DIR / 'examples'
    return {x.name for x in d.iterdir() if x.is_dir()} if d.is_dir() else set()


def _tree_size(path):
    """(files, bytes) under a file or folder; (0, 0) when it is missing."""
    path = Path(path)
    if path.is_file():
        return 1, path.stat().st_size
    n = size = 0
    if path.is_dir():
        for r, _dirs, files in os.walk(path):
            for f in files:
                n += 1
                try:
                    size += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
    return n, size


def _human(n):
    for unit in ('bytes', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'bytes' else f'{n:.1f} {unit}'
        n /= 1024


def removal_plan(dataset_id, also_lookup=False, st=None):
    """What 'remove from dashboard' would delete for one dataset: a list of (what, path or None, files, bytes)."""
    from chromas_activation import OUT_DIR
    items = []
    con_rows = 0
    if CHROMAS_DB_PATH.exists():
        con = connect(read_only=True)
        try:
            con_rows = con.execute('SELECT count(*) FROM chromas_areas WHERE dataset_id = ?', (dataset_id,)).fetchone()[0]
            in_db = con.execute('SELECT count(*) FROM chromas_runs WHERE dataset_id = ?', (dataset_id,)).fetchone()[0]
        finally:
            con.close()
        if in_db or con_rows:
            items.append((f'database rows ({con_rows:,} area values)', None, 0, 0))
    for what, p in (('exported files', EXPORT_DIR / dataset_id), ('Activation results', OUT_DIR / dataset_id)):
        if p.exists():
            n, b = _tree_size(p)
            items.append((what, p, n, b))
    if also_lookup:
        st = st or settings()
        t = st['output_dir'] / f'{dataset_id}.lookup_table.npz'
        if t.is_file():
            items.append(('saved lookup table', t, 1, t.stat().st_size))
    return items


def remove_dataset(dataset_id, also_lookup=False, log=print):
    """Remove a dataset from the dashboard: its database rows, processed/chromas_export/<name>/ and
    processed/chromas_activation/<name>/ (and, if asked, its saved lookup table).  The CHROMAS result (.zarr) is
    NOT touched, so Import from CHROMAS brings the dataset back.  Examples' datasets are refused."""
    if dataset_id in protected_names():
        raise ValueError(f'"{dataset_id}" belongs to examples/ and is not removed here')
    if not dataset_id or '/' in dataset_id or '\\' in dataset_id or dataset_id in ('.', '..'):
        raise ValueError(f'not a dataset name: {dataset_id!r}')
    if CHROMAS_DB_PATH.exists():
        con = connect()
        try:
            con.execute('BEGIN')
            for table in ('chromas_runs', 'chromas_areas', 'chromas_focus'):
                con.execute(f'DELETE FROM {table} WHERE dataset_id = ?', (dataset_id,))
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise
        finally:
            con.close()
        log(f'{dataset_id}: database rows removed')
    problems = []
    for what, p, _n, _b in removal_plan(dataset_id, also_lookup):
        if p is None:
            continue
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            log(f'{dataset_id}: {what} removed ({p})')
        except OSError as exc:
            problems.append(f'{what}: {exc}')
    if problems:
        raise RuntimeError(f'{dataset_id}: removed what it could, but ' + '; '.join(problems) +
                           ' (a file may be open: close the video or stop the dashboard page using it, then try again)')


def trash_zarr(zarr_path, log=print):
    """Move a CHROMAS result (.zarr folder) into a _deleted folder beside it (same drive, so instant and reversible:
    move it back to restore it).  Returns the new path."""
    zarr_path = Path(zarr_path)
    if zarr_path.suffix != '.zarr' or not zarr_path.is_dir():
        raise ValueError(f'not a CHROMAS result folder (.zarr): {zarr_path}')
    dest_dir = zarr_path.parent / '_deleted'
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / f'{zarr_path.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zarr'
    shutil.move(str(zarr_path), str(dest))
    log(f'{zarr_path.name} moved to {dest}')
    return dest


def list_datasets():
    if not CHROMAS_DB_PATH.exists():
        return []
    con = connect(read_only=True)
    try:
        return con.execute('SELECT dataset_id, n_chromatophores, n_frames, fps, method, synthetic, imported_utc '
                           'FROM chromas_runs ORDER BY imported_utc DESC').fetchall()
    finally:
        con.close()


# ── Command line ─────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check', help='check the CHROMAS paths')
    sub.add_parser('list', help='datasets already in the database')
    r = sub.add_parser('run', help='run CHROMAS on a video, then export and load the result')
    r.add_argument('video')
    r.add_argument('--name', help='dataset id (default: video file name)')
    r.add_argument('--method', choices=METHODS, help='segmentation method (default from config)')
    r.add_argument('--focus-threshold', type=float)
    r.add_argument('--weights', help='model file: .pth for neuralnet (default: models/ in this project, else model_weights/ in the CHROMAS folder), saved lookup_table.npz for lookup, rf_segmentation_model_wrapper.pkl for randomforest (required)')
    r.add_argument('--gpu', action='store_true')
    i = sub.add_parser('import', help='export and load an existing CHROMAS dataset (.zarr folder)')
    i.add_argument('zarr')
    i.add_argument('--name')
    ld = sub.add_parser('load', help='(re)load exported folders in processed/chromas_export into the database')
    ld.add_argument('names', nargs='*', help='folder names (default: all)')
    ld.add_argument('--method', choices=METHODS, help='record this segmentation method (default: keep what is recorded)')
    a = ap.parse_args(argv)

    if a.cmd == 'check':
        st = settings()
        for k in ('home', 'chromas_exe', 'python_exe', 'data_dir', 'output_dir'):
            print(f'{k:12} {st[k]}')
        bad = check_install(st)
        print('\n'.join(['Problems:'] + bad) if bad else 'CHROMAS setup looks complete.')
        return 1 if bad else 0
    if a.cmd == 'list':
        for row in list_datasets():
            print(*row, sep='  ')
        return 0
    try:
        if a.cmd == 'load':
            load_exports(a.names or None, a.method)
        elif a.cmd == 'run':
            run_and_import(a.video, a.name, a.method, a.focus_threshold, a.gpu, a.weights)
        else:
            import_dataset(a.zarr, a.name)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
