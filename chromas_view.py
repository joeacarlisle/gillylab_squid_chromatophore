"""CHROMAS parts of the dashboard: the run form (in the Datasets tab) and the Chromatophores tab, which shows the
chromatophore areas of the dataset chosen in the sidebar (dataset_state.py).

Used by chromatophore_holoviews.py and import_view.py.  Reads processed/gl_chromas.duckdb, which chromas_bridge.py
writes.  The database is opened read-only for each query and closed again, so a run that is importing results is
not blocked by the dashboard (and the other way round).

What CHROMAS measures: for every chromatophore it follows through the video, the area in pixels of its pigment
sac.  A chromatophore expands when the muscles pull its sac open and contracts when they relax, so the area
over time is the direct record of that cell's activity.  Brightness-in-a-box (the other tabs) is the sum of many.
"""
import contextlib
import re
import threading
import time
import weakref
from pathlib import Path

import duckdb
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

import chromas_bridge as cb
import lookup_picker as lp
import chromatophore_views as cv
from replay_view import cell_colour
import dataset_state

hv.extension('bokeh')            # also loaded by the dashboard; repeating it is harmless
import plot_tools  # noqa: E402
plot_tools.enable()                  # a full-screen button on every plot
pn.extension('tabulator')

HEATMAP_MAX_COLUMNS = 1500          # the heat map averages frames together beyond this many columns
TOP_N = 8                      # chromatophores plotted by default (the ones whose area changes most)
cv.TOP_N_DEFAULT = TOP_N
_job = {'running': False, 'log': [], 'status': '', 'started': 0.0, 'stopping': False}     # one run at a time; shared by all browser sessions
_job_lock = threading.Lock()


def _q(sql, params=()):
    """Run a query on gl_chromas.duckdb (read-only, opened per call).  None if the database does not exist yet."""
    if not cb.CHROMAS_DB_PATH.exists():
        return None
    con = duckdb.connect(str(cb.CHROMAS_DB_PATH), read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


@contextlib.contextmanager
def _timing(label):
    """Prints how long a part of the Chromatophores tab took to build (server side, in the terminal that runs the
    dashboard).  Drawing in the browser comes on top of this."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f'[chromatophores tab] {label}: {time.perf_counter() - t0:.2f} s', flush=True)


_cache = {}


def _db_stamp():
    """Changes whenever the database does: an import, a slice step, ... (the dashboard bumps dataset_state's version
    after each, and the files' times catch a change made elsewhere)."""
    out = [dataset_state.version()]
    for suffix in ('', '.wal'):
        f = Path(str(cb.CHROMAS_DB_PATH) + suffix)
        out.append(f.stat().st_mtime_ns if f.exists() else 0)
    return tuple(out)


def _cached(key, compute):
    """compute() once per database state: the same query is needed by several parts of the tab."""
    stamp = _db_stamp()
    hit = _cache.get(key)
    if hit is None or hit[0] != stamp:
        with _timing(f'query {key[0]}'):
            hit = _cache[key] = (stamp, compute())
    return hit[1]


def _datasets():
    df = _cached(('runs',), lambda: _q('SELECT * FROM chromas_runs ORDER BY imported_utc DESC'))
    return df.copy() if df is not None else pd.DataFrame()


def _cell_stats(dataset_id):
    """Per chromatophore: mean area, and how much it moves (5th-95th percentile range relative to the median)."""
    return _cached(('cell stats', dataset_id), lambda: _q("""SELECT chromatophore, avg(area) AS mean_area,
                        stddev(area) AS sd_area,
                        (quantile_cont(area, 0.95) - quantile_cont(area, 0.05)) / nullif(median(area), 0) AS rel_range
                 FROM chromas_areas WHERE dataset_id = ? GROUP BY chromatophore ORDER BY rel_range DESC""",
                                                                   (dataset_id,))).copy()


# The Run CHROMAS form of each browser session, so the Training tab can put a trained model in it (step 8).  Keyed by
# the session's Bokeh document; None (no server, e.g. a test) has its own slot.
_RUN_FORMS = weakref.WeakKeyDictionary()
_RUN_FORM_NO_DOC = {}


def run_form_widgets():
    """(method_in, weights_in) of the Run CHROMAS form of this session, or None if it has not been built yet."""
    doc = pn.state.curdoc
    return _RUN_FORM_NO_DOC.get('form') if doc is None else _RUN_FORMS.get(doc)


def run_form():
    """The "Run CHROMAS on a video" form (part of the Datasets tab)."""
    st = cb.settings()
    problems = cb.check_install(st)

    # ── Run form ─────────────────────────────────────────────────────────────────
    # a drop-down of the videos in the CHROMAS data folder; any other file name or full path can still be typed
    video_in = pn.widgets.AutocompleteInput(name='Video (file name or full path)', options=cb.source_video_names(st), restrict=False,
                                            case_sensitive=False, min_characters=0, placeholder='example.mp4', width=420)
    refresh_videos_btn = pn.widgets.Button(name='Refresh', width=80)
    refresh_videos_btn.on_click(lambda _: setattr(video_in, 'options', cb.source_video_names(st)))   # a video added to the data folder
    name_in = pn.widgets.TextInput(name='Dataset name (empty = the video name)', value='', width=240,
                                   placeholder='e.g. chromas_example')
    method_in = pn.widgets.Select(name='Segmentation', options=cb.METHODS, value=st['method'], width=200)
    focus_in = pn.widgets.FloatInput(name='Focus threshold', value=st['focus_threshold'], step=0.005, width=140)
    minchunk_in = pn.widgets.IntInput(name='Min chunk (frames)', value=st['min_chunk_frames'], step=10, start=10, width=140)
    gpu_in = pn.widgets.Checkbox(name='Use GPU', value=False)
    _torch = cb.torch_build(st)
    if _torch is not None and not _torch['cuda']:
        # --gpu switches CHROMAS to a GPU-only path with no CPU fallback: with a CPU-only PyTorch the run would fail
        gpu_in.disabled = True
        gpu_in.name = f'Use GPU (off: PyTorch {_torch["version"]} in CHROMAS is CPU-only)'
    weights_in = pn.widgets.TextInput(name='Model file (neuralnet: .pth, empty = models/ folder; randomforest: your trained .pkl, required)', value='', width=420)
    if pn.state.curdoc is None:
        _RUN_FORM_NO_DOC['form'] = (method_in, weights_in)
    else:
        _RUN_FORMS[pn.state.curdoc] = (method_in, weights_in)
    run_btn = pn.widgets.Button(name='Run CHROMAS', button_type='primary', disabled=bool(problems))
    status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
    # a busy bar, a clock and a Stop button next to Run CHROMAS, like the ones under the extra CHROMAS results
    job_bar = pn.indicators.Progress(name='job', active=False, value=None, bar_color='primary', visible=False,
                                     sizing_mode='stretch_width', height=18)
    job_text = pn.pane.Markdown('', visible=False, sizing_mode='stretch_width')
    stop_btn = pn.widgets.Button(name='Stop', button_type='danger', visible=False, width=90)
    log_view = pn.widgets.TextAreaInput(value='', disabled=True, height=200, sizing_mode='stretch_width',
                                        name='CHROMAS output (last 40 lines, newest first)')

    def _weights_visible(*_):
        weights_in.visible = method_in.value != 'lookup'       # lookup takes its colours from the colour picker below

    method_in.param.watch(_weights_visible, 'value')
    _weights_visible()

    def _work(video, method, focus, gpu, weights, new_lookup, name, minchunk=None):
        def log(line):
            with _job_lock:
                _job['log'].append(line)
        try:
            name = cb.run_and_import(video, name=name or None, method=method, focus_threshold=focus, gpu=gpu,
                                     weights=weights or None, log=log, new_lookup=new_lookup, min_chunk_frames=minchunk)
            _job['status'] = (f'Done: dataset "{name}" is loaded and chosen in the sidebar. See it in the '
                              'Chromatophores tab; the Activation tab analyses its events, neighbour coupling and waves.')
            dataset_state.bump(select=name)
        except cb.Stopped as exc:
            _job['status'] = f'Stopped: {exc}'
        except Exception as exc:
            _job['status'] = f'Failed: {exc}'
        finally:
            _job['running'] = False

    def _start(_):
        if _job['running']:
            return
        if not video_in.value.strip():
            with _job_lock:
                _job['status'] = 'Failed: enter a video first.'
            _poll()
            return
        weights = weights_in.value.strip()
        # Run CHROMAS with lookup always opens CHROMAS's own clicking window: no saved table is reused.  The colour
        # picker passes its table in the model file box, so a run started from the picker uses those colours.
        new_lookup = method_in.value == 'lookup' and not weights
        with _job_lock:
            _job.update(running=True, log=[], status='Running CHROMAS. A long video can take hours; this page stays usable.',
                        started=time.time(), stopping=False)
        cb.clear_stop()
        poller()
        threading.Thread(target=_work, daemon=True,
                         args=(video_in.value.strip(), method_in.value, float(focus_in.value), gpu_in.value,
                             weights, new_lookup, name_in.value.strip(), int(minchunk_in.value or 100))).start()

    run_btn.on_click(_start)

    def _stop(_):
        with _job_lock:
            if not _job['running']:
                return
            _job['stopping'] = True
        cb.request_stop()
        _poll()

    stop_btn.on_click(_stop)

    def _run_with_table(table_path):
        """Start a run with a table made in the colour picker (the model file box is only used for this call)."""
        if _job['running']:
            return 'A CHROMAS run is already going. Wait for it to finish.'
        method_in.value = 'lookup'
        weights_in.value = str(table_path)
        _start(None)                            # reads the form at once
        weights_in.value = ''
        return None

    picker = lp.ColourPicker(video_in, on_run=_run_with_table, name_in=name_in)
    picker_card = pn.Card(picker.panel, title='Choose the lookup colours here, with a live preview (recommended)',
                          collapsed=False, sizing_mode='stretch_width')
    window_note = pn.pane.Markdown(
        'CHROMAS opens its own window on the **first frame** of the video, as it does by itself (this project no longer '
        'changes the frame). Click the **background first, then at least one chromatophore**, then press Q. Only the '
        'first click is background: any later click is a chromatophore class. **Run CHROMAS** with lookup works as `chromas run` '
        'does: it opens this window every time (a table saved earlier is not reused), and CHROMAS saves the clicked table '
        'as lookup_table.npz in its output folder (replaced by the next run) and inside the dataset; this project keeps a '
        'copy with the dataset\'s export (processed/chromas_export/<name>/lookup_table.npz). A run '
        'never replaces an existing dataset: if the name is taken, it stops before the window opens and suggests a free '
        'name. To choose the colours on any frame, with a preview and with all '
        'background clicks counted as background, use the panel above.', sizing_mode='stretch_width')
    window_card = pn.Card(window_note, title='Or let CHROMAS open its own clicking window', collapsed=True,
                          sizing_mode='stretch_width')

    train_card = pn.pane.Markdown(
        'To train a **random forest** or fine-tune the **neural network** on your own painted frames, use the '
        '**6 Training** tab: its last step puts the trained model in the *Model file* box here and chooses the method.',
        sizing_mode='stretch_width')

    def _cards_visibility(*_):
        picker_card.visible = window_card.visible = method_in.value == 'lookup'
        train_card.visible = method_in.value in ('randomforest', 'neuralnet')
    method_in.param.watch(_cards_visibility, 'value')
    _cards_visibility()

    def _poll():
        with _job_lock:
            lines, msg, running = list(_job['log'][-40:]), _job['status'], _job['running']
        log_view.value = '\n'.join(reversed(lines))      # newest first: the box does not scroll to the end by itself
        status.visible = bool(msg)
        status.object = msg
        status.alert_type = ('info' if running else 'danger' if msg.startswith('Failed')
                             else 'warning' if msg.startswith('Stopped') else 'success')
        run_btn.disabled = running or bool(problems)
        job_bar.visible = job_bar.active = running
        job_text.visible = running
        stop_btn.visible = running
        stop_btn.disabled = _job['stopping']
        if running:
            secs = int(time.time() - _job['started'])
            step = next((l for l in reversed(lines) if l.startswith('> ')), '')
            pct = None                                  # chromas prints "(37.50%)" or a tqdm bar for the step it is on
            for l in reversed(lines[-3:]):
                m = re.search(r'(\d+(?:\.\d+)?)\s*%', l)
                if m:
                    pct = min(100.0, float(m.group(1)))
                    break
            job_bar.value = int(pct) if pct is not None else None   # None: a moving bar, progress unknown
            text = f'**Working: running CHROMAS** - {secs // 60}:{secs % 60:02d} elapsed' + (' - starting' if not step else '')
            if pct is not None:
                text += f' - this step {pct:.0f}%'
            job_text.object = '**Stopping ...** the process is being ended' if _job['stopping'] else text
        if not running:
            video_in.options = cb.source_video_names(st)        # and a video may have been added to the folders

    poller = dataset_state.poll_while_running(_poll, lambda: _job['running'], period=2000)

    if pn.state.curdoc is not None:               # periodic updates only exist inside a running server
        pn.state.onload(lambda: (_poll(), poller() if _job['running'] else None))   # show a run already going

    setup_note = (pn.pane.Alert('CHROMAS is not set up: ' + '; '.join(problems), alert_type='warning',
                                sizing_mode='stretch_width') if problems else
                  pn.pane.Markdown(f'CHROMAS found in `{st["home"]}`. Videos go in `{st["data_dir"]}`.'))
    return pn.Column(pn.pane.Markdown('### Run CHROMAS on a video'), setup_note,
                     pn.Row(video_in, pn.Column(pn.Spacer(height=22), refresh_videos_btn), name_in, method_in, focus_in, minchunk_in, pn.Column(pn.Spacer(height=22), gpu_in)),
                     pn.pane.Markdown(
                         'A file name is looked for in the CHROMAS data folder, `videos/` and `examples/<name>/`; '
                         '`.mp4` may be left off. **Lookup** labels pixels by colour, so it needs colours from this '
                         'video: choose them with the colour picker below (with a preview), or **Run CHROMAS** opens '
                         'CHROMAS\'s own window, where you click the **background first, then at least one '
                         'chromatophore**, then press Q (every run, as `chromas run` does). '
                         '**Neuralnet** needs no clicks: it uses the supplied network, or one you fine-tuned, given in the '
                         'box below. **Randomforest** needs the model file you trained, in the box below. Both are trained '
                         'in the **6 Training** tab.', sizing_mode='stretch_width'),
                     pn.Row(weights_in),
                     picker_card, window_card, train_card,
                     pn.Row(run_btn, job_text, stop_btn, sizing_mode='stretch_width', align='center'), job_bar,
                     status, log_view, sizing_mode='stretch_width')


def _in_db(dataset_id):
    ds = _datasets()
    return bool(dataset_id) and len(ds) and dataset_id in set(ds['dataset_id'])


def chromas_panel(dataset_sel):
    """The Chromatophores tab: areas of the dataset chosen in the sidebar selector `dataset_sel`."""
    cells_sel = pn.widgets.MultiSelect(name='Chromatophores', options=[], size=10, width=200)
    relative = pn.widgets.Checkbox(name='Relative change (area / median - 1)', value=True)

    def _on_dataset(*_):
        if not _in_db(dataset_sel.value):
            cells_sel.options, cells_sel.value = [], []
            return
        s = _cell_stats(dataset_sel.value)
        cells_sel.options = [int(c) for c in s['chromatophore']]        # most active first
        cells_sel.value = [int(c) for c in s['chromatophore'].head(TOP_N)]

    dataset_sel.param.watch(_on_dataset, 'value')
    _on_dataset()

    def _banner(dataset_id):
        if not dataset_id:
            return pn.pane.Markdown('*No CHROMAS dataset yet. Add one in the **Datasets** tab.*')
        if not _in_db(dataset_id):
            return pn.pane.Alert(f'**{dataset_id}** is not loaded in the database yet. Load it in the **Datasets** tab '
                                 '(tick it, then "Load exported results" or "Import from CHROMAS").',
                                 alert_type='warning', sizing_mode='stretch_width')
        r = _datasets().query('dataset_id == @dataset_id').iloc[0]
        try:                                   # CHROMAS numbers more chromatophores than it measures: say how many it did
            d = cv.load(dataset_id)
            measured = int(d['areas'].replace(0, np.nan).notna().any().sum())
            count = (f'{measured} chromatophores measured (CHROMAS numbered {int(r.n_chromatophores)}; it leaves out '
                     f'territories outside its body mask, within 40 px of the edge or under 200 px)')
        except Exception:
            count = f'{int(r.n_chromatophores)} chromatophore numbers'
        txt = (f'**{dataset_id}**: {count}, {int(r.n_chunks)} in-focus chunk(s), '
               f'{int(r.n_frames)} frames at {r.fps:g} fps (from {r.fps_source}); method: {r.method or "unknown"}; '
               f'imported {r.imported_utc}.')
        if r.synthetic:
            return pn.Column(pn.pane.Alert('**Synthetic data.** This dataset was generated to test the pipeline '
                                           '(make_fake_chromas.py). It is not a recording of an animal.',
                                           alert_type='warning', sizing_mode='stretch_width'), pn.pane.Markdown(txt))
        return pn.pane.Markdown(txt)

    def _areas(dataset_id, cells, rel):
        if not _in_db(dataset_id) or not cells:
            return pn.pane.Markdown('*Select a dataset and chromatophores.*')
        with _timing('areas plot'):
            return _areas_plot(dataset_id, cells, rel)

    def _areas_plot(dataset_id, cells, rel):
        df = _q(f'SELECT chunk, t_s, chromatophore, area FROM chromas_areas WHERE dataset_id = ? '
                f'AND chromatophore IN ({",".join("?" * len(cells))}) ORDER BY chromatophore, t_s', (dataset_id, *cells))
        df['value'] = df['area'] / df.groupby('chromatophore')['area'].transform('median') - 1 if rel else df['area']
        curves = []
        for c, d in df.groupby('chromatophore'):
            # a gap (out-of-focus part) is drawn as a break, not a straight line across it
            for _, seg in d.groupby('chunk'):
                curves.append(hv.Curve(seg, hv.Dimension('t_s', label='time (s)'),
                                       [hv.Dimension('value', label='relative change' if rel else 'area (px)'),
                                        'chromatophore', hv.Dimension('area', label='area (px)')], label=f'#{c}').opts(
                    line_width=1.5, color=cell_colour(cells, c), tools=['hover'], hover_mode='vline'))   # colours as in the flowers and the replay
        plot = hv.Overlay(curves).opts(height=340, responsive=True, xlabel='time in the video (s)', tools=['hover'],
                                       ylabel='relative area change' if rel else 'area (pixels)', legend_position='right',
                                       title='Area of each selected chromatophore')
        return pn.pane.HoloViews(plot, sizing_mode='stretch_width', linked_axes=False)

    def _heatmap(dataset_id):
        if not _in_db(dataset_id):
            return pn.pane.Markdown('')
        with _timing('heat map'):
            return _heatmap_plot(dataset_id)

    def _heatmap_plot(dataset_id):
        df = _q('SELECT t_s, chromatophore, area FROM chromas_areas WHERE dataset_id = ?', (dataset_id,))
        order = list(_cell_stats(dataset_id)['chromatophore'])
        wide = df.groupby(['chromatophore', 't_s'])['area'].mean().unstack().reindex(order)     # much faster than pivot_table
        if wide.shape[1] > HEATMAP_MAX_COLUMNS:           # more frames than the screen has pixels: average neighbours
            k = -(-wide.shape[1] // HEATMAP_MAX_COLUMNS)
            grp = np.arange(wide.shape[1]) // k
            times = wide.columns.values.astype(float)
            t_mean = pd.Series(times).groupby(grp).mean().values
            wide = wide.T.groupby(grp).mean().T
            wide.columns = t_mean
        # a chromatophore with no area in half the frames has median 0: leave its row blank instead of dividing by 0
        rel = wide.div(wide.median(axis=1).replace(0, np.nan), axis=0) - 1
        finite = np.abs(rel.values)[np.isfinite(rel.values)]
        lim = float(np.percentile(finite, 98)) if finite.size else 0.0
        lim = lim if np.isfinite(lim) and lim > 0 else 1.0
        t = rel.columns.values
        img = hv.Image(np.flipud(rel.values), bounds=(float(t.min()), 0, float(t.max()), len(order)),
                       kdims=['time (s)', 'chromatophores, most active at the top'], vdims=['relative area change'])
        return pn.pane.HoloViews(img.opts(cmap='RdBu_r', clim=(-lim, lim), colorbar=True, height=360, responsive=True,
                                          tools=['hover'], yaxis=None, title='All chromatophores (rows sorted by activity)'),
                                 sizing_mode='stretch_width', linked_axes=False)

    def _table(dataset_id):
        if not _in_db(dataset_id):
            return pn.pane.Markdown('')
        s = _cell_stats(dataset_id).round(3)
        s.columns = ['chromatophore', 'mean area (px)', 'sd of area (px)', 'relative range (5-95%)']
        return pn.widgets.Tabulator(s, theme='midnight', show_index=False, disabled=True, height=320, width=560)

    def _overlay(dataset_id):
        if not _in_db(dataset_id):
            return pn.pane.Markdown('')
        p = _datasets().query('dataset_id == @dataset_id')['overlay_path'].iloc[0]
        if not p or not Path(p).exists():
            return pn.pane.Markdown('*(no outline picture for this dataset)*')
        return pn.Column(pn.pane.Markdown('### Chromatophore territories'), pn.pane.PNG(str(p), width=440))

    def _section(build, *args, needs_positions=True):
        # a view of the export folder; says what is missing instead of failing
        def f(dataset_id):
            if not dataset_id:
                return pn.pane.Markdown('')
            exp = cv.export_dir(dataset_id)
            if not (exp / 'areas.csv').exists() or (needs_positions and not (exp / 'centroids.csv').exists()):
                return pn.pane.Markdown('*Needs the exported files with chromatophore positions: import the dataset '
                                        'again in the **Datasets** tab.*')
            try:
                with _timing(build.__name__):
                    return build(dataset_id, *args)
            except Exception as exc:              # one broken view must not take the tab down
                return pn.pane.Alert(f'Could not draw this view: {exc}', alert_type='warning')
        return pn.bind(f, dataset_sel)

    def _lazy(title, make):
        """A collapsed panel that builds its content the first time it is opened (make() returns the content, which then
        follows the dataset selector).  Panels that stay closed cost nothing."""
        card = pn.Card(pn.pane.Markdown('*Open this panel to draw it.*'), title=title, collapsed=True,
                       sizing_mode='stretch_width')
        built = []

        def _open(event):
            if not event.new and not built:
                built.append(True)
                card[:] = [make()]
        card.param.watch(_open, 'collapsed')
        return card

    view = pn.Column(
        pn.bind(_banner, dataset_sel),
        pn.pane.Markdown('### Map'), _section(cv.map_view, cells_sel),
        pn.pane.Markdown('### Chromatophore areas'),
        pn.Row(pn.Column(cells_sel, relative, width=220),
               pn.Column(pn.bind(_areas, dataset_sel, cells_sel, relative), sizing_mode='stretch_width'),
               sizing_mode='stretch_width'),
        _lazy('All chromatophores (heat map)', lambda: pn.bind(_heatmap, dataset_sel)),
        _lazy('Sizes and activity', lambda: _section(cv.distributions_view)),
        _lazy('Groups that move together', lambda: _section(cv.groups_view, cells_sel)),
        _lazy('Slices (flower plots)', lambda: _section(cv.slices_view, cells_sel)),
        _lazy('Activity per chromatophore and territories',
              lambda: pn.Row(pn.Column(pn.bind(_table, dataset_sel)), pn.bind(_overlay, dataset_sel))),
        pn.pane.Markdown('## Quality checks'),
        _lazy('Focus', lambda: _section(cv.focus_view, needs_positions=False)),
        _lazy('Pigment coverage', lambda: _section(cv.coverage_view, needs_positions=False)),
        _lazy('Registration between chunks', lambda: _section(cv.registration_view, needs_positions=False)),
        _lazy('Segmentation check', lambda: _section(cv.segmentation_view, needs_positions=False)),
        _lazy('Overlay video', lambda: _section(cv.video_view, needs_positions=False)),
        sizing_mode='stretch_width')
    return view
