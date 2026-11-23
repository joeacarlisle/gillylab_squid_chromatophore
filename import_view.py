"""Datasets tab for the dashboard: everything that builds CHROMAS data, in one place.

Used by chromatophore_holoviews.py (one line: a tab).  At the top, the form that runs CHROMAS on a video
(chromas_view.run_form).  Below, every CHROMAS dataset it can find and what has been done with it, with the two
import steps of chromas_bridge.py:

    Import from CHROMAS   read a finished CHROMAS dataset (<name>.zarr) with chromas_export.py in the CHROMAS Python,
                          write processed/chromas_export/<name>/, and load it into processed/gl_chromas.duckdb
    Load exported         load processed/chromas_export/<name>/ into the database again (no CHROMAS needed),
                          e.g. for results exported on another computer

A dataset imported before positions were exported (no centroids.csv) cannot be analysed in the Activation tab;
importing it again from its .zarr adds them.  Runs in a background thread, one job at a time.
"""
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import panel as pn

import chromas_activation as ca
import chromas_bridge as cb
import dataset_state
from chromas_view import run_form

pn.extension('tabulator')

METHOD_KEEP = '(keep / unknown)'
_job = {'running': False, 'log': [], 'status': '', 'finished': 0, 'started': 0.0, 'stopping': False}
_job_lock = threading.Lock()


def _log(text):
    with _job_lock:
        _job['log'].extend(line for line in str(text).splitlines() if line.strip())


def _start(fn, label):
    with _job_lock:
        if _job['running']:
            return False
        _job.update(running=True, log=[], status=f'{label} ...', started=time.time(), stopping=False)
    cb.clear_stop()

    def work():
        try:
            status = f'Done: {fn()}'
        except cb.Stopped as exc:
            status = f'Stopped: {exc}'
        except Exception as exc:          # shown in the status box
            status = f'Failed: {exc}'
        with _job_lock:
            _job.update(running=False, status=status, finished=_job['finished'] + 1)

    threading.Thread(target=work, daemon=True).start()
    return True


def _zarrs(extra_dir=None):
    """{dataset name: .zarr path} in the CHROMAS output folder (and one more folder, if given)."""
    found = {}
    dirs = []
    try:
        dirs.append(cb.settings()['output_dir'])
    except Exception:
        pass
    if extra_dir:
        dirs.append(Path(extra_dir))
    for d in dirs:
        if d.is_dir():
            for z in sorted(d.glob('*.zarr')):
                found.setdefault(z.stem, z)
    return found


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _slash(values):
    return ' / '.join(str(v) for v in values)


def run_summary(exp, db=None):
    """What the Datasets table says about how good a run is.  The counts of measured chromatophores, median areas and
    focus rows come from the database (`db`: this dataset's entry of chromas_bridge.area_summary(), None when the
    dataset is not loaded); the rest from small files of the export folder (info.json, run_info.json,
    extras/extras.json).  The areas and focus CSV files are not read."""
    out = {}
    try:
        info = json.loads((exp / 'info.json').read_text(encoding='utf-8'))
        chunks = info.get('chunks', [])
        numbered = [int(c['n_chromatophores']) for c in chunks]
        out['numbered'] = int(info.get('n_chromatophores', max(numbered, default=0)))
        out['fps'] = info.get('fps')
        out['in focus'] = f"{info.get('n_frames', '')}"
        out['ranges'] = ' | '.join(f"{c['start_frame']}-{c['start_frame'] + c['n_frames'] - 1}" for c in chunks)
        out['agree'] = ('yes' if len(set(numbered)) <= 1 else 'NO: numbered ' + _slash(numbered)) if numbered else ''
        if db is not None:
            if db.get('focus_rows') is not None:            # the whole video: every frame has a focus row
                out['in focus'] = f"{info.get('n_frames', '')} of {db['focus_rows']}"
            out['measured'] = db['measured']
            per_chunk = [db['chunks'].get(int(c['chunk']), (0, 0.0)) for c in chunks]
            out['measured per chunk'] = _slash(n for n, _ in per_chunk)
            out['median area'] = _slash(int(round(m)) if n else 0 for n, m in per_chunk)
    except (OSError, ValueError, KeyError):
        pass
    try:
        ri = json.loads((exp / 'run_info.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        ri = {}
    found = {}
    try:
        found = json.loads((exp / 'extras' / 'extras.json').read_text(encoding='utf-8')).get('found', {})
    except (OSError, ValueError):
        pass
    stale = False                                           # steps made before the areas were last measured no longer match
    try:
        stale = datetime.fromisoformat(ri['areas_utc']).timestamp() > _mtime(exp / 'extras' / 'extras.json') > 0
    except (KeyError, ValueError):
        pass
    for key, name in (('slices', 'slice step'), ('clusters', 'grouping'), ('motor_units', 'motor units')):
        out[name] = ('stale' if stale else 'yes') if found.get(key) else 'no'
    out['colours'] = (exp / 'lookup_table.npz').exists()
    return out


def _yn(flag):
    """'yes' or 'no', like the other columns of the table."""
    return 'yes' if flag else 'no'


def inventory(extra_dir=None):
    """One row per dataset found anywhere: CHROMAS output, export folders, database, activation results."""
    zarrs = _zarrs(extra_dir)
    exported = {d.name: d for d in ca.EXPORT_DIR.iterdir() if (d / 'info.json').exists()} if ca.EXPORT_DIR.exists() else {}
    in_db = {r[0]: r for r in cb.list_datasets()}
    stats = cb.area_summary()
    names = sorted(set(zarrs) | set(exported) | set(in_db))
    rows = []
    for n in names:
        exp = exported.get(n)
        has_pos = bool(exp and (exp / 'centroids.csv').exists())
        analysed = (ca.OUT_DIR / n / 'summary.json').exists()
        if not exp:
            todo = 'Import from CHROMAS' if n in zarrs else 'Exported files missing: needs its .zarr'
        elif not has_pos:
            todo = ('Import from CHROMAS again (adds positions)' if n in zarrs else
                    'No positions (CHROMAS tab only); the Activation tab needs its .zarr')
        elif n not in in_db:
            todo = 'Load exported'
        elif not analysed:
            todo = 'Ready: analyse in the Activation tab'
        else:
            todo = 'Done'
        db = in_db.get(n)
        sm = run_summary(exp, stats.get(n)) if exp else {}
        method = (db[4] or '') if db else ''
        measured = sm.get('measured')
        rows.append({'dataset': n, 'CHROMAS result (.zarr)': str(zarrs[n]) if n in zarrs else '',
                     'exported': _yn(exp), 'positions': _yn(has_pos), 'in database': _yn(n in in_db),
                     'method': method,
                     'measured / numbered': (f"{measured} of {sm['numbered']}" if measured is not None else
                                             (str(int(db[1])) if db else '')),
                     'chunks agree': sm.get('agree', ''),
                     'measured per chunk': sm.get('measured per chunk', ''),
                     'median area per chunk (px)': sm.get('median area', ''),
                     'frames in focus': sm.get('in focus', ''),
                     'chunk frames (video)': sm.get('ranges', ''),
                     'fps': sm.get('fps'),
                     'lookup table kept': ('yes' if sm.get('colours') else 'no') if method == 'lookup' else '',
                     'imported (UTC)': str(db[6])[:16].replace('T', ' ') if db else '',
                     'analysed (Activation)': _yn(analysed),
                     'slice step': sm.get('slice step', ''), 'grouping': sm.get('grouping', ''),
                     'motor units': sm.get('motor units', ''),
                     'quality checks': _yn(exp and (exp / 'extras' / 'extras.json').exists()), 'next step': todo})
    return pd.DataFrame(rows), zarrs, exported


def datasets_panel(dataset_sel=None):
    """The Datasets tab.  dataset_sel is the sidebar selector: a dataset that is open there cannot be removed or have its
    result deleted while other datasets exist (the other tabs are showing it)."""
    st = cb.settings()
    problems = cb.check_install(st)
    extra_in = pn.widgets.TextInput(name='Also look for .zarr results in this folder (optional)', width=420,
                                    placeholder=r'e.g. C:\data\chromas_runs')
    refresh_btn = pn.widgets.Button(name='Refresh', width=90)
    untick_btn = pn.widgets.Button(name='Untick all', width=100)
    table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, disabled=True, theme='midnight',
                                 selectable='checkbox', height=320, sizing_mode='stretch_width')
    method_sel = pn.widgets.Select(name='Segmentation method used (recorded with the data)',
                                   options=[METHOD_KEEP] + list(cb.METHODS), value=METHOD_KEEP, width=300)
    extras_btn = pn.widgets.Button(name='Export quality checks', disabled=bool(problems))
    slice_btn = pn.widgets.Button(name='Run slice step', disabled=bool(problems))
    cluster_btn = pn.widgets.Button(name='Run CHROMAS grouping', disabled=bool(problems))
    units_btn = pn.widgets.Button(name='Run motor-unit analysis', disabled=bool(problems))
    area_btn = pn.widgets.Button(name='Measure areas again', disabled=bool(problems))
    area_min = pn.widgets.IntInput(name='Minimum territory area (px)', value=200, start=1, step=10, width=190)
    area_border = pn.widgets.IntInput(name='Border to remove (px)', value=40, start=0, step=5, width=170)
    area_dil = pn.widgets.IntInput(name='Mask widening (default 10)', value=10, start=1, step=5, width=190)
    slice_chunks = pn.widgets.TextInput(name='Chunks for slice and grouping (empty = all)', placeholder='e.g. 0,2', width=200)
    chunk_info = pn.pane.Markdown('', sizing_mode='stretch_width')
    video_kind = pn.widgets.Select(name='Quality-control video', options=list(cb.VIDEO_KINDS), width=170,
                                   value='segmentation')
    video_chunk = pn.widgets.IntInput(name='Chunk', value=0, start=0, width=70)
    video_secs = pn.widgets.IntInput(name='Seconds (0 = whole chunk)', value=20, start=0, width=170)
    video_btn = pn.widgets.Button(name='Make video', disabled=bool(problems))
    import_btn = pn.widgets.Button(name='Import from CHROMAS (.zarr)', button_type='primary',
                                   disabled=bool(problems))
    load_btn = pn.widgets.Button(name='Load exported results')
    remove_btn = pn.widgets.Button(name='Remove from dashboard', button_type='warning')
    zarr_btn = pn.widgets.Button(name='Delete CHROMAS result (.zarr)', button_type='danger')
    confirm_text = pn.pane.Markdown('', sizing_mode='stretch_width')
    lookup_chk = pn.widgets.Checkbox(name='Also remove the saved lookup table of each', value=False)
    typed_in = pn.widgets.TextInput(name='Type the dataset name to confirm', width=320, visible=False)
    confirm_btn = pn.widgets.Button(name='Yes', button_type='danger', width=190)
    cancel_btn = pn.widgets.Button(name='Cancel', width=90)
    confirm_box = pn.Column(confirm_text, lookup_chk, typed_in, pn.Row(confirm_btn, cancel_btn), visible=False,
                            sizing_mode='stretch_width',
                            styles={'border': '2px solid #c0392b', 'border-radius': '6px', 'padding': '6px 14px'})
    pend = {'kind': None, 'names': []}
    status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
    # a busy bar and a clock next to the buttons, so a running job (a CHROMAS run, the slice step, a video) is seen
    # without scrolling to the output box
    job_bar = pn.indicators.Progress(name='job', active=False, value=None, bar_color='primary', visible=False,
                                     sizing_mode='stretch_width', height=18)
    job_text = pn.pane.Markdown('', visible=False, sizing_mode='stretch_width')
    stop_btn = pn.widgets.Button(name='Stop', button_type='danger', visible=False, width=90)
    log_view = pn.widgets.TextAreaInput(value='', disabled=True, height=160, sizing_mode='stretch_width',
                                        name='Output (newest first)')
    state = {'zarrs': {}, 'exported': {}, 'finished': 0, 'version': dataset_state.version(), 'n_log': 0, 't_log': 0.0}

    def _refresh(*_):
        df, state['zarrs'], state['exported'] = inventory(extra_in.value.strip() or None)
        table.value = df
        table.selection = []
        _show_chunks()

    def _show_chunks(*_):
        # the chunks of the first ticked dataset with a result, and which ones the slice step can take together
        names = [n for n in _selected() if n in state['zarrs']]
        if not names:
            chunk_info.object = ''
            return
        rows, suggestion = cb.chunk_summary(state['zarrs'][names[0]])
        if not rows:
            chunk_info.object = ''
            return
        lines = ['| chunk | frames | chromatophores numbered |', '|---|---|---|']
        lines += [f"| {r['chunk']} | {r['frames']} | {r['chromatophores']} |" for r in rows]
        if suggestion:
            note = (f'**{names[0]}**: the chunks do not all have the same number of chromatophores, so the slice step '
                    f'cannot take them all. Type **{suggestion}** in the Chunks box (the largest group with the same number).')
        else:
            note = f'**{names[0]}**: every chunk has the same number of chromatophores, so leave the Chunks box empty.'
        chunk_info.object = note + '\n\n' + '\n'.join(lines)

    def _selected():
        df = table.value
        return [df.iloc[i]['dataset'] for i in (table.selection or []) if i < len(df)]

    def _method():
        return None if method_sel.value == METHOD_KEEP else method_sel.value

    def _say(msg):
        with _job_lock:
            _job['status'] = msg
        _poll()                                   # no job runs, so nothing else would show it

    def _extra(step):
        # quality checks / slice / grouping for the ticked datasets that have a CHROMAS result (.zarr)
        def handler(_):
            names = _selected()
            todo = [n for n in names if n in state['zarrs']]
            if not todo:
                _say('Failed: tick one or more datasets that have a CHROMAS result (.zarr) in the table.')
                return
            zarrs = dict(state['zarrs'])
            try:
                chunks = cb.parse_chunk_list(slice_chunks.value)
            except ValueError as exc:
                _say(f'Failed: {exc}')
                return

            def run():
                for n in todo:
                    if step == 'extras':
                        cb.export_extras(zarrs[n], n, log=_log)
                    else:
                        cb.run_extra_step(zarrs[n], n, step, log=_log, chunks=chunks if step in ('slice', 'cluster') else '')
                dataset_state.bump(select=todo[-1])
                tab = 'Chromatophores' if step == 'extras' else 'Advanced analytics'
                return f'{", ".join(todo)}: done. See the {tab} tab.'
            label = {'extras': 'Exporting quality checks', 'slice': 'Running the slice step (can take a while)',
                     'cluster': 'Running CHROMAS grouping',
                     'motor_units': 'Running the motor-unit analysis (PCA, ICA, clustering)'}[step]
            poller()
            _start(run, f'{label} for {", ".join(todo)}')
        return handler

    def _remeasure(_):
        # the area step again on the ticked datasets, with another minimum territory size, border and mask widening (CHROMAS defaults: 200, 40 and 10)
        todo = [n for n in _selected() if n in state['zarrs']]
        if not todo:
            _say('Failed: tick one or more datasets that have a CHROMAS result (.zarr) in the table.')
            return
        zarrs, mn, bd, dl = dict(state['zarrs']), int(area_min.value), int(area_border.value), int(area_dil.value)

        def run():
            for n in todo:
                cb.rerun_areas(zarrs[n], n, mn, bd, log=_log, dilation=dl)
            dataset_state.bump(select=todo[-1])
            return (f'{", ".join(todo)}: areas measured again (minimum territory {mn} px, border {bd} px, mask widening {dl}). Territory numbers can '
                    'change: run the slice, grouping and motor-unit steps again if you had run them.')
        poller()
        _start(run, f'Measuring areas again for {", ".join(todo)} (minimum territory {mn} px, border {bd} px, mask widening {dl})')

    def _video(_):
        # a quality-control video of one chunk, for the ticked datasets that have a CHROMAS result (.zarr)
        todo = [n for n in _selected() if n in state['zarrs']]
        if not todo:
            _say('Failed: tick one or more datasets that have a CHROMAS result (.zarr) in the table.')
            return
        zarrs, kind, chunk, secs = dict(state['zarrs']), video_kind.value, int(video_chunk.value), int(video_secs.value)

        def run():
            for n in todo:
                cb.make_video(zarrs[n], n, kind, chunk, secs, log=_log)
            dataset_state.bump(select=todo[-1])
            return f'{", ".join(todo)}: {kind} video of chunk {chunk} done. Play it in the Chromatophores tab (Quality checks).'
        poller()
        _start(run, f'Making the {kind} video of chunk {chunk} for {", ".join(todo)} (needs the CHROMAS Python; '
                    f'about as long as the video)')

    def _import(_):
        names = _selected()
        todo = [n for n in names if n in state['zarrs']]
        if not todo:
            _say('Failed: tick one or more datasets that have a CHROMAS result (.zarr) in the table.')
            return
        zarrs, method = dict(state['zarrs']), _method()
        known = dict(zip(table.value['dataset'], table.value['method']))

        def run():
            for n in todo:
                cb.import_dataset(zarrs[n], n, method=method or known.get(n) or None, log=_log)
            dataset_state.bump(select=todo[-1])
            skipped = [n for n in names if n not in todo]
            extra = f' Skipped (no .zarr found): {", ".join(skipped)}.' if skipped else ''
            return f'imported {", ".join(todo)}.{extra}'
        poller()
        _start(run, f'Importing {", ".join(todo)}')

    def _load(_):
        names = _selected()
        todo = [n for n in names if n in state['exported']]
        if not todo:
            _say('Failed: tick one or more datasets that are exported (the "exported" column).')
            return
        method = _method()
        def run():
            done = cb.load_exports(todo, method, log=_log)
            dataset_state.bump(select=done[-1] if done else None)
            return f'loaded {", ".join(done)}.'
        poller()
        _start(run, f'Loading {", ".join(todo)}')

    def _close_confirm(*_):
        pend.update(kind=None, names=[])
        confirm_box.visible = False
        typed_in.value = ''
        lookup_chk.value = False

    def _open_in_sidebar(names):
        """(refusal text or '', note for the confirm box): a dataset chosen in the sidebar is in use by the other tabs."""
        cur = getattr(dataset_sel, 'value', None) if dataset_sel is not None else None
        used = [n for n in names if n == cur]
        if not used:
            return '', ''
        if len(list(dataset_sel.options)) > 1:
            return (f'Failed: {", ".join(used)} is the dataset chosen in the sidebar, so the other tabs are showing it. '
                    'Choose a different dataset in the sidebar first, then delete this one.'), ''
        return '', (f'  \n**{used[0]} is open in the sidebar and is the only dataset. Reload the page (F5) after it is '
                    'gone.**')

    def _ask_remove(_):
        names = _selected()
        if not names:
            _say('Failed: tick one or more datasets in the table.')
            return
        refusal, use_note = _open_in_sidebar(names)
        if refusal:
            _say(refusal)
            return
        protected = [n for n in names if n in cb.protected_names()]
        if protected:
            _say(f'Failed: {", ".join(protected)} are example datasets (named after a folder in examples/) and '
                 'cannot be removed here. Untick them.')
            return
        lines = []
        for n in names:
            plan = cb.removal_plan(n)
            what = '; '.join(f'{w}' + (f' ({f:,} files, {cb._human(b)})' if p is not None else '') for w, p, f, b in plan)
            kept = ('Kept: its CHROMAS result (.zarr), so **Import from CHROMAS** brings it back.' if n in state['zarrs']
                    else '**Warning: there is no .zarr for it here, so it cannot be brought back after this.**')
            lines.append(f'- **{n}**: removes {what or "nothing (already gone)"}. {kept}')
        confirm_text.object = ('**Remove from the dashboard?**  \n' + '\n'.join(lines) +
                               '  \nVideos, lookup tables and CHROMAS results are not touched unless you tick the box.' + use_note)
        pend.update(kind='remove', names=names)
        lookup_chk.visible, typed_in.visible = True, False
        confirm_btn.name = 'Yes, remove from dashboard'
        confirm_box.visible = True

    def _ask_zarr(_):
        names = _selected()
        if len(names) != 1:
            _say('Failed: tick exactly one dataset to delete its CHROMAS result (.zarr).')
            return
        n = names[0]
        if n not in state['zarrs']:
            _say(f'Failed: {n} has no CHROMAS result (.zarr) in the list.')
            return
        refusal, use_note = _open_in_sidebar(names)
        if refusal:
            _say(refusal)
            return
        z = state['zarrs'][n]
        files, size = cb._tree_size(z)
        confirm_text.object = (
            f'**Delete the CHROMAS result of "{n}"?**  \nThe folder `{z}` ({files:,} files, {cb._human(size)}) is moved '
            f'into a `_deleted` folder beside it, not erased: move it back to restore it. The dashboard copies of the '
            f'dataset stay. Making the result again means running CHROMAS again, which takes hours for a long video.  \n'
            f'Type the dataset name below to confirm.' + use_note)
        pend.update(kind='zarr', names=[n])
        lookup_chk.visible, typed_in.visible = False, True
        typed_in.value = ''
        confirm_btn.name = 'Yes, delete the .zarr'
        confirm_box.visible = True

    def _confirm(_):
        kind, names = pend['kind'], list(pend['names'])
        if not kind or not names:
            return
        if kind == 'zarr':
            if typed_in.value.strip() != names[0]:
                _say(f'Nothing deleted: the name typed does not match "{names[0]}".')
                return
            z = state['zarrs'].get(names[0])

            def run():
                dest = cb.trash_zarr(z, log=_log)
                dataset_state.bump()
                return f'{names[0]}.zarr moved to {dest}. Move it back to restore it.'
            label = f'Moving the CHROMAS result of {names[0]} to _deleted'
        else:
            also = bool(lookup_chk.value)

            def run():
                for n in names:
                    cb.remove_dataset(n, also, log=_log)
                dataset_state.bump()
                return f'removed {", ".join(names)} from the dashboard (CHROMAS results kept).'
            label = f'Removing {", ".join(names)} from the dashboard'
        _close_confirm()
        poller()
        _start(run, label)

    def _stop(_):
        with _job_lock:
            if not _job['running']:
                return
            _job['stopping'] = True
        cb.request_stop()
        _poll()

    def _poll():
        with _job_lock:
            lines, msg, running, finished = list(_job['log'][-60:]), _job['status'], _job['running'], _job['finished']
            started, stopping, n_log = _job['started'], _job['stopping'], len(_job['log'])
        if n_log != state['n_log'] or not state['t_log']:          # new output: note when it came
            state['n_log'], state['t_log'] = n_log, time.time()
        log_view.value = '\n'.join(reversed(lines))
        status.visible = bool(msg)
        status.object = msg
        status.alert_type = ('info' if running else 'danger' if msg.startswith('Failed')
                             else 'warning' if msg.startswith('Stopped') else 'success')
        job_bar.visible = job_bar.active = running
        job_text.visible = running
        stop_btn.visible = running
        stop_btn.disabled = stopping
        if running:
            secs = int(time.time() - started)
            stage = next((l for l in reversed(lines) if l.startswith('> ')), '')
            what = ('converting the video for the browser' if 'ffmpeg' in stage.lower() or 'libx264' in stage else
                    'CHROMAS is working' if stage else 'starting')
            pct = None                                  # chromas prints "(37.50%)" or a tqdm bar for the step it is on
            for l in reversed(lines[-3:]):
                m = re.search(r'(\d+(?:\.\d+)?)\s*%', l)
                if m:
                    pct = min(100.0, float(m.group(1)))
                    break
            job_bar.value = int(pct) if pct is not None else None   # None: a moving bar, progress unknown
            idle = int(time.time() - state['t_log'])
            text = f'**Working: {msg.rstrip(" .")}** - {what} - {secs // 60}:{secs % 60:02d} elapsed'
            if pct is not None:
                text += f' - this step {pct:.0f}%'
            if stopping:
                text = '**Stopping ...** the process is being ended'
            elif idle >= 30:
                text += f' - no new output for {idle // 60}:{idle % 60:02d} (press Stop if this is too long)'
            job_text.object = text
        for b in (import_btn, extras_btn, slice_btn, cluster_btn, units_btn, area_btn, video_btn):
            b.disabled = running or bool(problems)
        load_btn.disabled = running
        for b in (remove_btn, zarr_btn, confirm_btn):
            b.disabled = running
        if finished != state['finished'] or dataset_state.version() != state['version']:
            state['finished'], state['version'] = finished, dataset_state.version()
            _refresh()

    poller = dataset_state.poll_while_running(_poll, lambda: _job['running'], period=1500)

    table.param.watch(_show_chunks, 'selection')
    refresh_btn.on_click(_refresh)
    untick_btn.on_click(lambda _: setattr(table, 'selection', []))
    import_btn.on_click(_import)
    extras_btn.on_click(_extra('extras'))
    slice_btn.on_click(_extra('slice'))
    cluster_btn.on_click(_extra('cluster'))
    area_btn.on_click(_remeasure)
    units_btn.on_click(_extra('motor_units'))
    video_btn.on_click(_video)
    stop_btn.on_click(_stop)
    load_btn.on_click(_load)
    remove_btn.on_click(_ask_remove)
    zarr_btn.on_click(_ask_zarr)
    confirm_btn.on_click(_confirm)
    cancel_btn.on_click(_close_confirm)
    extra_in.param.watch(_refresh, 'value')
    if pn.state.curdoc is not None:
        pn.state.onload(lambda: (_poll(), poller() if _job['running'] else None))   # show a job already going
    _refresh()

    setup = (pn.pane.Alert('CHROMAS is not set up, so results cannot be read from .zarr folders (exported results can '
                           'still be loaded): ' + '; '.join(problems), alert_type='warning', sizing_mode='stretch_width')
             if problems else pn.pane.Markdown(f'CHROMAS results are looked for in `{st["output_dir"]}`.'))
    intro = pn.pane.Markdown(
        'Brings CHROMAS results into the dashboard. **Import from CHROMAS** reads a finished CHROMAS dataset '
        '(`<name>.zarr`, using the CHROMAS Python), saves areas, positions and the territory picture to '
        '`processed/chromas_export/<name>/`, and loads them into the database for the CHROMAS tab. **Load exported '
        'results** only does the last part, for results exported earlier or on another computer. Runs started in the '
        'form above are imported automatically. Tick datasets in the table, then press a button. The dataset you work on in '
        'the other tabs is chosen in the sidebar; a new import is chosen there by itself.',
        sizing_mode='stretch_width')
    return pn.Column(
        run_form(), pn.layout.Divider(),
        pn.pane.Markdown('### Import CHROMAS results'), intro, setup,
        pn.Row(extra_in, pn.Column(pn.Spacer(height=22), pn.Row(refresh_btn, untick_btn))), table,
        pn.Row(method_sel, pn.Column(pn.Spacer(height=22), pn.Row(import_btn, load_btn))),
        pn.pane.Markdown('**Extra CHROMAS results** for the ticked datasets (they need the `.zarr`): **Export quality '
                         'checks** saves pigment coverage, segmentation samples, registration between chunks and motion '
                         'markers for the Chromatophores tab (an import does this too); **Run slice step** cuts each '
                         'chromatophore into 36 wedges for the flower plots (minutes to an hour; the Chunks box limits it to some chunks, '
                         'for example 0,2, which a dataset needs when its chunks have different numbers of chromatophores); **Run CHROMAS '
                         'grouping** runs `chromas analyse cluster areas` on each chunk of the Chunks box (empty: the chunks that match); **Run motor-unit analysis** (after the slice step) runs the '
                         'CHROMAS papers\' own method, `chromas ica` and `chromas analyse cluster ics`, for the '
                         'Advanced analytics tab.', sizing_mode='stretch_width'),
        pn.Row(extras_btn, slice_btn, cluster_btn, units_btn, slice_chunks),
        pn.pane.Markdown('**Measure areas again** runs CHROMAS\'s area step (`chromas area`) with another **minimum territory '
                         'area**, **border** and **mask widening**, then loads the result. CHROMAS leaves out territories under the minimum area '
                         'and within the border of the frame (its defaults are 200 px and 40 px); a smaller minimum measures more, '
                         'smaller territories, which are often debris. CHROMAS keeps only the largest connected group of pigmented regions as the body and gives territories only inside it, so chromatophores far from the main group get none; a larger **mask widening** (try 30 to 40) joins them to it, and also adds territory in the gaps. It replaces the areas of the ticked dataset and can '
                         'renumber the territories, so run the slice, grouping and motor-unit steps again afterwards.', sizing_mode='stretch_width'),
        pn.Row(area_btn, area_min, area_border, area_dil),
        chunk_info,
        pn.Row(job_text, stop_btn, sizing_mode='stretch_width', align='center'), job_bar,
        pn.pane.Markdown('**Quality-control video** for the ticked datasets: `chromas generate` draws what it found on the '
                         'footage - the **segmentation** (what it took for pigment) or each chromatophore\'s '
                         '**territory** - for one chunk, from its start. Play it in the Chromatophores tab, Quality checks. '
                         'The best way to see pigment it missed, or debris it took for pigment.',
                         sizing_mode='stretch_width'),
        pn.Row(video_kind, video_chunk, video_secs, pn.Column(pn.Spacer(height=22), video_btn)),
        pn.pane.Markdown('**Remove datasets** (tick them in the table first). **Remove from dashboard** deletes the '
                         'dataset\'s database rows, exported files and Activation results; its CHROMAS result (.zarr) is '
                         'kept, so it can be imported again. **Delete CHROMAS result (.zarr)** moves one result folder '
                         'into a `_deleted` folder (recoverable) and asks you to type its name. Example '
                         'datasets (named after a folder in `examples/`) are never removed.', sizing_mode='stretch_width'),
        pn.Row(remove_btn, zarr_btn), confirm_box,
        status, log_view, sizing_mode='stretch_width')
