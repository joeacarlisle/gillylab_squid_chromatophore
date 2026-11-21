"""Activation tab for the dashboard: activation events, neighbour coupling and waves on CHROMAS results.

Used by chromatophore_holoviews.py (one line: a tab).  It runs chromas_activation.py on an imported CHROMAS dataset
(processed/chromas_export/<dataset>/) and shows what it wrote (processed/chromas_activation/<dataset>/):

    summary          plain-language result, with the p values of the neighbour-coupling test
    coupling         neighbour pairs against all other pairs, and coupling against distance
    events           every activation onset (time against position), coloured by the wave it belongs to
    traces           relative area of chosen chromatophores with their event onsets marked
    waves            table of waves (speed, direction, R^2) and an activation map of each one
    replay           the chromatophores animated over time next to the video (replay_view.py)

The analysis runs in a background thread, one at a time, so the page stays usable.
"""
import json
import contextlib
import threading
import time
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

import chromas_activation as ca
import chromatophore_views as cv
import replay_view as rv
import dataset_state

hv.extension('bokeh')
import plot_tools  # noqa: E402
plot_tools.enable()                  # a full-screen button on every plot
pn.extension('tabulator')

EXAMPLES_DIR = ca.BASE_DIR / 'examples'
NO_CONFIG = '(project defaults)'
_job = {'running': False, 'log': [], 'status': '', 'finished': 0}
_job_lock = threading.Lock()


@contextlib.contextmanager
def _timing(label):
    """Prints how long a part of the Activation tab took to build (server side, in the terminal of the dashboard)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f'[activation tab] {label}: {time.perf_counter() - t0:.2f} s', flush=True)


def _timed(label, make):
    with _timing(label):
        return make()


# ── Files ───────────────────────────────────────────────────────────────────────
def _datasets():
    """Imported CHROMAS datasets (export folders), newest first."""
    if not ca.EXPORT_DIR.exists():
        return []
    dirs = [d for d in ca.EXPORT_DIR.iterdir() if (d / 'areas.csv').exists()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def _configs():
    found = sorted(EXAMPLES_DIR.glob('*/example.toml')) if EXAMPLES_DIR.exists() else []
    return {NO_CONFIG: None, **{f'examples/{p.parent.name}': p for p in found}}


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _result_dir(dataset):
    return ca.OUT_DIR / dataset if dataset else None


# ── Background job ──────────────────────────────────────────────────────────────
def _log(text):
    with _job_lock:
        _job['log'].extend(line for line in str(text).splitlines() if line.strip())


def _start(fn, label):
    with _job_lock:
        if _job['running']:
            return False
        _job.update(running=True, log=[], status=f'{label} ...')

    def work():
        try:
            msg = fn()
            status = f'Done: {msg}'
        except Exception as exc:          # shown in the status box
            status = f'Failed: {exc}'
        with _job_lock:
            _job.update(running=False, status=status, finished=_job['finished'] + 1)

    threading.Thread(target=work, daemon=True).start()
    return True


# ── Plots ───────────────────────────────────────────────────────────────────────
def _summary_pane(res):
    txt = (res / 'summary.txt').read_text(encoding='utf-8') if (res / 'summary.txt').exists() else ''
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8')) if (res / 'summary.json').exists() else {}
    tests = [t for t in s.get('coupling_tests', []) if t.get('measure') != 'chance_wave_size']
    coupled = any((t.get('p_value') is not None and t['p_value'] < 0.05) for t in tests)
    parts = []
    if s.get('synthetic'):
        parts.append(pn.pane.Alert('**Synthetic data** with a known answer, made to check the method. '
                                   'Not a recording of an animal.', alert_type='warning', sizing_mode='stretch_width'))
    parts.append(pn.pane.Alert(txt.replace('\n', '  \n'), alert_type='success' if coupled else 'secondary',
                               sizing_mode='stretch_width'))
    return pn.Column(*parts, sizing_mode='stretch_width')


def _coupling_pane(res):
    tests = _read_csv(res / 'coupling_tests.csv')
    prof = _read_csv(res / 'distance_profile.csv')
    items = []
    if len(tests):
        t = tests[tests['measure'] != 'chance_wave_size'].copy()
        if len(t):
            t['verdict'] = np.where(t['p_value'] < 0.05, 'neighbours more coupled', 'no evidence')
            t['n_neighbour_pairs'] = t['n_neighbour_pairs'].astype(int)
            t = t[['chunk', 'measure', 'n_neighbour_pairs', 'neighbour_mean', 'other_mean', 'z', 'p_value', 'verdict']]
            items.append(pn.widgets.Tabulator(t.round(4), show_index=False, disabled=True, theme='midnight',
                                              height=40 + 30 * len(t), sizing_mode='stretch_width'))
    if len(prof):
        prof = prof.copy()
        lo = prof['distance_spacings'].str.split('-').str[0].astype(float)
        hi = prof['distance_spacings'].str.split('-').str[1].replace('inf', np.nan).astype(float)
        prof['mid'] = np.where(hi.notna(), (lo + hi) / 2, lo + 2)
        curves = []
        for col, label, color in (('mean_rate_correlation', 'rate correlation', '#1f77b4'),
                                  ('mean_coincidence', 'event coincidence', '#d62728')):
            for chunk, d in prof.groupby('chunk'):
                pts = d[['mid', col]].dropna().rename(columns={col: 'coupling'})
                lab = label if prof['chunk'].nunique() == 1 else f'{label} (chunk {chunk})'
                pts = pts.assign(measure=lab)
                curves.append(hv.Curve(pts, 'mid', ['coupling', 'measure'], label=lab).opts(color=color, line_width=2,
                                                                                         tools=['hover'], hover_mode='vline'))
                curves.append(hv.Scatter(pts, 'mid', ['coupling', 'measure'], label=lab).opts(color=color, size=7))
        mm = float(prof['spacing_mm'].dropna().iloc[0]) if 'spacing_mm' in prof and prof['spacing_mm'].notna().any() else None
        xlabel = 'distance between centres (in typical spacings' + (f'; 1 spacing = {mm:.3g} mm)' if mm else ')')
        plot = hv.Overlay(curves).opts(height=300, responsive=True, legend_position='top_right', tools=['hover'],
                                       xlabel=xlabel, ylabel='mean coupling',
                                       title='Coupling against distance (neighbours are at about 1)')
        items.append(pn.pane.HoloViews(plot, sizing_mode='stretch_width', linked_axes=False))
        if mm:
            tab = prof[['chunk', 'distance_spacings', 'distance_mm_from', 'distance_mm_to', 'n_pairs',
                        'mean_rate_correlation', 'mean_coincidence']].round(4)
            items.append(pn.widgets.Tabulator(tab, show_index=False, disabled=True, theme='midnight',
                                              height=40 + 30 * len(tab), sizing_mode='stretch_width'))
        else:
            items.append(pn.pane.Markdown('*Distances in mm appear here when the settings give the scale (pixels per mm).*'))
        items.append(pn.pane.Markdown(
            'Coupling shows that chromatophores change together, **not why**. Shared motor units (in cuttlefish one motor '
            'unit spans about ten neighbouring chromatophores; Renard, Ukrow et al. 2026), a spreading wave and mechanical '
            'pull between neighbours all give neighbour coupling. The row *event_coincidence_prominence* repeats the '
            'event test with the cross-check detector.'))
    return pn.Column(*items, sizing_mode='stretch_width') if items else pn.pane.Markdown('*No coupling results.*')


def _synchrony_pane(res):
    y = _read_csv(res / 'synchrony.csv')
    d = _read_csv(res / 'detector_agreement.csv')
    if not len(y):
        return pn.pane.Markdown('*Not in these results: press **Analyse dataset** again to add the whole-patch test.*')
    t = y[['chunk', 'n_chromatophores', 'mean_pair_correlation', 'null_mean', 'p_value', 'median_sync_with_patch',
           'rhythm_hz', 'rhythm_power_share', 'rhythm_clear', 'phase_agreement_R']].copy()
    t['verdict'] = np.where(t['p_value'] >= 0.05, 'no evidence',
                            np.where(t['mean_pair_correlation'] < 0.05, 'shared, very small', 'patch moves together'))
    t.loc[~t['rhythm_clear'].astype(bool), ['rhythm_hz', 'phase_agreement_R']] = np.nan
    t = t[['chunk', 'verdict', 'mean_pair_correlation', 'null_mean', 'p_value', 'rhythm_hz', 'phase_agreement_R',
           'n_chromatophores', 'median_sync_with_patch']]
    items = [pn.widgets.Tabulator(t.round(4), show_index=False, disabled=True, theme='midnight',
                                  height=40 + 30 * len(t), sizing_mode='stretch_width'),
             pn.pane.Markdown(
                 '**mean_pair_correlation**: mean correlation of the expansion rates over all pairs; **null_mean**: the '
                 'same after shifting each signal in time by its own random amount. **rhythm_hz** and '
                 '**phase_agreement_R** (1 = all chromatophores in phase) are shown only when the patch mean has a clear '
                 'rhythm. If the whole patch moves together, as in Humboldt squid flashing, neighbours need not be more '
                 'coupled than distant pairs: an empty neighbour test is then a result, not a failure. Each '
                 "chromatophore's own values are in `chromatophores.csv` (sync_with_patch, phase_deg).")]
    if len(d):
        d = d.copy()
        items += [pn.pane.Markdown('**Cross-check of the events** with a second detector (peaks picked by prominence):'),
                  pn.widgets.Tabulator(d.round(3), show_index=False, disabled=True, theme='midnight',
                                       height=40 + 30 * len(d), sizing_mode='stretch_width')]
    return pn.Column(*items, sizing_mode='stretch_width')


def _events_pane(res):
    ev = _read_csv(res / 'events.csv')
    if not len(ev):
        return pn.pane.Markdown('*No activation events found.*')
    ev = ev.copy()
    order = ev.groupby('chromatophore')['x_px'].first().rank(method='first')
    ev['position_rank'] = ev['chromatophore'].map(order)
    colours = rv.wave_colours(ev)           # the same colours as the replay
    layers = []
    for w, d in sorted(ev.groupby(ev['wave'].astype(int))):
        g, color = ('single event' if w < 0 else f'wave {w}'), colours[w]
        layers.append(hv.Scatter(d, 'onset_s', ['position_rank', 'chromatophore', 'amplitude', 'chunk'], label=g)
                      .opts(color=color, size=6 if g == 'single event' else 8, alpha=0.8))
    plot = hv.Overlay(layers).opts(height=340, responsive=True, legend_position='right', tools=['hover'],
                                   xlabel='onset (s)', ylabel='chromatophores, left to right in the frame',
                                   title='Every activation onset; a wave appears as a slanted line of dots')
    return pn.pane.HoloViews(plot, sizing_mode='stretch_width', linked_axes=False)


def _waves_pane(res):
    wv = _read_csv(res / 'waves.csv')
    if not len(wv):
        return pn.pane.Markdown('*No waves: no group of linked activations was larger than chance gives.*')
    cols = [c for c in ['chunk', 'wave', 'kind', 'start_s', 'spread_s', 'n_chromatophores', 'speed_px_per_s',
                        'speed_mm_per_s', 'direction_deg', 'r2', 'p_r2', 'first_chromatophore'] if c in wv.columns]
    table = pn.widgets.Tabulator(wv[cols].round(3), show_index=False, disabled=True, theme='midnight',
                                 height=min(300, 40 + 30 * len(wv)), sizing_mode='stretch_width')
    maps = sorted(res.glob('wave_map_*.png'), key=lambda p: int(p.stem.split('_')[-1]))
    if not maps:
        return table
    pick = pn.widgets.Select(name='Activation map', options={p.stem.replace('_', ' '): p for p in maps}, width=220)
    img = pn.bind(lambda p: _region_png(p, res, wv, int(p.stem.split('_')[-1])), pick)
    note = pn.pane.Markdown('Colour = onset time (blue first, red last). Small arrows: from the neighbour that '
                            'activated just before. Black arrow: fitted direction of travel. Direction in degrees '
                            'is measured in the image: 0 = right, 90 = down.', sizing_mode='stretch_width')
    return pn.Column(table, pn.Row(pick, note, sizing_mode='stretch_width'), img, sizing_mode='stretch_width')


def _region_png(path, res, waves=None, wave=None, width=760):
    """A saved map (wave_map_<n>.png, pathways.png) cropped to the analysis region, so the chromatophores fill the
    picture (the maps are drawn on the whole video frame), with its caption written above it."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return pn.pane.Markdown(f'*{Path(path).name} could not be read.*')
    H, W = img.shape[:2]
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8')) if (res / 'summary.json').exists() else {}
    roi = (s.get('settings') or {}).get('roi') or s.get('roi')
    tab = _read_csv(res / 'chromatophores.csv')
    if roi:
        x, y, w, h = (int(v) for v in roi)
    elif len(tab):
        used = tab[tab['used']] if 'used' in tab else tab
        x, y = int(used['x_px'].min()), int(used['y_px'].min())
        w, h = int(used['x_px'].max()) - x, int(used['y_px'].max()) - y
    else:
        x, y, w, h = 0, 0, W, H
    pad = 30
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1] if (x1 - x0 > 50 and y1 - y0 > 50) else img
    caption = ''
    if waves is not None and wave is not None and len(waves):
        r = waves[waves['wave'] == wave]
        if len(r):
            r = r.iloc[0]
            caption = (f"**Wave {wave}**: {r['kind']}, {int(r['n_chromatophores'])} chromatophores, spread "
                       f"{r['spread_s']:.2f} s" + (f", {r['speed_mm_per_s']:.2f} mm/s" if r['kind'] == 'travelling'
                                                 and pd.notna(r.get('speed_mm_per_s')) else ''))
    scale = min(1.0, width / crop.shape[1]) if crop.shape[1] > width else 1.0
    out_w = int(crop.shape[1] * scale) if scale < 1 else min(width, crop.shape[1] * 2)
    png = cv2.imencode('.png', crop)[1].tobytes()
    return pn.Column(pn.pane.Markdown(caption) if caption else pn.Spacer(height=0), pn.pane.PNG(png, width=out_w))


def _trace_pane(dataset, res, cells):
    if not cells:
        return pn.pane.Markdown('*Choose chromatophores.*')
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
    cfg = {**ca.DEFAULTS, **s.get('settings', {})}
    data = ca.load_export(ca.resolve_export(dataset))
    ev = _read_csv(res / 'events.csv')
    curves = []
    for chunk, (wide, times, cent) in data['chunks'].items():
        keep = [c for c in cells if c in wide.columns]
        if not keep:
            continue
        rel, _, _ = ca.prepare(wide[keep], cent.loc[keep], data['fps'], {**cfg, 'roi': []})
        for c in rel.columns:
            colour = rv.cell_colour(cells, c)          # the same colour as its label in the replay
            d = pd.DataFrame({'time (s)': times, 'relative area': rel[c].to_numpy(), 'chromatophore': int(c)})
            curves.append(hv.Curve(d, 'time (s)', ['relative area', 'chromatophore'], label=f'#{c}')
                          .opts(line_width=1.5, color=colour, tools=['hover'], hover_mode='vline'))
    if not curves:
        gone = [c for c in cells if not any(c in w.columns for w, _, _ in data['chunks'].values())]
        if gone:
            return pn.pane.Alert(f'Chromatophores {", ".join(str(c) for c in gone)} are not in the dataset as it is now: it '
                                 'was imported again after the analysis. Press **Analyse dataset** to refresh the results.',
                                 alert_type='warning', sizing_mode='stretch_width')
        return pn.pane.Markdown('*These chromatophores were left out of the analysis.*')
    marks = ev[ev['chromatophore'].isin(cells)] if len(ev) else ev
    vlines = [hv.VLine(t).opts(color='black', line_dash='dotted', line_width=1) for t in marks.get('onset_s', [])]
    plot = hv.Overlay(curves + vlines).opts(height=320, responsive=True, legend_position='right', tools=['hover'],
                                            title='Relative area (0 = at rest); dotted lines = detected onsets')
    return pn.pane.HoloViews(plot, sizing_mode='stretch_width', linked_axes=False)


# ── The tab ─────────────────────────────────────────────────────────────────────
def activation_panel(dataset_sel):
    """The Activation tab for the dataset chosen in the sidebar selector `dataset_sel`."""
    defaults = ca.load_settings()
    configs = _configs()
    config_sel = pn.widgets.Select(name='Settings file', options=list(configs), value=NO_CONFIG, width=240)
    px_in = pn.widgets.FloatInput(name='Pixels per mm (0 = unknown)', value=float(defaults['px_per_mm']), step=1,
                                  width=180)
    roi_in = pn.widgets.TextInput(name='Region: x y width height (empty = whole frame)', width=300,
                                  value=' '.join(str(int(v)) for v in defaults['roi']))
    z_in = pn.widgets.FloatInput(name='Onset (noise sigmas)', value=float(defaults['onset_z']), step=0.5, width=150)
    amp_in = pn.widgets.FloatInput(name='Min. growth (share of rest)', value=float(defaults['min_amplitude']),
                                   step=0.05, width=170)
    lag_in = pn.widgets.FloatInput(name='Max. lag between neighbours (s)', value=float(defaults['max_lag_s']),
                                   step=0.05, width=200)
    coin_in = pn.widgets.FloatInput(name='Coincidence window (s)', value=float(defaults['coincidence_s']),
                                    step=0.05, width=170)
    nb_in = pn.widgets.FloatInput(name='Neighbour edge limit (x median)', value=float(defaults['neighbour_factor']),
                                  step=0.1, width=200)
    analyse_btn = pn.widgets.Button(name='Analyse dataset', button_type='primary')
    check_btn = pn.widgets.Button(name='Check the method (synthetic data)')
    reset_settings_btn = pn.widgets.Button(name='Reset settings to defaults')
    status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
    log_view = pn.widgets.TextAreaInput(value='', disabled=True, height=150, sizing_mode='stretch_width',
                                        name='Output (newest first)')
    results = pn.Column(sizing_mode='stretch_width')
    cells_sel = pn.widgets.MultiSelect(name='Chromatophores (most events first)', options=[], size=10, width=200)
    # the same choice, offered inside the Replay panel too (a widget can sit in only one place), kept in step both ways
    replay_sel = pn.widgets.MultiSelect(name='Chromatophores to ring (most events first)', options=[], size=8, width=230)
    cells_sel.link(replay_sel, options='options')
    cells_sel.link(replay_sel, value='value', bidirectional=True)
    reset_sel_btn = pn.widgets.Button(name='Reset to the most active', width=200)
    reset_sel_btn.on_click(lambda _: setattr(cells_sel, 'value', list(cells_sel.options)[:4]))
    seen = {'finished': 0}

    def _settings():
        roi = [float(v) for v in roi_in.value.replace(',', ' ').split()] if roi_in.value.strip() else []
        if roi and len(roi) != 4:
            raise ValueError('The region needs four numbers: x y width height.')
        return ca.load_settings(configs[config_sel.value], px_per_mm=float(px_in.value), roi=roi,
                                onset_z=float(z_in.value), min_amplitude=float(amp_in.value),
                                max_lag_s=float(lag_in.value), coincidence_s=float(coin_in.value),
                                neighbour_factor=float(nb_in.value))

    def _on_config(*_):
        cfg = ca.load_settings(configs[config_sel.value])
        px_in.value = float(cfg['px_per_mm'])
        roi_in.value = ' '.join(str(int(v)) for v in cfg['roi'])
        for w, k in ((z_in, 'onset_z'), (amp_in, 'min_amplitude'), (lag_in, 'max_lag_s'),
                     (coin_in, 'coincidence_s'), (nb_in, 'neighbour_factor')):
            w.value = float(cfg[k])


    def _render_now(*_):
        ds = dataset_sel.value
        if not ds:
            results.objects = [pn.pane.Markdown('*No CHROMAS dataset yet. Add one in the **Datasets** tab, '
                                                'or press "Check the method" to analyse synthetic data.*')]
            return
        export = ca.EXPORT_DIR / ds
        if not (export / 'info.json').exists():
            results.objects = [pn.pane.Alert(
                f'**{ds}** is in the database but its exported files are missing, so it cannot be analysed. Import it '
                'again in the **Datasets** tab (it needs the CHROMAS result, `' + ds + '.zarr`).',
                alert_type='warning', sizing_mode='stretch_width')]
            return
        if not (export / 'centroids.csv').exists():
            results.objects = [pn.pane.Alert(
                f'**{ds}** was imported before chromatophore positions were exported, so it cannot be analysed yet. '
                'Import it again in the **Datasets** tab (it needs the CHROMAS result, `' + ds + '.zarr`).', alert_type='warning',
                sizing_mode='stretch_width')]
            return
        res = _result_dir(ds)
        if not (res / 'summary.json').exists():
            results.objects = [pn.pane.Markdown(f'*{ds} has not been analysed yet. Choose settings and press '
                                                '"Analyse dataset".*')]
            return
        ev = _read_csv(res / 'events.csv')
        top = list(ev['chromatophore'].value_counts().index.astype(int)) if len(ev) else []
        cells_sel.options = top
        cells_sel.value = top[:4]
        def lazy(title, make, label):
            return cv.lazy_card(title, lambda: _timed(label, make))
        stale = (res / 'summary.json').stat().st_mtime < (export / 'areas.csv').stat().st_mtime
        items = ([pn.pane.Alert(
            '**These results are older than the data.** The dataset was imported again after it was analysed, and each '
            'CHROMAS run numbers its chromatophores afresh, so the numbers below (and in the lists) may belong to '
            'chromatophores that no longer exist. Press **Analyse dataset** above to refresh them.',
            alert_type='warning', sizing_mode='stretch_width')] if stale else []) + [
                 pn.Card(_summary_pane(res), title='Result', collapsed=False, sizing_mode='stretch_width'),
                 lazy('Are neighbours activated together?', lambda: _coupling_pane(res), 'coupling'),
                 lazy('Does the whole patch move together?', lambda: _synchrony_pane(res), 'synchrony'),
                 lazy('Activation events', lambda: _events_pane(res), 'events'),
                 lazy('Traces of the chosen chromatophores', lambda: pn.Row(
                     pn.Column(cells_sel, reset_sel_btn, width=220),
                     pn.Column(pn.bind(_trace_pane, ds, res, cells_sel), sizing_mode='stretch_width'),
                     sizing_mode='stretch_width'), 'traces'),
                 lazy('Waves', lambda: _waves_pane(res), 'waves'),
                 lazy('Replay', lambda: pn.Column(
                     pn.Row(replay_sel, pn.pane.Markdown(
                         'Choose the chromatophores to ring and number here: with **Numbers: Selected only** the '
                         'replay rings these (the four most active to begin with). It is the same list as in the '
                         'Traces panel, so a choice made in either one shows in both.', width=520)),
                     rv.replay_pane(ds, cells_sel), sizing_mode='stretch_width'), 'replay')]
        if (res / 'pathways.png').exists():
            items.append(lazy('Neighbours and pathway links (first chunk)', lambda: _region_png(res / 'pathways.png', res),
                              'pathway map'))
        files = ', '.join(p.name for p in sorted(res.iterdir()) if p.suffix == '.csv')
        items.append(pn.pane.Markdown(f'*All results are in `{res}`: {files}.*'))
        results.objects = items

    def _render(*_):
        with _timing('result summary (every other panel is built when opened)'):
            _render_now()

    def _analyse(_):
        ds = dataset_sel.value
        if not ds:
            return
        try:
            cfg = _settings()
        except ValueError as exc:
            with _job_lock:
                _job['status'] = f'Failed: {exc}'
            _poll()
            return
        def run():
            ca.analyse(ds, cfg, log=_log)
            dataset_state.bump(select=ds)
            return f'"{ds}" analysed.'
        poller()
        _start(run, f'Analysing {ds}')

    def _check(_):
        def run():
            ok = ca.run_example(ca.load_settings(), log=_log)
            return ('the method found the planted waves and no coupling in the control (datasets synthetic_waves '
                    'and synthetic_no_coupling).' if ok else 'some checks FAILED; read the output below.')
        poller()
        _start(run, 'Checking the method on synthetic data')

    def _poll():
        with _job_lock:
            lines, msg, running, finished = list(_job['log'][-60:]), _job['status'], _job['running'], _job['finished']
        log_view.value = '\n'.join(reversed(lines))
        status.visible = bool(msg)
        status.object = msg
        status.alert_type = 'info' if running else ('danger' if msg.startswith('Failed') else 'success')
        for b in (analyse_btn, check_btn):
            b.disabled = running
        if finished != seen['finished']:
            seen['finished'] = finished
            _render()

    poller = dataset_state.poll_while_running(_poll, lambda: _job['running'], period=1500)

    analyse_btn.on_click(_analyse)
    check_btn.on_click(_check)
    config_sel.param.watch(_on_config, 'value')
    reset_settings_btn.on_click(_on_config)         # puts back the values of the chosen settings file (or the defaults)
    def _on_dataset(*_):
        # a dataset with its own examples/<name>/example.toml gets those settings (scale, region) by default
        key = f'examples/{dataset_sel.value}'
        if dataset_sel.value and key in configs and config_sel.value != key:
            config_sel.value = key                 # _on_config fills the fields
        _render()

    dataset_sel.param.watch(_on_dataset, 'value')
    if pn.state.curdoc is not None:
        pn.state.onload(lambda: (_poll(), poller() if _job['running'] else None))   # show a job already going
    _on_dataset()

    intro = pn.pane.Markdown(
        'Finds when each chromatophore **activates** (expands faster than its own noise allows and grows at least the '
        'minimum share), tests whether **neighbouring chromatophores activate together** (permutation test, p value), '
        'and finds **waves** of activation with their speed and direction, for the dataset chosen in the sidebar. '
        'A dataset with its own `examples/` folder gets its settings file (scale, region) by itself; otherwise choose '
        'one or enter them. Details: README, '
        '"Activation, neighbours and waves".')
    form = pn.Column(
        pn.pane.Markdown('### Analyse a CHROMAS dataset'), intro,
        pn.Row(config_sel),
        pn.Row(px_in, roi_in),
        pn.Row(z_in, amp_in, lag_in, coin_in, nb_in),
        pn.Row(analyse_btn, check_btn, reset_settings_btn), status, log_view, sizing_mode='stretch_width')
    return pn.Column(form, pn.layout.Divider(), results, sizing_mode='stretch_width')
