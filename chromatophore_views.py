"""More views of a CHROMAS dataset for the Chromatophores tab (used by chromas_view.py).

    Map              each chromatophore at its position, coloured by a measure; click or box-select to plot it
    Distributions    resting size, largest expansion, share of time expanded
    Groups           correlation matrix ordered by group, and a map of the groups (chromatophores that move
                     together, the CHROMAS paper's motor-unit analysis); groups from CHROMAS if it computed them,
                     else computed here the same way (correlation of the area time series, affinity propagation)
    Slices           "flower" plots: how much each of the 36 wedges of a chromatophore expands (after `chromas slice`)
    Quality checks   focus over time, pigment coverage, registration between chunks, segmentation samples

Reads the database (areas, focus) and the export folder processed/chromas_export/<dataset>/ (centroids,
overlay.png, extras/ written by chromas_export.py).  Plots have fixed widths: responsive plots nested this deep
were drawn with no width by Bokeh 3.9.
"""
import functools
import json
from pathlib import Path

import cv2
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

import chromas_activation as ca

hv.extension('bokeh')          # also loaded by the dashboard; repeating it is harmless
import plot_tools  # noqa: E402
plot_tools.enable()                  # a full-screen button on every plot
pn.extension('tabulator')

W = 980                   # width of the wide plots (pixels)


def lazy_card(title, make, before_open=None):
    """A collapsed panel that builds its content the first time it is opened: make() returns the content (usually a
    pn.bind, which then follows the dataset selector).  before_open(), if given, runs first, every time the panel is
    opened for the first time.  Panels that stay closed cost nothing."""
    card = pn.Card(pn.pane.Markdown('*Open this panel to draw it.*'), title=title, collapsed=True,
                   sizing_mode='stretch_width')
    built = []

    def _open(event):
        if not event.new and not built:
            built.append(True)
            if before_open is not None:
                before_open()
            card[:] = [make()]
    card.param.watch(_open, 'collapsed')
    return card
EXPANDED = 0.2            # a chromatophore counts as expanded when its area is 20% above its resting area
GROUP_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2', '#bcbd22',
                '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd', '#e6550d', '#31a354', '#756bb1']
MEASURES = {'activity (5-95% range / median)': 'activity', 'resting size (px)': 'rest_px',
            'largest expansion (x rest)': 'expansion', 'time expanded (%)': 'time_expanded_pct', 'group': 'group'}


# ── Data ────────────────────────────────────────────────────────────────────────
def export_dir(dataset):
    return ca.EXPORT_DIR / dataset


def extras_dir(dataset):
    return export_dir(dataset) / 'extras'


def _read(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@functools.lru_cache(maxsize=8)
def _load(dataset, stamp):
    """Area table (frames x chromatophores, all chunks stacked), times, chunk of each row, centroids."""
    data = ca.load_export(ca.resolve_export(dataset))
    walls, times, chunks, cents = [], [], [], []
    for chunk, (wide, t, cent) in sorted(data['chunks'].items()):
        walls.append(wide)
        times.append(np.asarray(t, float))
        chunks.append(np.full(len(wide), chunk))
        cents.append(cent)
    areas = pd.concat(walls, ignore_index=True).astype(float)
    cent = pd.concat(cents).groupby(level=0).first()            # first chunk's position of each chromatophore
    return {'areas': areas, 't': np.concatenate(times), 'chunk': np.concatenate(chunks), 'cent': cent,
            'fps': data['fps'], 'info': data['info']}


def load(dataset):
    """Cached per dataset; re-read when its export changes."""
    p = export_dir(dataset) / 'areas.csv'
    return _load(dataset, p.stat().st_mtime if p.exists() else 0)


def measures(d):
    a = d['areas'].replace(0, np.nan)
    with np.errstate(invalid='ignore'):       # a chromatophore never measured has no value (NaN), not a warning
        rest = a.quantile(0.10)
        peak = a.quantile(0.95)
        med = a.median()
        m = pd.DataFrame({'rest_px': rest, 'peak_px': peak, 'expansion': peak / rest,
                          'activity': (a.quantile(0.95) - a.quantile(0.05)) / med,
                          'time_expanded_pct': 100 * (a > rest * (1 + EXPANDED)).mean()})
    m = m.join(d['cent'][['x_px', 'y_px']], how='inner')
    m.index.name = 'chromatophore'
    return m.replace([np.inf, -np.inf], np.nan)


def correlation(d, min_rest=5):
    """Correlation of the area time series (as CHROMAS's area clustering), for chromatophores big enough to measure."""
    a = d['areas']
    with np.errstate(invalid='ignore'):
        rest = a.replace(0, np.nan).quantile(0.10)
    keep = [c for c in a.columns if rest.get(c, 0) >= min_rest and a[c].std() > 0]
    corr = np.nan_to_num(np.corrcoef(a[keep].fillna(0).to_numpy(), rowvar=False))
    return pd.DataFrame(corr, index=keep, columns=keep)


def affinity_propagation(s, damping=0.5, max_iter=400, conv_iter=20):
    """Affinity propagation (Frey & Dueck 2007) on a similarity matrix s, preference = median, as scikit-learn does.
    Returns labels 0..k-1."""
    n = s.shape[0]
    s = s.copy()
    s[np.diag_indices(n)] = np.median(s)
    s = s + 1e-12 * np.random.default_rng(0).standard_normal(s.shape)       # break ties, as scikit-learn does
    r = np.zeros_like(s)
    a = np.zeros_like(s)
    last, stable = None, 0
    for _ in range(max_iter):
        as_ = a + s
        idx = np.argmax(as_, axis=1)
        first = as_[np.arange(n), idx]
        as_[np.arange(n), idx] = -np.inf
        second = as_.max(axis=1)
        rn = s - first[:, None]
        rn[np.arange(n), idx] = s[np.arange(n), idx] - second
        r = damping * r + (1 - damping) * rn
        rp = np.maximum(r, 0)
        rp[np.diag_indices(n)] = r[np.diag_indices(n)]
        an = rp.sum(axis=0)[None, :] - rp
        dA = np.diag(an).copy()
        an = np.minimum(an, 0)
        an[np.diag_indices(n)] = dA
        a = damping * a + (1 - damping) * an
        ex = np.flatnonzero(np.diag(a) + np.diag(r) > 0)
        key = tuple(ex)
        stable = stable + 1 if key == last else 0
        last = key
        if stable >= conv_iter and len(ex):
            break
    ex = np.flatnonzero(np.diag(a) + np.diag(r) > 0)
    if not len(ex):
        return np.zeros(n, int)
    lab = np.argmax(s[:, ex], axis=1)
    lab[ex] = np.arange(len(ex))
    return lab


def groups(dataset, method):
    """DataFrame chromatophore -> group (and the correlation matrix used), by method."""
    d = load(dataset)
    corr = correlation(d)
    ids = list(corr.index)
    if method == 'CHROMAS (computed by chromas)':
        cl = _read(extras_dir(dataset) / 'clusters_chromas.csv')
        lab = cl.drop_duplicates('chromatophore').set_index('chromatophore')['cluster'].reindex(ids)
        lab = lab.fillna(-1).astype(int).to_numpy()
    elif method == 'Hierarchical (Ward)' and len(ids) > 2:
        z = linkage(squareform(np.clip(1 - corr.to_numpy(), 0, None), checks=False), method='ward')
        lab = fcluster(z, 0.5 * z[:, 2].max(), criterion='distance') - 1
    elif len(ids) > 2:
        lab = affinity_propagation(corr.to_numpy() - 1)
    else:
        lab = np.zeros(len(ids), int)
    return pd.Series(lab, index=ids, name='group'), corr


def group_order(corr, lab):
    """Order for the matrix: by group (biggest first), and within a group by similarity."""
    order = []
    sizes = lab.value_counts()
    for g in sizes.index:
        members = list(lab[lab == g].index)
        if len(members) > 2:
            sub = corr.loc[members, members].to_numpy()
            z = linkage(squareform(np.clip(1 - sub, 0, None), checks=False), method='average')
            members = [members[i] for i in leaves_list(z)]
        order += members
    return order


def _colour(g):
    return '#9a9a9a' if g < 0 else GROUP_COLORS[int(g) % len(GROUP_COLORS)]


def _background(dataset, width=900):
    """overlay.png scaled to `width`, as an hv.RGB in image pixel coordinates (y down), or None."""
    p = export_dir(dataset) / 'overlay.png'
    img = cv2.imread(str(p)) if p.exists() else None
    if img is None:
        return None
    H, Wd = img.shape[:2]
    small = cv2.resize(img, (width, int(H * width / Wd)), interpolation=cv2.INTER_AREA)
    small = (small * 0.45 + 255 * 0.55).astype(np.uint8)
    return hv.RGB(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), bounds=(0, -H, Wd, 0), kdims=['x_px', 'y_plot'])


TOP_N_DEFAULT = 8          # set from chromas_view.TOP_N: how many chromatophores the Reset map button chooses


def _map_size(m, longest):
    """Plot size keeping the image's proportions, with the longer side `longest` pixels."""
    f = _map_frame(m)
    w, h = f['xlim'][1] - f['xlim'][0], f['ylim'][1] - f['ylim'][0]
    k = longest / max(w, h)
    # responsive with the image's proportions, at most `longest` pixels: a fixed size would not grow when the plot
    # is shown full screen (it was drawn as a small cropped corner)
    return {'responsive': True, 'data_aspect': 1, 'max_width': int(max(w * k, 200)) + 120,
            'max_height': int(max(h * k, 200)) + 40}


def _keep_proportions(plot, element):
    """Map hook: ranges that follow the chromatophores and keep x and y at the same scale, also full screen (Bokeh's
    full-screen view otherwise stretches the picture to the screen's shape)."""
    from bokeh.models import DataRange1d, GlyphRenderer
    import plot_tools
    fig = plot.state
    pts = [r for r in fig.renderers if isinstance(r, GlyphRenderer) and type(r.glyph).__name__ == 'Scatter']
    if pts:
        fig.match_aspect = True
        fig.x_range = DataRange1d(renderers=pts[:1], range_padding=0.04)
        fig.y_range = DataRange1d(renderers=pts[:1], range_padding=0.04)
    plot_tools._add_fullscreen(plot, element)


def fit(width, height):
    """Size options of a plot: a fixed width x height.  (Responsive plots with an aspect ratio were laid out with no
    height when their dataset changed while another tab was open: the sections below were drawn over each other.)"""
    return {'width': width, 'height': height}


def wide(obj, **kw):
    # linked_axes=False: Panel would otherwise tie this plot's axes to any other plot on the page whose axis has the
    # same name, including plots of another dataset (a 5 s trace was drawn on a 33 s axis)
    # a fixed width and stretch_width cannot be combined (Panel warns and turns the width into a minimum)
    mode = None if 'width' in kw else 'stretch_width'
    return pn.pane.HoloViews(obj, sizing_mode=mode, linked_axes=False, **kw)


def _map_frame(m):
    """Plot coordinates: x = column, y = -row (so the picture is upright), limits around the chromatophores."""
    pad = 30
    return dict(xlim=(m['x_px'].min() - pad, m['x_px'].max() + pad), ylim=(-m['y_px'].max() - pad, -m['y_px'].min() + pad))


# ── Views ───────────────────────────────────────────────────────────────────────
def map_view(dataset, cells_sel):
    """Map of the chromatophores; clicking or box-selecting sets the chosen chromatophores (cells_sel)."""
    d = load(dataset)
    m = measures(d).reset_index()
    if not len(m):
        return pn.pane.Markdown('*No positions for this dataset (import it again in the Datasets tab).*')
    mm = _read(extras_dir(dataset) / 'motion_markers.csv')
    measure = pn.widgets.Select(name='Colour by', options=list(MEASURES), value='activity (5-95% range / median)',
                                width=260)
    picture = pn.widgets.Checkbox(name='Territory picture behind', value=True)
    show_mm = pn.widgets.Checkbox(name=f'Motion markers ({mm["chromatophore"].nunique() if len(mm) else 0})',
                                  value=False, disabled=not len(mm))
    method = pn.widgets.Select(name='Groups from', options=group_methods(dataset), width=260)
    numbers = pn.widgets.RadioButtonGroup(name='Chromatophore numbers', options=['Off', 'Selected only', 'All'],
                                          value='Off', button_type='primary', button_style='outline', width=330)
    bg = _background(dataset)
    m['y_plot'] = -m['y_px']
    m['size'] = np.clip(np.sqrt(m['rest_px'].fillna(1)) * 0.9 + 4, 5, 26)

    def plot(meas, pic, show_markers, meth, nums, cells):
        col = MEASURES[meas]
        mm_ = m.copy()
        if col == 'group':
            lab, _ = groups(dataset, meth)
            mm_['group'] = mm_['chromatophore'].map(lab).fillna(-1).astype(int)
            mm_['colour'] = mm_['group'].map(_colour)
            pts = hv.Points(mm_, ['x_px', 'y_plot'], ['chromatophore', 'group', 'rest_px', 'activity', 'colour', 'size'])
            pts = pts.opts(color='colour', size='size')
        else:
            pts = hv.Points(mm_, ['x_px', 'y_plot'], ['chromatophore', col, 'rest_px', 'activity', 'expansion',
                                                     'time_expanded_pct', 'size'])
            # equalised colours: a few extreme chromatophores do not wash out the rest
            pts = pts.opts(color=col, cmap='viridis', colorbar=True, cnorm='eq_hist', size='size')
        pts = pts.opts(line_color='black', line_width=0.6, alpha=0.9, tools=['hover', 'tap', 'box_select'],
                       nonselection_alpha=0.35, selection_line_width=2.5, selection_line_color='red')
        sel = hv.streams.Selection1D(source=pts)

        def _picked(index):
            if index:
                _choose(cells_sel, [int(mm_.iloc[i]['chromatophore']) for i in index])
        sel.add_subscriber(_picked)
        layers = [bg] if (pic and bg is not None) else []
        layers.append(pts)
        if show_markers and len(mm):
            mk = m[m['chromatophore'].isin(mm['chromatophore'])]
            layers.append(hv.Points(mk, ['x_px', 'y_plot'], ['chromatophore']).opts(
                marker='x', size=14, color='black', line_width=3))
        layers += _number_layers(m, nums, cells)
        ov = hv.Overlay(layers).opts(**_map_size(m, 640), xaxis=None, yaxis=None, active_tools=['tap'],
                                     hooks=[_keep_proportions],
                                     title='Click a chromatophore, or drag a box (box-select tool), to plot it below',
                                     **_map_frame(m))
        return wide(ov)

    reset_btn = pn.widgets.Button(name='Reset map', button_type='default', width=110)

    def _reset(_):
        # back to the starting choice (the most active chromatophores, as when the dataset opened); the map is drawn
        # again from scratch, so zoom and pan are reset too even when the choice was already that one
        opts = list(cells_sel.options)
        cells_sel.value = opts[:TOP_N_DEFAULT]
        cells_sel.param.trigger('value')
    reset_btn.on_click(_reset)

    return pn.Column(pn.Row(measure, method, pn.Column(pn.Spacer(height=10), picture, show_mm),
                            pn.Column(pn.Spacer(height=22), reset_btn)),
                     pn.Row(pn.pane.Markdown('**Chromatophore numbers:**', margin=(14, 5, 0, 10)), numbers),
                     pn.bind(plot, measure, picture, show_mm, method, numbers, cells_sel))


def _number_layers(m, nums, cells):
    """Numbers on a map, as in the Activation replay: 'Selected only' rings and numbers the chromatophores chosen
    in the list (in the colours of their curves); 'All' also numbers the rest, small and dark."""
    from replay_view import cell_colour
    if nums == 'Off':
        return []
    cells = [int(c) for c in (cells or [])]
    out = []
    span = float(m['x_px'].max() - m['x_px'].min()) or 1.0
    dx = span * 0.012                                     # labels sit just right of the dot
    if nums == 'All':
        rest = m[~m['chromatophore'].isin(cells)].assign(text=lambda d: d['chromatophore'].astype(int).astype(str),
                                                        xl=lambda d: d['x_px'] + dx)
        out.append(hv.Labels(rest, ['xl', 'y_plot'], 'text').opts(text_font_size='8pt', text_color='#222222',
                                                                 text_align='left', background_fill_color='white',
                                                                 background_fill_alpha=0.6))
    sel = m[m['chromatophore'].isin(cells)].copy()
    if len(sel):
        sel['colour'] = [cell_colour(cells, c) for c in sel['chromatophore']]
        sel['text'] = '#' + sel['chromatophore'].astype(int).astype(str)
        sel['xl'] = sel['x_px'] + dx * 1.8
        out.append(hv.Points(sel, ['x_px', 'y_plot'], ['colour', 'chromatophore']).opts(
            color='colour', fill_alpha=0, size=24, line_width=3, marker='circle'))
        out.append(hv.Labels(sel, ['xl', 'y_plot'], ['text', 'colour']).opts(
            text_color='colour', text_font_size='11pt', text_font_style='bold', text_align='left',
            background_fill_color='white', background_fill_alpha=0.85))
    return out


def _choose(cells_sel, ids):
    """Set the chosen chromatophores (only ones the list offers, at most 10)."""
    allowed = set(cells_sel.options if isinstance(cells_sel.options, list) else list(cells_sel.options))
    ids = [c for c in ids if c in allowed][:10]
    if ids:
        cells_sel.value = ids


def group_methods(dataset):
    opts = ['Affinity propagation (as CHROMAS)', 'Hierarchical (Ward)']
    if (extras_dir(dataset) / 'clusters_chromas.csv').exists():
        opts.insert(0, 'CHROMAS (computed by chromas)')
    return opts


def distributions_view(dataset):
    m = measures(load(dataset))
    if not len(m):
        return pn.pane.Markdown('')

    def hist(col, label, log=False):
        v = m[col].dropna()
        v = v[v > 0] if log else v
        if not len(v):
            return hv.Div('')
        vals = np.log10(v) if log else v
        edges = np.histogram_bin_edges(vals, bins=min(30, max(8, len(v) // 4)))
        h = hv.Histogram(np.histogram(vals, edges), kdims=[f'log10 {label}' if log else label], vdims=['chromatophores'])
        return h.opts(**fit(320, 240), color='#4c72b0', line_color='white', tools=['hover'])

    return pn.Column(
        pn.pane.Markdown(f'Resting size = 10th percentile of the area; largest expansion = 95th percentile / resting '
                         f'size; expanded = area more than {EXPANDED:.0%} above resting size. Two humps in the size '
                         'histogram usually mean two populations, e.g. young and mature chromatophores.', width=W),
        pn.Row(wide(hist('rest_px', 'resting size (px)', log=True)), wide(hist('expansion', 'largest expansion (x rest)')),
               wide(hist('time_expanded_pct', 'time expanded (%)')), width=W))


def groups_view(dataset, cells_sel):
    method = pn.widgets.Select(name='Groups from', options=group_methods(dataset), width=280)

    def view(meth):
        lab, corr = groups(dataset, meth)
        if len(corr) < 3:
            return pn.pane.Markdown('*Too few measurable chromatophores to group.*')
        order = group_order(corr, lab)
        n = len(order)
        mat = corr.loc[order, order].to_numpy()
        img = hv.Image(np.flipud(mat), bounds=(0, 0, n, n), kdims=['chromatophore (column)', 'chromatophore (row)'],
                       vdims=['correlation']).opts(cmap='RdBu_r', clim=(-1, 1), colorbar=True, **fit(520, 470), active_tools=['pan'],
                                                    xaxis=None, yaxis=None, tools=['hover'],
                                                    title='Correlation of the area time series (ordered by group)')
        # group borders on the matrix
        bounds, pos = [], 0
        for g in pd.unique(lab.loc[order]):
            k = int((lab.loc[order] == g).sum())
            if k > 1:
                bounds.append(hv.Bounds((pos, n - pos - k, pos + k, n - pos)).opts(color='black', line_width=1.5))
            pos += k
        matrix = hv.Overlay([img] + bounds)

        m = measures(load(dataset))
        m = m.loc[[c for c in order if c in m.index]].copy()
        m['group'] = lab.reindex(m.index).fillna(-1).astype(int)
        m['y_plot'] = -m['y_px']
        sizes = m['group'].value_counts()
        multi = sizes[sizes > 1].index
        m['colour'] = [(_colour(g) if g in multi and g >= 0 else '#c8c8c8') for g in m['group']]
        lines = []
        for g in multi:
            if g < 0:
                continue
            mem = m[m['group'] == g]
            sub = corr.loc[mem.index, mem.index].to_numpy()
            centre = mem.index[int(np.argmax(sub.mean(axis=1)))]        # the most typical member
            cx, cy = m.loc[centre, ['x_px', 'y_plot']]
            for c, r in mem.iterrows():
                if c != centre:
                    lines.append(hv.Curve([(cx, cy), (r['x_px'], r['y_plot'])]).opts(color=_colour(g), line_width=1.5,
                                                                                     alpha=0.8))
        pts = hv.Points(m.reset_index(), ['x_px', 'y_plot'], ['chromatophore', 'group', 'colour']).opts(
            color='colour', size=9, line_color='black', line_width=0.5, tools=['hover', 'tap'])
        sel = hv.streams.Selection1D(source=pts)
        rows = m.reset_index()

        def _picked(index):
            if index:
                g = rows.iloc[index[0]]['group']
                _choose(cells_sel, [int(c) for c in rows[rows['group'] == g]['chromatophore']])
        sel.add_subscriber(_picked)
        gmap = hv.Overlay(lines + [pts]).opts(**_map_size(m, 440), xaxis=None, yaxis=None, **_map_frame(m),
                                              hooks=[_keep_proportions],
                                              active_tools=['tap'],
                                              title='Groups on the skin (lines join each member to the most typical one)')
        summary = []
        for g in multi:
            if g < 0:
                continue
            mem = list(m[m['group'] == g].index)
            sub = corr.loc[mem, mem].to_numpy()
            within = (sub.sum() - len(mem)) / max(len(mem) * (len(mem) - 1), 1)
            xy = m.loc[mem, ['x_px', 'y_px']].to_numpy()
            summary.append({'group': int(g), 'chromatophores': len(mem), 'mean correlation within': round(within, 3),
                            'spread (px)': round(float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean()), 1),
                            'members': ' '.join(map(str, mem[:20])) + (' ...' if len(mem) > 20 else '')})
        tab = pd.DataFrame(summary).sort_values('chromatophores', ascending=False) if summary else pd.DataFrame()
        n_single = int((sizes == 1).sum())
        txt = (f'**{len(multi)} groups** of two or more chromatophores, **{n_single}** chromatophores on their own '
               f'(grey). {n} chromatophores were large enough to measure. Click a point on the right to plot its whole '
               'group above. Groups whose members lie far apart are worth a second look: CHROMAS\'s authors found that '
               'one motor unit can include distant chromatophores, but a group can also be two units that happened to '
               'fire together.')
        return pn.Column(pn.pane.Markdown(txt, width=W), pn.Row(wide(matrix), wide(gmap), width=W),
                         pn.widgets.Tabulator(tab, show_index=False, disabled=True, theme='midnight', width=W,
                                              height=min(320, 40 + 30 * len(tab))) if len(tab) else pn.Spacer())

    return pn.Column(method, pn.bind(view, method))


MAX_FLOWERS = 12          # flower plots drawn at once in the Slices panel


def slices_view(dataset, cells_sel):
    summ = _read(extras_dir(dataset) / 'slices_summary.csv')
    if not len(summ):
        return pn.pane.Markdown(
            '*No slice data for this dataset.* CHROMAS\'s `slice` step cuts each chromatophore into 36 wedges and follows '
            'the area of each. Run it in the **Datasets** tab (tick the dataset, then **Run slice step**); it needs the '
            'CHROMAS result (.zarr) and works best when a fully expanded chromatophore is at least 50 pixels across.',
            width=W)
    summ = summ.drop_duplicates(['chromatophore', 'slice'])
    piv = summ.pivot(index='chromatophore', columns='slice', values='expansion')
    stats = pd.DataFrame({'mean expansion': piv.mean(axis=1),
                          'unevenness': (piv.max(axis=1) - piv.min(axis=1)) / piv.mean(axis=1).replace(0, np.nan)})
    stats = stats.dropna().sort_values('unevenness', ascending=False)
    n_sl = piv.shape[1]
    ang = np.deg2rad(np.arange(n_sl) * 360 / n_sl)

    def flower(c, colour='#4c72b0'):
        e = piv.loc[c].to_numpy(float)
        e = np.nan_to_num(e)
        top = max(e.max(), 1e-6)
        r = e / top
        ring = [hv.Curve((np.cos(np.linspace(0, 2 * np.pi, 60)) * k, np.sin(np.linspace(0, 2 * np.pi, 60)) * k),
                         'petal_x', 'petal_y').opts(color='#dddddd', line_width=1) for k in (0.5, 1.0)]
        xs, ys = [], []
        for k in range(n_sl):
            a0, a1 = ang[k] - np.pi / n_sl, ang[k] + np.pi / n_sl
            xs += [0, r[k] * np.cos(a0), r[k] * np.cos(a1), 0]
            ys += [0, -r[k] * np.sin(a0), -r[k] * np.sin(a1), 0]          # image y points down
        poly = hv.Polygons([{'petal_x': xs, 'petal_y': ys}], kdims=['petal_x', 'petal_y']).opts(color=colour, alpha=0.7, line_color='black', line_width=0.5)
        u = stats.loc[c, 'unevenness'] if c in stats.index else np.nan
        k = int(piv.loc[c].notna().sum())          # wedges with a rest area above 0 (the others have no expansion)
        title = (f'#{c}: unevenness {u:.2f}' + (f' ({k} of {n_sl} wedges)' if k < n_sl else '')
                 if np.isfinite(u) else f'#{c}: too small to slice')
        return hv.Overlay(ring + [poly]).opts(width=230, height=230, xaxis=None, yaxis=None, xlim=(-1.1, 1.1),
                                              ylim=(-1.1, 1.1), title=title, toolbar=None)

    def view(cells):
        from replay_view import cell_colour
        chosen = [c for c in (cells or []) if c in piv.index]
        if not chosen:
            chosen = list(stats.index[:6])
            note = 'None of the chromatophores chosen above has slice data; showing the six most uneven.'
        else:
            note = 'The chromatophores chosen above, in the colours of their curves.'
        if len(chosen) > MAX_FLOWERS:
            note += f' Showing the first {MAX_FLOWERS} of the {len(chosen)} chosen; untick some to see the others.'
        flowers = [flower(c, cell_colour(cells, c) if c in (cells or []) else '#4c72b0') for c in chosen[:MAX_FLOWERS]]
        table = stats.round(3).reset_index().rename(columns={'chromatophore': 'chromatophore'})
        # linked_axes=False: Panel would otherwise tie their axes to other plots on the page with the same names
        rows = [pn.Row(*[pn.pane.HoloViews(f, linked_axes=False) for f in flowers[i:i + 4]])
                for i in range(0, len(flowers), 4)]
        return pn.Column(pn.pane.Markdown(note), *rows,
                         pn.pane.Markdown('**All sliced chromatophores, most uneven first**'),
                         pn.widgets.Tabulator(table, show_index=False, disabled=True, theme='midnight', width=420,
                                              height=260))

    return pn.Column(pn.pane.Markdown(
        'Each petal is one of the 36 wedges of a chromatophore; its length is how much that wedge expands (95th / 10th '
        'percentile of its area), scaled to the largest wedge. A round flower expands evenly; a lopsided one is pulled '
        'from one side, which can mean more than one nerve feeds it. **Unevenness** = (largest - smallest wedge '
        'expansion) / mean. Wedge 0 points right in the image.', width=W), pn.bind(view, cells_sel))


def focus_view(dataset):
    f = _read(export_dir(dataset) / 'focus.csv')
    info = load(dataset)['info']
    thr = (info.get('attrs') or {}).get('focus_threshold')
    if not len(f):
        return pn.pane.Markdown('*No focus check in this dataset.*')
    f = f.copy()
    f['in_focus'] = f['in_focus'].astype(str).str.lower().isin(['true', '1'])
    dropped = 100 * (~f['in_focus']).mean()
    spans, start = [], None
    for t, ok in zip(f['t_s'], f['in_focus']):
        if not ok and start is None:
            start = t
        if ok and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, float(f['t_s'].iloc[-1])))
    layers = [hv.VSpan(a, b).opts(color='#e45756', alpha=0.25, line_alpha=0) for a, b in spans]
    if f['statistic'].notna().any():
        layers.append(hv.Curve(f, hv.Dimension('t_s', label='time (s)'), [hv.Dimension('statistic', label='focus score'), 'in_focus']).opts(color='#1f77b4', line_width=1.5, tools=['hover'], hover_mode='vline'))
        if thr is not None:
            layers.append(hv.HLine(float(thr)).opts(color='black', line_dash='dashed', line_width=1.5))
    else:
        layers.append(hv.Curve(f.assign(v=f['in_focus'].astype(int)), 't_s', 'v').opts(color='#1f77b4', tools=['hover'], hover_mode='vline'))
    plot = hv.Overlay(layers).opts(**fit(W, 230), xlabel='time in the video (s)', ylabel='focus score',
                                   tools=['hover'], title='Focus check (red = frames dropped as out of focus)')
    return pn.Column(pn.pane.Markdown(
        f'CHROMAS scores the sharpness of every frame; frames below the threshold (dashed, {thr}) are dropped, which '
        f'is why traces have gaps there. **{dropped:.1f}%** of the frames were dropped.', width=W),
        wide(plot, width=W, height=230))


def coverage_view(dataset):
    ex = extras_dir(dataset)
    by = _read(ex / 'coverage_by_frame.csv')
    items = []
    row_h = 0                    # a Row does not reserve the height of an image in it: say it (else the next heading is drawn over it)
    if (ex / 'coverage.png').exists():
        q = cv2.imread(str(ex / 'coverage.png'), cv2.IMREAD_GRAYSCALE)
        col = cv2.applyColorMap(q, cv2.COLORMAP_VIRIDIS)
        scale = min(1.0, 520 / q.shape[1])
        col = cv2.resize(col, (int(q.shape[1] * scale), int(q.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        row_h = max(row_h, col.shape[0] + 50)
        items.append(pn.Column(pn.pane.Markdown('**Share of frames each pixel was pigmented** (dark = never, yellow = '
                                                'always)'), pn.pane.PNG(cv2.imencode('.png', col)[1].tobytes(),
                                                                         width=col.shape[1], height=col.shape[0])))
    if len(by):
        mean = by['coverage_pct'].mean()
        lay = [hv.Curve(by.reset_index(), 'index', [hv.Dimension('coverage_pct', label='coverage (%)'), 'chunk', 'frame']).opts(color='#1f77b4',
                                                                                           line_width=1.5, tools=['hover'], hover_mode='vline'),
               hv.HLine(75).opts(color='#e45756', line_dash='dashed'), hv.HLine(5).opts(color='#e45756', line_dash='dotted')]
        plot = hv.Overlay(lay).opts(**fit(430, 260), xlabel='frame', ylabel='% of the frame segmented', tools=['hover'],
                                    ylim=(0, max(80, float(by['coverage_pct'].max()) * 1.1)),
                                    title=f'Segmentation coverage per frame (mean {mean:.1f}%)')
        warn = ''
        if mean > 75:
            warn = ' **Over 75%: the segmentation has probably taken the background for pigment.**'
        elif mean < 5:
            warn = ' **Under 5%: sparse skin; registration needs the sparse-skin patch.**'
        row_h = max(row_h, 260 + 80)
        items.append(pn.Column(wide(plot, width=430, height=260), pn.pane.Markdown(
            f'Dashed line: 75%, above which the lookup method has usually gone wrong; dotted: 5%, below which CHROMAS '
            f'struggles to register.{warn}', width=430)))
    if not items:
        return _no_extras(dataset, 'coverage')
    return pn.Row(*items, height=row_h) if row_h else pn.Row(*items)


TRACK_MIN_POINTS = 40      # CHROMAS's own default for the fewest points it wants to follow
TRACK_P99_STEP_PX = 5.0    # placeholder: half of CHROMAS's 10 px limit


def tracking_pane(dataset):
    """The points CHROMAS followed inside each chunk (quality check 5)."""
    t = _read(extras_dir(dataset) / 'tracking_points.csv')
    if not len(t):
        return pn.pane.Markdown('*Tracking points are not in the saved quality checks. In the **Datasets** tab press '
                                '**Export quality checks** for this dataset to add them.*', width=W)
    t = t.copy()
    t['flag'] = np.where(t['n_points'] < TRACK_MIN_POINTS, 'few points',
                         np.where(t['step_p99_px'] > TRACK_P99_STEP_PX, 'large steps', 'ok'))
    t = t[['chunk', 'flag'] + [c for c in t.columns if c not in ('chunk', 'flag')]]
    bad = t[t['flag'] != 'ok']
    note = (f'**{len(bad)} of {len(t)} chunk(s) flagged.** ' if len(bad) else 'No chunk flagged. ')
    return pn.Column(
        pn.pane.Markdown('**Tracking inside each chunk.** CHROMAS follows points on the skin from frame to frame and '
                         'drops any point that moves 10 px or more in one frame, so the points listed are the ones it '
                         f'kept. {note}Flags (placeholders): fewer than {TRACK_MIN_POINTS} points kept, or the 99th '
                         f'percentile of the frame-to-frame step above {TRACK_P99_STEP_PX:g} px.', width=W),
        pn.widgets.Tabulator(t.round(3), show_index=False, disabled=True, theme='midnight', height=40 + 30 * len(t),
                             width=860))


def registration_view(dataset):
    return pn.Column(tracking_pane(dataset), _registration_between_chunks(dataset))


def _registration_between_chunks(dataset):
    r = _read(extras_dir(dataset) / 'registration_error.csv')
    if not len(r):
        info = load(dataset)['info']
        if int(info.get('n_chunks', 1)) < 2:
            return pn.pane.Markdown('*One chunk only: there is nothing to register it against.*')
        return _no_extras(dataset, 'registration')
    n = int(max(r['src_chunk'].max(), r['trg_chunk'].max())) + 1
    m = r.pivot(index='src_chunk', columns='trg_chunk', values='value').reindex(index=range(n), columns=range(n))
    hm = hv.HeatMap(r, ['trg_chunk', 'src_chunk'], 'value').opts(
        cmap='RdYlGn', clim=(0, 1), colorbar=True, **fit(420, 360), tools=['hover'], invert_yaxis=True,
        xlabel='to chunk', ylabel='from chunk', title='Share of the overlap that maps back within 3 px')
    off = m.where(~np.eye(n, dtype=bool))
    per = pd.DataFrame({'chunk': range(n), 'mean': off.mean(axis=1).to_numpy()})
    bars = hv.Bars(per, 'chunk', 'mean').opts(**fit(420, 360), ylim=(0, 1), color='#4c72b0', tools=['hover'],
                                              title='Mean agreement with the other chunks')
    bad = per[per['mean'] < 0.5]['chunk'].tolist()
    note = (f'**Chunks {", ".join(map(str, bad))}** agree poorly with the rest: chromatophore numbers there may not '
            'match the other chunks.' if bad else 'All chunks agree with each other (0.5 or more).')
    return pn.Column(pn.pane.Markdown('CHROMAS maps every chunk onto every other and back; the share of the overlap that '
                                      f'lands within 3 pixels of where it started is shown. {note}', width=W),
                     pn.Row(wide(hm, height=360), wide(bars, height=360), width=860))


def segmentation_view(dataset):
    sd = extras_dir(dataset) / 'segmentation'
    masks = sorted(sd.glob('*_mask.png')) if sd.exists() else []
    if not masks:
        return _no_extras(dataset, 'segmentation')
    names = {f"chunk {p.stem.split('_')[0]}, frame {int(p.stem.split('_')[1])}": p for p in masks}
    pick = pn.widgets.DiscreteSlider(name='Sample frame', options=list(names), value=list(names)[0], width=500)
    mode = pn.widgets.RadioButtonGroup(options=['Side by side', 'Overlay'], value='Side by side', button_type='primary',
                                       button_style='outline')

    def show(key, how):
        mp = names[key]
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        ip = mp.with_name(mp.name.replace('_mask.png', '_image.jpg'))
        img = cv2.imread(str(ip)) if ip.exists() else None
        lab = np.zeros((*mask.shape, 3), np.uint8)
        lab[mask > 0] = (40, 40, 220)
        cov = 100 * float((mask > 0).mean())
        if img is None:
            out = np.where(mask[..., None] > 0, lab, 255).astype(np.uint8)
        elif how == 'Overlay':
            out = img.copy()
            out[mask > 0] = (0.45 * img[mask > 0] + 0.55 * np.array((40, 40, 220))).astype(np.uint8)
            cnt, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnt, -1, (0, 220, 255), 1)
        else:
            right = np.where(mask[..., None] > 0, lab, 255).astype(np.uint8)
            out = np.hstack([img, np.full((img.shape[0], 8, 3), 255, np.uint8), right])
        width = min(out.shape[1], 1000)
        return pn.Column(pn.pane.Markdown(f'{cov:.1f}% of this frame segmented as chromatophore (red).'),
                         pn.pane.JPG(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), width=width,
                                     height=int(round(out.shape[0] * width / out.shape[1]))))

    return pn.Column(pn.pane.Markdown(
        'The video frame next to what CHROMAS segmented as chromatophore. Look for pigment it missed (false negatives), '
        'and for debris, the electrode or shadows it took for pigment. Scores against hand-made masks: '
        '`chromas_score_segmentation.py` (docs/TRAINING_THE_NETWORK.md).', width=W),
        pn.Row(pick, mode), pn.bind(show, pick, mode))


def video_view(dataset):
    """Play the quality-control videos made in the Datasets tab (chromas generate overlay-video / cleanqueen-video)."""
    import chromas_bridge as cb

    def listing():
        out = {}
        for p in cb.list_videos(dataset):
            kind, _, chunk = p.stem[len('video_'):].rpartition('_chunk')
            what = {'segmentation': 'Segmentation overlay', 'territories': 'Territories'}.get(kind, kind)
            out[f'{what}, chunk {chunk} ({p.stat().st_size / 1e6:.1f} MB)'] = str(p)
        return out

    options = listing()
    sel = pn.widgets.Select(name='Video', options=options, value=next(iter(options.values()), None), width=420)
    refresh = pn.widgets.Button(name='Refresh list', width=110)

    def _refresh(*_):
        opts = listing()
        sel.options = opts
        if sel.value not in opts.values():
            sel.value = next(iter(opts.values()), None)
    refresh.on_click(_refresh)

    def play(path):
        if not path:
            return pn.pane.Markdown(
                '*No video yet.* In the **Datasets** tab tick this dataset (it needs its CHROMAS result, `.zarr`), choose '
                '**segmentation** or **territories**, and press **Make video**. It is the best check that CHROMAS found the '
                'pigment and not debris or shadows: the segmentation is drawn on the footage, so missed pigment and '
                'false detections show up directly. Press **Refresh list** here when it finishes.', width=W)
        return pn.pane.Video(path, width=min(W, 900), loop=False)

    return pn.Column(pn.Row(sel, pn.Column(pn.Spacer(height=22), refresh)), pn.bind(play, sel))


def _no_extras(dataset, what):
    ex = extras_dir(dataset) / 'extras.json'
    note = ''
    if ex.exists():
        note = json.loads(ex.read_text(encoding='utf-8')).get('notes', {}).get(what, '')
    return pn.pane.Markdown(
        (f'*Not available: {note}.*' if note else
         '*Not exported for this dataset.* In the **Datasets** tab, tick it and press **Export quality checks** (needs '
         'the CHROMAS result, `.zarr`).'), width=W)
