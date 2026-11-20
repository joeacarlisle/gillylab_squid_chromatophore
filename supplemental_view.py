"""The Advanced analytics tab (was "Supplemental"): reports modelled on the figures of the CHROMAS papers (Ukrow et al. 2025, and the motor-unit
paper that follows it), for the dataset chosen in the sidebar.  The papers' notebooks were not available, so these
are built from the figures and from what the exported CHROMAS results contain.

    Slice traces        the 36 wedge areas of one chromatophore over time, stacked, and as a heat map
    Motor units         the papers' own method: PCA then ICA on each chromatophore's wedges (CHROMAS `ica`), and
                        clustering of the independent components (CHROMAS `analyse cluster ics`); read from
                        extras/motor_unit_*.csv
    Correlation clusters wedges (slices) that vary together, found here by hierarchical clustering of their
                        correlation, drawn on the skin; a table of the clusters.  A shortcut of this project, not the
                        papers' method: do not report these as motor units
    Cluster statistics  sizes, hull areas, and how many clusters each chromatophore belongs to
    Cluster network     chromatophores of one cluster joined by lines (shortest tree through their positions)
    Slice orientation   do the wedges of a cluster point towards the cluster's centre?
    Area clusters       CHROMAS's own grouping of whole chromatophores (affinity propagation), as hulls on the skin
    Expansion speed     fastest expansion against fastest relaxation of each chromatophore
    Chunk overlap       which chromatophores two chunks have in common (datasets with two or more chunks)

The motor-unit report uses CHROMAS's own PCA/ICA and clustering (run in the CHROMAS environment, which has
scikit-learn).  The correlation clusters below it are this project's shortcut: scikit-learn is not a dependency of
this project, so there the wedges are clustered with scipy (average linkage on 1 - correlation, cut at a chosen
correlation).

Wedge directions: CHROMAS measures the wedges of a chromatophore from its own orientation angle (the direction to its
motion marker), not from the image's x axis.  The export stores that angle (extras/slices_<chunk>.npz, 'orientation');
wedge j then points at 265 - orientation - 10 * j degrees in image coordinates (x = column, y = row), as CHROMAS draws
it.  Exports made before the angle was stored have no 'orientation': positions and the orientation report then say so.

Reads processed/chromas_export/<dataset>/ (areas, centroids, extras/slices_<chunk>.npz written by the CHROMAS
`slice` step).  Plots have fixed or capped widths, as in chromatophore_views.py.
"""
import contextlib
import functools
import time
import warnings

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.palettes import Turbo256
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import binary_dilation
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform

import chromatophore_views as cv

hv.extension('bokeh')
import plot_tools  # noqa: E402
plot_tools.enable()
pn.extension('tabulator')

W = cv.W
S_DEFAULT = 36                 # wedges per chromatophore (CHROMAS `slice` step)
MAX_SLICES = 7000              # clustering is O(n^2) in memory; above this the user is asked to drop the tiny wedges
NOISE = '#9a9a9a'
CHUNK_COLOURS = {'both': '#000000', 'a': '#d62728', 'b': '#1f77b4'}


# ── Data ────────────────────────────────────────────────────────────────────────
def _stamp(dataset):
    ex = cv.extras_dir(dataset)
    files = list(ex.glob('slices_*.npz')) if ex.exists() else []
    return max((p.stat().st_mtime for p in files), default=0.0)


@functools.lru_cache(maxsize=4)
def _slices(dataset, stamp):
    """{chunk: (ids, areas frames x cells x wedges, orientation per cell in degrees or None)} from
    extras/slices_<chunk>.npz."""
    out = {}
    for p in sorted(cv.extras_dir(dataset).glob('slices_*.npz')):
        try:
            chunk = int(p.stem.split('_')[1])
            z = np.load(p)
            out[chunk] = (z['ids'].astype(int), z['slice_areas'].astype(np.float32),
                          z['orientation'].astype(float) if 'orientation' in z.files else None)
        except (ValueError, IndexError, KeyError, OSError):
            continue
    return out


def slices(dataset):
    return _slices(dataset, _stamp(dataset))


def _times(dataset, chunk, n):
    d = cv.load(dataset)
    t = d['t'][d['chunk'] == chunk]
    return t if len(t) == n else np.arange(n) / d['fps']


def _cell_radius(arr):
    """Rough radius (px) of each chromatophore: from its median total wedge area."""
    return np.maximum(np.sqrt(np.median(arr.sum(axis=2), axis=0) / np.pi), 2.0)


def _no_slices():
    return pn.pane.Markdown(
        '*No slice data for this dataset.* CHROMAS\'s `slice` step cuts each chromatophore into 36 wedges and follows '
        'the area of each; these reports need it. Run it in the **Datasets** tab (tick the dataset, then **Run slice '
        'step**).', width=W)


def _empty(msg):
    return pn.pane.Markdown(msg, width=W)


def _directions(cells, wedge, ids, orient, S=S_DEFAULT):
    """Direction (degrees, image coordinates: x = column, y = row) each wedge points in, as CHROMAS draws it."""
    oa = dict(zip(ids, orient)) if orient is not None else {}
    o = np.array([oa.get(int(c), 0.0) for c in cells], float)
    step = 360.0 / S
    return np.mod(265.0 - o - step * np.asarray(wedge, float), 360.0)


def has_orientation(dataset, chunk):
    sl = slices(dataset)
    return chunk in sl and sl[chunk][2] is not None


# ── Clustering of the wedges ──────────────────────────────────────────────────────
@functools.lru_cache(maxsize=12)
def _cluster(dataset, stamp, chunk, thr, min_size, min_rest):
    """Cluster the wedges of one chunk.  Returns (wedges, clusters, cells):
    wedges   one row per wedge kept: cell, slice, cluster (-1 = in no cluster), x_px, y_px (position on the skin)
    clusters one row per cluster: n_slices, n_cells, hull_px2, elongation, shape, mean_corr
    cells    chromatophore -> (x_px, y_px), the ones with slice data
    """
    ids, arr, orient = _slices(dataset, stamp)[chunk]
    F, N, S = arr.shape
    d = cv.load(dataset)
    cent = d['cent']
    X = arr.reshape(F, N * S)
    rest = np.percentile(X, 10, axis=0)
    sd = X.std(axis=0)
    keep = np.where((rest >= min_rest) & (sd > 1e-3))[0]
    cell_pos = pd.DataFrame({'chromatophore': ids}).join(cent[['x_px', 'y_px']], on='chromatophore')
    cell_pos = cell_pos.dropna().set_index('chromatophore')
    if len(keep) < 3:
        return pd.DataFrame(), pd.DataFrame(), cell_pos
    if len(keep) > MAX_SLICES:
        raise ValueError(f'{len(keep)} wedges to cluster is too many; raise "Ignore wedges smaller than" above '
                         f'{min_rest:g} px so fewer remain')
    Z = X[:, keep].astype(np.float64)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    corr = np.clip(Z.T @ Z / F, -1, 1)
    dist = (1 - corr + (1 - corr).T) / 2
    np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), method='average'), 1 - thr, criterion='distance')
    sizes = pd.Series(lab).value_counts()
    order = [g for g in sizes.index if sizes[g] >= min_size]
    remap = {g: i for i, g in enumerate(order)}
    cluster = np.array([remap.get(g, -1) for g in lab])

    cell_of = ids[keep // S]
    slice_of = keep % S
    rad = dict(zip(ids, _cell_radius(arr) * 0.9))
    wedges = pd.DataFrame({'cell': cell_of, 'slice': slice_of, 'cluster': cluster})
    wedges = wedges.join(cell_pos, on='cell').dropna(subset=['x_px', 'y_px']).reset_index(drop=True)
    wedges['direction'] = _directions(wedges['cell'].to_numpy(), wedges['slice'].to_numpy(), ids, orient, S)
    theta = np.deg2rad(wedges['direction'].to_numpy())
    r = wedges['cell'].map(rad).to_numpy()
    wedges['wx'] = wedges['x_px'] + r * np.cos(theta)
    wedges['wy'] = wedges['y_px'] + r * np.sin(theta)           # image y points down

    rows = []
    for g in sorted(set(cluster) - {-1}):
        w = wedges[wedges['cluster'] == g]
        cells = w['cell'].unique()
        pos = cell_pos.loc[cells, ['x_px', 'y_px']].to_numpy(float)
        hull = np.nan
        elong = np.nan
        if len(cells) >= 3:
            try:
                hull = float(ConvexHull(pos).volume)           # 2-D: .volume is the area
            except Exception:                                   # collinear
                hull = 0.0
            ev = np.sort(np.linalg.eigvalsh(np.cov(pos.T)))[::-1]
            elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
        rows.append({'cluster': g, 'n_slices': len(w), 'n_cells': len(cells), 'hull_px2': hull, 'elongation': elong,
                     'shape': ('single chromatophore' if len(cells) < 2 else 'pair' if len(cells) == 2 else
                               'elongated' if elong >= 3 else 'compact')})
    clusters = pd.DataFrame(rows)
    # mean pairwise correlation of the wedges of each cluster
    mc = []
    for g in clusters['cluster']:
        members = np.where(cluster == g)[0]
        sub = corr[np.ix_(members, members)]
        n = len(members)
        mc.append(float((sub.sum() - n) / max(n * (n - 1), 1)))
    clusters['mean_corr'] = mc
    return wedges, clusters, cell_pos


def cluster(dataset, chunk, thr, min_size, min_rest):
    return _cluster(dataset, _stamp(dataset), int(chunk), round(float(thr), 3), int(min_size), float(min_rest))


# ── Map helpers ─────────────────────────────────────────────────────────────────
def _map(dataset, layers, pos, title):
    """Overlay of layers on the territory picture, sized and ranged like the Chromatophores map."""
    bg = cv._background(dataset)
    ov = hv.Overlay(([bg] if bg is not None else []) + layers).opts(
        **cv._map_size(pos, 800), xaxis=None, yaxis=None, hooks=[cv._keep_proportions], title=title,
        **cv._map_frame(pos))
    return cv.wide(ov)


def _cells_layer(pos):
    p = pos.reset_index()
    p['y_plot'] = -p['y_px']
    return hv.Points(p, ['x_px', 'y_plot'], ['chromatophore']).opts(
        size=4, color='white', line_color='black', line_width=0.7, tools=['hover'])


def _colour(g):
    return cv._colour(g)


# ── 1. Slice traces ─────────────────────────────────────────────────────────────
def traces_view(dataset, cell, spacing):
    sl = slices(dataset)
    if not sl:
        return _no_slices()
    chunk = next((c for c, v in sl.items() if cell in v[0]), None)
    if chunk is None:
        return _empty('*Choose a chromatophore that has slice data.*')
    ids, arr, _ = sl[chunk]
    k = int(np.where(ids == cell)[0][0])
    a = arr[:, k, :].astype(float)                                 # frames x 36
    t = _times(dataset, chunk, len(a))
    S = a.shape[1]
    curves = []
    for s in range(S):
        stacked = a[:, s] - a[:, s].mean() + s * spacing
        df = pd.DataFrame({'t_s': t, 'stacked': stacked, 'wedge': s, 'area': a[:, s]})
        curves.append(hv.Curve(df, hv.Dimension('t_s', label='time (s)'),
                               [hv.Dimension('stacked', label='area, offset per wedge'), 'wedge',
                                hv.Dimension('area', label='area (px)')]).opts(
            color=Turbo256[int(s * 255 / max(S - 1, 1))], line_width=1.1, tools=['hover'], hover_mode='mouse'))
    stack = hv.Overlay(curves).opts(height=520, responsive=True, show_legend=False, xlabel='time in the video (s)',
                                    ylabel=f'wedge area, offset {spacing:g} px per wedge (wedge 0 at the bottom)',
                                    title=f'Chromatophore #{cell}: area of each of its {S} wedges')
    rel = a / np.maximum(np.median(a, axis=0), 1e-6) - 1
    lim = float(np.nanpercentile(np.abs(rel), 98)) or 1.0
    img = hv.Image(np.flipud(rel.T), bounds=(float(t.min()), 0, float(t.max()), S),
                   kdims=[hv.Dimension('t_s', label='time (s)'), hv.Dimension('wedge_pos', label='wedge')],
                   vdims=[hv.Dimension('rel', label='relative change')]).opts(
        cmap='RdBu_r', clim=(-lim, lim), colorbar=True, height=300, responsive=True, tools=['hover'],
        title='Same wedges as a heat map (relative change from each wedge\'s median; wedge 0 at the bottom)')
    return pn.Column(cv.wide(stack), cv.wide(img), sizing_mode='stretch_width')


# ── 2. Clusters on the skin and the table ────────────────────────────────────────
def clusters_view(dataset, chunk, thr, min_size, min_rest):
    if not slices(dataset) or chunk is None:
        return _no_slices()
    wedges, clusters, pos = cluster(dataset, chunk, thr, min_size, min_rest)
    if not len(wedges):
        return _empty('*Too few measurable wedges. Lower "Ignore wedges smaller than".*')
    w = wedges.drop(columns=['x_px', 'y_px']).rename(columns={'wx': 'x_px'})       # a wedge's own position
    w['y_plot'] = -w['wy']
    w['colour'] = w['cluster'].map(_colour)
    inn, out = w[w['cluster'] >= 0], w[w['cluster'] < 0]
    layers = [_cells_layer(pos)]
    if len(out):
        layers.append(hv.Points(out, ['x_px', 'y_plot'], ['cell', 'slice', 'cluster']).opts(
            size=4, color=NOISE, alpha=0.45, tools=['hover']))
    if len(inn):
        layers.append(hv.Points(inn, ['x_px', 'y_plot'], ['cell', 'slice', 'cluster', 'colour']).opts(
            size=6, color='colour', line_color='black', line_width=0.4, alpha=0.9, tools=['hover']))
    n_cells = wedges['cell'].nunique()
    multi = (wedges[wedges['cluster'] >= 0].groupby('cell')['cluster'].nunique() > 1).sum() if len(inn) else 0
    head = pn.pane.Markdown(
        f'**{len(clusters)} clusters** from {len(wedges)} wedges of {n_cells} chromatophores; '
        f'{len(inn)} wedges are in a cluster, {len(out)} in none (grey). {int(multi)} chromatophores contribute wedges '
        f'to more than one cluster (the overlapping motor units of the paper). Each dot is one wedge, drawn just inside '
        f'its chromatophore (white dot) on the side it covers; dot colour = cluster (colours repeat after 18 clusters).'
        + ('' if has_orientation(dataset, int(chunk)) else
           ' **This export has no wedge-orientation angles, so the dots sit at assumed directions;** press '
           '**Export quality checks** in the Datasets tab to add them.'), width=W)
    table = clusters.copy()
    for c in ('hull_px2', 'elongation', 'mean_corr'):
        table[c] = table[c].round(2)
    tab = pn.widgets.Tabulator(table, show_index=False, disabled=True, theme='midnight', width=720, height=260)
    return pn.Column(head, _map(dataset, layers, pos, 'Clusters of wedges that vary together'), tab,
                     sizing_mode='stretch_width')


# ── 2b. Motor units by the papers' method ─────────────────────────────────────────
def motor_units_view(dataset, chunk):
    ex = cv.extras_dir(dataset)
    cells, comps, wed = (cv._read(ex / f'motor_unit_{k}.csv') for k in ('cells', 'components', 'slices'))
    if not len(cells) or not len(comps):
        return _empty('*No motor-unit analysis for this dataset yet.* In the **Datasets** tab tick the dataset, press '
                      '**Run slice step** if it has not been run, then **Run motor-unit analysis**. It runs the CHROMAS '
                      'papers\' own method: PCA then ICA on the 36 wedge traces of each chromatophore, then clustering '
                      'of the independent components of all chromatophores.')
    if chunk is None:
        chunk = int(cells['chunk'].min())
    cells, comps, wed = (t[t['chunk'] == int(chunk)] for t in (cells, comps, wed))
    if not len(comps):
        return _empty(f'*No motor-unit analysis for chunk {chunk}.*')
    d = cv.load(dataset)
    pos = d['cent'][['x_px', 'y_px']]
    cell_pos = pos.loc[[c for c in cells['chromatophore'] if c in pos.index]]
    units = []
    for u, g in comps.groupby('unit'):
        cs = [c for c in g['chromatophore'].unique() if c in cell_pos.index]
        p = cell_pos.loc[cs].to_numpy(float)
        hull = np.nan
        if len(cs) >= 3:
            try:
                hull = float(ConvexHull(p).volume)
            except Exception:
                hull = 0.0
        units.append({'unit': int(u), 'n_components': len(g), 'n_chromatophores': g['chromatophore'].nunique(),
                      'hull_px2': hull})
    units = pd.DataFrame(units)
    k = cells['n_components']
    n_multi = int((units['n_chromatophores'] > 1).sum())
    per_cell = comps.groupby('chromatophore')['unit'].nunique()
    unlabelled = int((wed.groupby('chromatophore')['unit'].max() < 0).sum()) if len(wed) else 0
    head = pn.pane.Markdown(
        f'**{len(units)} putative motor units** from {len(comps)} independent components of {len(cells)} '
        f'chromatophores (chunk {chunk}). Components per chromatophore: mean {k.mean():.1f}, standard deviation '
        f'{k.std(ddof=0):.1f}, range {int(k.min())} to {int(k.max())}. {n_multi} of {len(units)} units span more than '
        f'one chromatophore; {int((per_cell > 1).sum())} chromatophores belong to more than one unit. '
        + (f'{unlabelled} chromatophores have no wedge labels (their ICA gave no usable result). ' if unlabelled else '')
        + 'Each component is a **putative** motor neuron and each cluster a **putative** motor unit: they are inferred '
          'from how the outline moves, not recorded from nerves. The CHROMAS authors found 3 to 4 components per '
          'chromatophore in cuttlefish, and ask for about 50 pixels across a chromatophore; with fewer pixels, or '
          'few frames, the numbers here are not reliable.', width=W)
    layers = [_cells_layer(cell_pos)]
    sl = slices(dataset)
    if len(wed) and int(chunk) in sl:
        ids, arr, orient = sl[int(chunk)]
        rad = dict(zip(ids, _cell_radius(arr) * 0.9))
        w = wed[wed['unit'] >= 0].join(cell_pos, on='chromatophore').dropna(subset=['x_px', 'y_px']).copy()
        if len(w):
            theta = np.deg2rad(_directions(w['chromatophore'].to_numpy(), w['slice'].to_numpy(), ids, orient,
                                           arr.shape[2]))
            r = w['chromatophore'].map(rad).fillna(4.0).to_numpy()
            w['wx'] = w['x_px'] + r * np.cos(theta)
            w['y_plot'] = -(w['y_px'] + r * np.sin(theta))
            w['colour'] = w['unit'].map(_colour)
            layers.append(hv.Points(w, ['wx', 'y_plot'], ['chromatophore', 'slice', 'unit', 'colour']).opts(
                size=6, color='colour', line_color='black', line_width=0.4, alpha=0.9, tools=['hover']))
    comp_hist = _hist(k, np.arange(0.5, k.max() + 1.5), 'independent components (putative motor neurons)',
                      'Components per chromatophore')
    unit_hist = _hist(units['n_chromatophores'], np.arange(0.5, units['n_chromatophores'].max() + 1.5),
                      'chromatophores in the unit', 'Chromatophores per motor unit', '#55a868')
    table = units.sort_values('n_chromatophores', ascending=False)
    table['hull_px2'] = table['hull_px2'].round(0)
    tab = pn.widgets.Tabulator(table, show_index=False, disabled=True, theme='midnight', width=560, height=240)
    return pn.Column(head, _map(dataset, layers, cell_pos, 'Putative motor units: each dot is a wedge, coloured by '
                                                           'the unit of its dominant component'),
                     pn.Row(comp_hist, unit_hist), tab, sizing_mode='stretch_width')


# ── 3. Cluster statistics ─────────────────────────────────────────────────────────
def _hist(values, bins, xlabel, title, colour='#4c72b0'):
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(values):
        return pn.pane.Markdown(f'*{title}: nothing to show.*', width=440)
    counts, edges = np.histogram(values, bins=bins)
    h = hv.Histogram((edges, counts), kdims=[hv.Dimension('x', label=xlabel)],
                     vdims=[hv.Dimension('count', label='clusters')]).opts(
        color=colour, tools=['hover'], width=440, height=280, title=title)
    return pn.pane.HoloViews(h, linked_axes=False)


def stats_view(dataset, chunk, thr, min_size, min_rest):
    if not slices(dataset) or chunk is None:
        return _no_slices()
    wedges, clusters, _ = cluster(dataset, chunk, thr, min_size, min_rest)
    if not len(clusters):
        return _empty('*No clusters with these settings. Lower the correlation or the smallest cluster size.*')
    sizes = _hist(clusters['n_slices'], min(20, max(5, len(clusters))), 'wedges in the cluster',
                  'Cluster size (wedges)')
    cells = _hist(clusters['n_cells'], np.arange(0.5, clusters['n_cells'].max() + 1.5), 'chromatophores in the cluster',
                  'Chromatophores per cluster', '#55a868')
    hulls = _hist(clusters['hull_px2'] / 1000.0, 15, 'convex hull area (thousand px²)',
                  'Cluster hull area (clusters of 3+ chromatophores)', '#c44e52')
    h = clusters.dropna(subset=['hull_px2']).assign(hull_kpx2=lambda t: t['hull_px2'] / 1000.0)
    if len(h):
        sc = hv.Scatter(h, [hv.Dimension('n_cells', label='chromatophores in the cluster')],
                        [hv.Dimension('hull_kpx2', label='hull area (thousand px²)'), 'cluster', 'shape']).opts(
            size=8, color='#8172b2', tools=['hover'], width=440, height=280,
            title='Chromatophore count against hull area')
        scatter = pn.pane.HoloViews(sc, linked_axes=False)
    else:
        scatter = pn.pane.Markdown('*No cluster spans three or more chromatophores.*', width=440)
    inn = wedges[wedges['cluster'] >= 0]
    per = inn.groupby('cell')['cluster'].nunique().value_counts().sort_index()
    bars = hv.Bars([(str(int(k)), int(v)) for k, v in per.items()],
                   kdims=[hv.Dimension('k', label='clusters the chromatophore contributes to')],
                   vdims=[hv.Dimension('n', label='chromatophores')]).opts(
        color='#dd8452', tools=['hover'], width=440, height=280, title='Chromatophores in more than one cluster')
    return pn.GridBox(sizes, cells, hulls, scatter, pn.pane.HoloViews(bars, linked_axes=False), ncols=2)


# ── 4. Network ──────────────────────────────────────────────────────────────────
def network_view(dataset, chunk, thr, min_size, min_rest):
    if not slices(dataset) or chunk is None:
        return _no_slices()
    wedges, clusters, pos = cluster(dataset, chunk, thr, min_size, min_rest)
    if not len(clusters):
        return _empty('*No clusters with these settings.*')
    layers = [_cells_layer(pos)]
    drawn = 0
    for _, row in clusters.iterrows():
        g = int(row['cluster'])
        cells = wedges[wedges['cluster'] == g]['cell'].unique()
        if len(cells) < 2:
            continue
        p = pos.loc[cells, ['x_px', 'y_px']].to_numpy(float)
        dmat = np.linalg.norm(p[:, None] - p[None], axis=2)
        mst = minimum_spanning_tree(dmat).tocoo()
        segs = pd.DataFrame({'x0': p[mst.row, 0], 'y0': -p[mst.row, 1], 'x1': p[mst.col, 0], 'y1': -p[mst.col, 1]})
        layers.append(hv.Segments(segs, ['x0', 'y0', 'x1', 'y1']).opts(color=_colour(g), line_width=2.5, alpha=0.8))
        layers.append(hv.Points(pd.DataFrame({'x_px': p[:, 0], 'y_plot': -p[:, 1], 'cluster': g}), ['x_px', 'y_plot'],
                                ['cluster']).opts(size=7, color=_colour(g), line_color='black', tools=['hover']))
        drawn += 1
    note = pn.pane.Markdown(
        f'{drawn} clusters span two or more chromatophores. Lines are the shortest tree through the chromatophores of '
        'one cluster; a chromatophore with dots of two colours contributes to two clusters. A tree running in a line is '
        'an elongated cluster; a clump is compact (see the shape column above).', width=W)
    return pn.Column(note, _map(dataset, layers, pos, 'Chromatophores of the same cluster, joined'),
                     sizing_mode='stretch_width')


# ── 5. Orientation ────────────────────────────────────────────────────────────────
def orientation_view(dataset, chunk, thr, min_size, min_rest):
    if not slices(dataset) or chunk is None:
        return _no_slices()
    wedges, clusters, pos = cluster(dataset, chunk, thr, min_size, min_rest)
    if not len(clusters):
        return _empty('*No clusters with these settings.*')
    if not has_orientation(dataset, int(chunk)):
        return _empty('*This export has no wedge-orientation angles* (the direction each chromatophore\'s wedge 0 '
                      'points in), so the wedge directions are unknown. In the **Datasets** tab, tick the dataset and '
                      'press **Export quality checks** again to add them.')
    diffs = []
    for g in clusters.loc[clusters['n_cells'] >= 2, 'cluster']:
        w = wedges[wedges['cluster'] == g]
        centre = pos.loc[w['cell'].unique(), ['x_px', 'y_px']].to_numpy(float).mean(axis=0)
        dx, dy = centre[0] - w['x_px'].to_numpy(), centre[1] - w['y_px'].to_numpy()
        ok = np.hypot(dx, dy) > 3
        phi = np.degrees(np.arctan2(dy[ok], dx[ok]))
        theta = w['direction'].to_numpy()[ok]
        diffs += list(np.abs((theta - phi + 180) % 360 - 180))
    if not diffs:
        return _empty('*No cluster spans two or more chromatophores, so there is no cluster centre to point at.*')
    diffs = np.array(diffs)
    counts, edges = np.histogram(diffs, bins=np.arange(0, 190, 15))
    h = hv.Histogram((edges, counts), kdims=[hv.Dimension('x', label='angle between wedge and the direction to its cluster centre (degrees)')],
                     vdims=[hv.Dimension('count', label='wedges')]).opts(
        color='#4c72b0', tools=['hover'], width=720, height=320,
        title='0° = the wedge points at the cluster centre, 180° = away from it')
    inward = float((diffs <= 45).mean() * 100)
    txt = pn.pane.Markdown(
        f'{len(diffs)} wedges from clusters of two or more chromatophores. **{inward:.0f}%** point within 45° of the '
        'centre of their cluster (25% if wedges pointed in random directions). In the motor-unit paper the wedges of a '
        'cluster point inward to a virtual centre between the chromatophores, as if one "virtual chromatophore" were '
        'pulled by one motor unit. Wedge directions come from each chromatophore\'s orientation angle, as CHROMAS draws '
        'its wedges. ' + ('That inward pattern is visible here.' if inward >= 40 else
                         'Here the directions look close to random, so these clusters show no inward pattern '
                         '(they may be correlated for other reasons, or the settings above may need changing).'),
        width=W)
    return pn.Column(txt, pn.pane.HoloViews(h, linked_axes=False))


# ── 6. CHROMAS's own area clusters ────────────────────────────────────────────────
def area_clusters_view(dataset):
    d = cv.load(dataset)
    m = cv.measures(d).reset_index()
    if not len(m):
        return _empty('*No positions for this dataset.*')
    method = 'CHROMAS (computed by chromas)' if 'CHROMAS (computed by chromas)' in cv.group_methods(dataset) \
        else 'Affinity propagation (as CHROMAS)'
    lab, _ = cv.groups(dataset, method)
    m['group'] = m['chromatophore'].map(lab).fillna(-1).astype(int)
    m['y_plot'] = -m['y_px']
    pos = m.set_index('chromatophore')[['x_px', 'y_px']]
    sizes = m['group'].value_counts()
    m['colour'] = m['group'].map(_colour)
    outline = pn.widgets.Checkbox(name='Outline each group with its convex hull (crowded when groups are spread out)',
                                  value=False)

    def plot(hulls):
        layers = [hv.Points(m[['x_px', 'y_plot', 'chromatophore']], ['x_px', 'y_plot'], ['chromatophore']).opts(
            size=1, alpha=0)]                      # first, invisible: sets the map's ranges to all chromatophores
        if hulls:
            for g in sizes.index:
                mem = m[m['group'] == g]
                if g < 0 or len(mem) < 3:
                    continue
                try:
                    v = mem[['x_px', 'y_plot']].to_numpy()[ConvexHull(mem[['x_px', 'y_plot']].to_numpy()).vertices]
                    layers.append(hv.Polygons([{'x_px': v[:, 0], 'y_plot': v[:, 1]}], kdims=['x_px', 'y_plot']).opts(
                        color=_colour(g), alpha=0.15, line_color=_colour(g), line_width=1.5))
                except Exception:
                    pass
        layers.append(hv.Points(m, ['x_px', 'y_plot'], ['chromatophore', 'group', 'colour']).opts(
            color='colour', size=9, line_color='black', line_width=0.5, tools=['hover']))
        return _map(dataset, layers, pos, 'Groups of chromatophores whose areas move together')
    big = int((sizes.drop(-1, errors='ignore') >= 2).sum())
    txt = pn.pane.Markdown(
        f'Grouping: **{method}** (the Chromatophores tab\'s Groups section uses the same data). {big} groups of two '
        f'or more chromatophores; mean size {sizes.drop(-1, errors="ignore").mean():.1f}, largest '
        f'{int(sizes.drop(-1, errors="ignore").max()) if big else 0}. Dot colour = group (colours repeat after 18 '
        'groups); grey = in no group. Groups here are of whole chromatophores (area correlation); the wedge clusters '
        'above are finer.', width=W)
    return pn.Column(txt, outline, pn.bind(plot, outline), sizing_mode='stretch_width')


# ── 7. Expansion against relaxation ───────────────────────────────────────────────
def speed_view(dataset):
    d = cv.load(dataset)
    a = d['areas'].astype(float)
    fps = d['fps']
    rest = a.replace(0, np.nan).quantile(0.10)
    keep = [c for c in a.columns if rest.get(c, 0) >= 5]
    if len(keep) < 3:
        return _empty('*Too few measurable chromatophores.*')
    raw = a[keep].to_numpy()
    win = max(5, int(round(0.28 * fps)) | 1)
    sm = savgol_filter(np.nan_to_num(raw), win, 2, axis=0)
    rate = np.gradient(sm, axis=0) * fps / rest[keep].to_numpy()          # relative area change per second
    bad = binary_dilation(raw == 0, structure=np.ones((win, 1), bool))      # frames next to a lost chromatophore
    rate = np.where(bad, np.nan, rate)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # a chromatophore lost for the whole video: all NaN
        rise = np.nanpercentile(rate, 99, axis=0)
        fall = -np.nanpercentile(rate, 1, axis=0)
    df = pd.DataFrame({'chromatophore': keep, 'expansion_rate': rise, 'relaxation_rate': fall}).dropna()
    df = df[(df['expansion_rate'] > 0) & (df['relaxation_rate'] > 0)]
    if len(df) < 3:
        return _empty('*Too few chromatophores with a measurable rate.*')
    df['ratio'] = df['expansion_rate'] / df['relaxation_rate']
    hi = float(max(df['expansion_rate'].max(), df['relaxation_rate'].max()) * 1.05)
    sc = hv.Scatter(df, [hv.Dimension('expansion_rate', label='fastest expansion (relative area per s)')],
                    [hv.Dimension('relaxation_rate', label='fastest relaxation (relative area per s)'),
                     'chromatophore', 'ratio']).opts(
        size=7, color='#4c72b0', alpha=0.8, tools=['hover'], width=480, height=420, xlim=(0, hi), ylim=(0, hi),
        title='One dot per chromatophore (above the line: relaxes faster)')
    line = hv.Curve([(0, 0), (hi, hi)]).opts(color='black', line_dash='dashed', line_width=1)
    counts, edges = np.histogram(np.log2(df['ratio']), bins=np.linspace(-2, 2, 21))
    h = hv.Histogram((edges, counts), kdims=[hv.Dimension('x', label='log2 (expansion rate / relaxation rate)')],
                     vdims=[hv.Dimension('count', label='chromatophores')]).opts(
        color='#55a868', tools=['hover'], width=480, height=420, title='Ratio of the two rates (0 = equal)')
    faster = float((df['ratio'] > 1).mean() * 100)
    txt = pn.pane.Markdown(
        f'{len(df)} chromatophores. Expansion is faster than relaxation in **{faster:.0f}%** (median ratio '
        f'{df["ratio"].median():.2f}). The motor-unit paper reports expansion as consistently faster and more '
        'stereotyped (active muscle contraction) than relaxation (passive recoil). Rates are the 99th percentile of the '
        'smoothed area change per second, relative to the resting area, so noise lifts both rates a little.', width=W)
    return pn.Column(txt, pn.Row(pn.pane.HoloViews(sc * line, linked_axes=False), pn.pane.HoloViews(h, linked_axes=False)))


# ── 8. Overlap of two chunks ──────────────────────────────────────────────────────
def overlap_view(dataset):
    cents = cv._read(cv.export_dir(dataset) / 'centroids.csv')
    if not len(cents) or 'chunk' not in cents.columns or cents['chunk'].nunique() < 2:
        return _empty('*Needs a dataset with two or more chunks* (a video that CHROMAS cut into in-focus chunks, such '
                      'as example_chunk_0 to example_chunk_2 imported as one dataset). This one has a single chunk.')
    chunks = sorted(cents['chunk'].unique())
    a_sel = pn.widgets.Select(name='Chunk A (red)', options=chunks, value=chunks[0], width=140)
    b_sel = pn.widgets.Select(name='Chunk B (blue)', options=chunks, value=chunks[1], width=140)

    def draw(a, b):
        if a == b:
            return _empty('*Choose two different chunks.*')
        ca_ = cents[cents['chunk'] == a].set_index('chromatophore')
        cb_ = cents[cents['chunk'] == b].set_index('chromatophore')
        both = ca_.index.intersection(cb_.index)
        only_a = ca_.index.difference(cb_.index)
        only_b = cb_.index.difference(ca_.index)
        rows = [(ca_.loc[both], 'both', CHUNK_COLOURS['both']), (ca_.loc[only_a], f'only chunk {a}', CHUNK_COLOURS['a']),
                (cb_.loc[only_b], f'only chunk {b}', CHUNK_COLOURS['b'])]
        allpos = pd.concat([r[0][['x_px', 'y_px']] for r in rows])
        layers = []
        for df, name, col in rows:
            if not len(df):
                continue
            p = df.reset_index()[['chromatophore', 'x_px', 'y_px']]
            p['y_plot'] = -p['y_px']
            p['group'] = name
            layers.append(hv.Points(p, ['x_px', 'y_plot'], ['chromatophore', 'group']).opts(
                color=col, size=7, line_color='white', line_width=0.5, tools=['hover']))
        txt = pn.pane.Markdown(
            f'**{len(both)}** chromatophores are in both chunks (black), **{len(only_a)}** only in chunk {a} (red), '
            f'**{len(only_b)}** only in chunk {b} (blue). Chunks are separate in-focus stretches; chromatophores '
            'missing from one are out of focus there, hidden, or newly segmented. Positions are those of each '
            'chunk\'s own frame.', width=W)
        return pn.Column(txt, _map(dataset, layers, allpos, 'Chromatophores common to two chunks'),
                         sizing_mode='stretch_width')
    return pn.Column(pn.Row(a_sel, b_sel), pn.bind(draw, a_sel, b_sel), sizing_mode='stretch_width')


# ── The tab ─────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _timing(label):
    """Prints how long a report took to build (server side, in the terminal that runs the dashboard)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f'[advanced analytics tab] {label}: {time.perf_counter() - t0:.2f} s', flush=True)


def supplemental_panel(dataset_sel):
    """The Supplemental tab, following the dataset chosen in the sidebar selector `dataset_sel`."""
    chunk = pn.widgets.Select(name='Chunk', options=[], width=100)
    thr = pn.widgets.FloatSlider(name='Wedges join a cluster when correlated at least', start=0.2, end=0.95, step=0.05,
                                 value=0.6, width=330)
    min_size = pn.widgets.IntSlider(name='Smallest cluster (wedges)', start=2, end=40, step=1, value=6, width=220)
    min_rest = pn.widgets.FloatSlider(name='Ignore wedges smaller than (px)', start=0.5, end=20, step=0.5, value=3.0,
                                      width=250)
    cell = pn.widgets.Select(name='Chromatophore', options=[], width=160)
    spacing = pn.widgets.FloatSlider(name='Offset between wedge traces (px)', start=0, end=100, step=1, value=10, width=260)

    def _reset_settings(*_):
        # the starting values: the sliders as first drawn, the first chunk and the most variable chromatophore
        thr.value, min_size.value, min_rest.value, spacing.value = 0.6, 6, 3.0, 10
        if chunk.options:
            chunk.value = chunk.options[0]
        if cell.options:
            cell.value = cell.options[0]
    reset_btn = pn.widgets.Button(name='Reset settings', width=120)
    reset_btn.on_click(_reset_settings)
    reset_btn2 = pn.widgets.Button(name='Reset settings', width=120)
    reset_btn2.on_click(_reset_settings)

    def _on_dataset(*_):
        ds = dataset_sel.value
        try:
            sl = slices(ds) if ds else {}
        except Exception:
            sl = {}
        chunk.options = sorted(sl)
        chunk.value = chunk.options[0] if chunk.options else None
        if sl:
            c0 = sorted(sl)[0]
            ids, arr, _ = sl[c0]
            order = [int(i) for i in ids[np.argsort(-arr.std(axis=0).sum(axis=1))]]      # most variable first
            cell.options = order
            cell.value = order[0]
        else:
            cell.options, cell.value = [], None

    # The wedge data is read only when a report is first opened (and after that whenever the dataset changes), so
    # opening the dashboard does not wait for it.
    ready = []

    def _ensure():
        if not ready:
            ready.append(True)
            with _timing('read wedge data'):
                _on_dataset()

    def _dataset_changed(*_):
        if ready:
            _on_dataset()

    dataset_sel.param.watch(_dataset_changed, 'value')

    def section(build, *widgets):
        def f(dataset_id, *vals):
            if not dataset_id:
                return pn.pane.Markdown('')
            exp = cv.export_dir(dataset_id)
            if not (exp / 'areas.csv').exists() or not (exp / 'centroids.csv').exists():
                return pn.pane.Markdown('*Needs the exported files with chromatophore positions: import the dataset '
                                        'again in the **Datasets** tab.*')
            try:
                with _timing(build.__name__):
                    return build(dataset_id, *vals)
            except Exception as exc:                  # one broken report must not take the tab down
                return pn.pane.Alert(f'Could not draw this report: {exc}', alert_type='warning')
        return pn.bind(f, dataset_sel, *widgets)

    def traces(ds, c, sp):
        return traces_view(ds, int(c), sp) if c is not None else _no_slices()

    cluster_controls = pn.Row(chunk, thr, min_size, min_rest, pn.Column(pn.Spacer(height=22), reset_btn))

    def lazy(title, make, needs_wedges=True):
        # the wedge reports need the chunk and chromatophore lists, which come from the wedge data
        return cv.lazy_card(title, make, before_open=_ensure if needs_wedges else None)

    return pn.Column(
        pn.pane.Markdown(
            '## Reports from the CHROMAS paper figures\n'
            'Built from the figures of the CHROMAS papers (the notebooks themselves were not available), for the '
            'dataset chosen on the left. The wedge reports need the CHROMAS **slice** step (Datasets tab). Each report '
            'is drawn when you open its panel.',
            width=W),
        pn.pane.Markdown('**Chunk and cluster settings.** The chunk applies to the motor units and to the four '
                         'correlation-cluster reports; the other three settings only to the correlation-cluster '
                         'reports. They are filled in when you open the first report.', width=W),
        cluster_controls,
        lazy('Wedge traces', lambda: pn.Column(
            pn.pane.Markdown('The 36 wedge areas of one chromatophore over time, stacked as in the paper, and as a heat '
                             'map. Hover a trace for its wedge number and area. Chromatophores are listed most variable '
                             'first.', width=W),
            pn.Row(cell, spacing, pn.Column(pn.Spacer(height=22), reset_btn2)),
            section(traces, cell, spacing))),
        lazy("Motor units (the papers' method: PCA, ICA, clustering of components)", lambda: pn.Column(
            pn.pane.Markdown('Run by CHROMAS itself (**Run motor-unit analysis** in the Datasets tab). This is the '
                             'result to report when the question is how the chromatophores are wired.', width=W),
            section(motor_units_view, chunk))),
        lazy("Wedge correlation clusters (this project's shortcut)", lambda: pn.Column(
            pn.pane.Markdown('Wedges whose areas rise and fall together, drawn on the skin, found by clustering their '
                             'correlations directly. **This is not the papers\' method: do not report these clusters '
                             'as motor units.** It needs no extra step and is useful for a first look. The settings '
                             'above apply to this and the next three reports.', width=W),
            section(clusters_view, chunk, thr, min_size, min_rest))),
        lazy('Cluster statistics', lambda: section(stats_view, chunk, thr, min_size, min_rest)),
        lazy('Cluster network', lambda: section(network_view, chunk, thr, min_size, min_rest)),
        lazy('Wedge orientation', lambda: section(orientation_view, chunk, thr, min_size, min_rest)),
        lazy('Chromatophore groups (CHROMAS area clusters)', lambda: section(area_clusters_view), needs_wedges=False),
        lazy('Expansion against relaxation', lambda: section(speed_view), needs_wedges=False),
        lazy('Chunk overlap', lambda: section(overlap_view), needs_wedges=False),
        sizing_mode='stretch_width')
