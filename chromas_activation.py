"""Chromatophore activation analysis on CHROMAS results.

This module turns the areas and positions that CHROMAS measures into activation events, neighbour coupling tests and
activation pathways:

    chromas_export.py  areas and centroids of each chromatophore    ->  areas.csv, centroids.csv
    this module        neighbours, activation events                ->  neighbours(), detect_events()
                       "are neighbours activated together?",         ->  coupling_test(), waves()
                       activation pathway

How it works, and why:

* Time is in seconds (the frame rate comes from the video), so the settings do not change with the frame rate.
* Each chromatophore's area is divided by its own resting size (a low percentile), so one threshold suits small and
  large chromatophores alike.
* An activation is an expansion whose rate rises above a multiple of that chromatophore's own noise, then counts
  only if it grows by at least min_amplitude.  Its onset is where the rate first left the noise (hysteresis), so
  noise wiggles neither start nor end events.
* Neighbours come from a Delaunay triangulation (each chromatophore is linked to the ones around it, whatever the
  density), trimmed of long edges.  A fixed radius in mm is still available.
* "Are neighbours coupled?" is a permutation test with a p value: the signals are shuffled among the positions
  many times, and the observed neighbour coupling is compared with that null distribution.
* Two coupling measures are used: the correlation of the expansion rates (needs no event threshold) and the share of
  events that coincide within coincidence_s.
* A second, independent event detector (peaks of the smoothed size picked by prominence, after Renard, Ukrow et al.
  2026) is run as a cross-check; the share of events both detectors find is reported.
* Whole-patch synchrony is tested separately (mean rate correlation of all pairs against circularly shifted
  signals), with the patch's dominant rhythm and each chromatophore's phase.  A display in which every
  chromatophore moves together (Humboldt squid flashing) gives no neighbour excess, so the neighbour test alone
  would look empty.
* A pathway links only neighbouring chromatophores whose onsets follow within max_lag_s.  Linked events form a
  wave; a plane fitted to onset time against position gives the wave's speed and direction, with R^2 and a
  permutation p value.

Input: an export folder written by chromas_export.py (areas.csv, centroids.csv, info.json, overlay.png), normally
processed/chromas_export/<dataset_id>/.  Output: processed/chromas_activation/<dataset_id>/.

Command line (this project's environment):
    python chromas_activation.py analyse <dataset_id | export folder> [--config examples/x/example.toml]
                                         [--px-per-mm 58.4] [--roi X Y W H]
    python chromas_activation.py example          # synthetic data with a known answer: checks the method

Settings: [activation] in config.defaults.toml; a --config file's [activation] table overrides them.
"""
import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import Delaunay, cKDTree

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / 'processed' / 'chromas_export'
OUT_DIR = BASE_DIR / 'processed' / 'chromas_activation'
CONFIG_PATH = BASE_DIR / 'config.defaults.toml'

DEFAULTS = {
    'px_per_mm': 0.0,            # pixels per mm in the CHROMAS master frame; 0 = unknown (distances in pixels only)
    'roi': [],                   # [x, y, width, height] in master-frame pixels; empty = whole frame
    'baseline_pct': 10.0,        # resting size = this percentile of a chromatophore's area
    'min_baseline_px': 20.0,     # smaller resting size than this: too small to measure, left out
    'max_missing_frac': 0.1,     # more frames than this with no area (0): left out
    'smooth_s': 0.2,             # Savitzky-Golay smoothing window for the expansion rate
    'onset_z': 4.0,              # an event starts when the rate exceeds this many noise sigmas
    'noise_from': 'rest',        # noise of the rate from the quietest 30 % of frames ('rest') or from all frames ('all')
    'min_amplitude': 0.10,       # and counts only if the chromatophore grows by this share of its resting size
    'min_gap_s': 0.2,            # events of one chromatophore closer than this are merged
    'neighbour_factor': 1.6,     # Delaunay edges longer than this times the median edge are not neighbours
    'neighbour_radius_mm': 0.0,  # > 0 (and px_per_mm set): neighbours = centres closer than this instead
    'coincidence_s': 0.2,        # two events coincide when their onsets are this close
    'max_lag_s': 0.5,            # a pathway link: a neighbour's onset follows within this time
    'min_wave_size': 4,          # chromatophores needed to fit a wave's speed and direction
    'min_wave_r2': 0.5,          # below this R^2 a wave is called unstructured (no single direction)
    'n_permutations': 999,
    'seed': 1,
    'prominence_range_frac': 0.15,   # cross-check detector: a peak counts if its prominence is at least this share
                                     # of the chromatophore's own range (and at least min_amplitude)
    'rhythm_band_hz': [0.5, 8.0],    # band searched for the patch's dominant rhythm (cut to what the recording allows)
    'rhythm_share': 0.5,             # the rhythm is called clear when this share of the band's power lies within
                                     # 15 % of the peak frequency (and the peak is not at the band's edge)
    'sync_trim_s': 1.0,              # the synchrony test leaves out this much at each end of the recording (at most
                                     # a tenth of it): filter edges and the start of activity are shared by all
}


# ── Settings ────────────────────────────────────────────────────────────────────
def load_settings(config=None, **overrides):
    cfg = dict(DEFAULTS)
    for path in (CONFIG_PATH, config):
        if path and Path(path).exists():
            with open(path, 'rb') as f:
                cfg.update(tomllib.load(f).get('activation', {}))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ── Input ───────────────────────────────────────────────────────────────────────
def resolve_export(dataset):
    """An export folder from a dataset id or a path."""
    p = Path(dataset)
    folder = p if p.is_dir() else EXPORT_DIR / str(dataset)
    if not (folder / 'areas.csv').exists():
        raise FileNotFoundError(f'No CHROMAS export at {folder} (areas.csv missing). Import the dataset first: '
                                f'python chromas_bridge.py import <dataset>.zarr')
    if not (folder / 'centroids.csv').exists():
        raise FileNotFoundError(f'{folder} has no centroids.csv (exported before positions were added). Import the '
                                f'dataset again to add it: python chromas_bridge.py import <dataset>.zarr')
    return folder


_EXPORT_CACHE = {}          # folder -> (stamp, data): the tabs and the replay all need the same big area table


def load_export(folder):
    """{'fps', 'info', 'chunks': {chunk: (areas DataFrame frames x chromatophores, times array, centroids)}}.

    Read once and kept (the last few exports) until one of its files changes; the callers only read the result."""
    folder = Path(folder)
    stamp = tuple(f.stat().st_mtime_ns if f.exists() else 0
                  for f in (folder / 'info.json', folder / 'areas.csv', folder / 'centroids.csv'))
    hit = _EXPORT_CACHE.get(str(folder))
    if hit is None or hit[0] != stamp:
        data = _read_export(folder)
        if len(_EXPORT_CACHE) >= 3:
            _EXPORT_CACHE.pop(next(iter(_EXPORT_CACHE)))
        hit = _EXPORT_CACHE[str(folder)] = (stamp, data)
    return hit[1]


def _read_export(folder):
    folder = Path(folder)
    info = json.loads((folder / 'info.json').read_text(encoding='utf-8'))
    areas = pd.read_csv(folder / 'areas.csv')
    cents = pd.read_csv(folder / 'centroids.csv')
    chunks = {}
    for chunk, a in areas.groupby('chunk'):
        wide = a.pivot(index='frame', columns='chromatophore', values='area').sort_index()
        times = a.groupby('frame')['t_s'].first().reindex(wide.index).to_numpy(float)
        c = cents[cents['chunk'] == chunk].set_index('chromatophore')[['x_px', 'y_px']]
        common = [k for k in wide.columns if k in c.index]
        chunks[int(chunk)] = (wide[common], times, c.loc[common])
    return {'fps': float(info['fps']), 'info': info, 'chunks': chunks, 'folder': folder}


# ── Signals ─────────────────────────────────────────────────────────────────────
def _odd(n, lo=5):
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


def prepare(wide, cent, fps, cfg):
    """Relative size and expansion rate of the chromatophores that can be measured.

    Returns (rel, rate, table): rel and rate are DataFrames (frames x kept chromatophores), table has one row per
    chromatophore with its position, resting size and, if left out, the reason.
    """
    a = wide.astype(float).replace(0, np.nan)
    with np.errstate(invalid='ignore'):          # a chromatophore never measured has no baseline (NaN), not a warning
        base = a.quantile(cfg['baseline_pct'] / 100.0)
    missing = a.isna().mean()
    x, y = cent['x_px'], cent['y_px']
    reason = pd.Series('', index=wide.columns, dtype=object)
    if cfg['roi']:
        rx, ry, rw, rh = cfg['roi']
        reason[~((x >= rx) & (x < rx + rw) & (y >= ry) & (y < ry + rh))] = 'outside ROI'
    reason[(reason == '') & (missing > cfg['max_missing_frac'])] = 'area missing in too many frames'
    reason[(reason == '') & ~(base >= cfg['min_baseline_px'])] = 'too small'
    keep = reason[reason == ''].index
    table = pd.DataFrame({'x_px': x, 'y_px': y, 'resting_px': base, 'missing_frac': missing,
                          'used': reason == '', 'left_out': reason})
    if cfg['px_per_mm']:
        table['resting_mm2'] = table['resting_px'] / cfg['px_per_mm'] ** 2
    rel = a[keep].interpolate(limit_direction='both') / base[keep] - 1.0
    if not len(keep):
        return rel, rel.copy(), table
    n = len(rel)
    win = _odd(cfg['smooth_s'] * fps)
    if win >= n:
        win = _odd(n - 2 if n % 2 == 0 else n - 1, lo=3) if n >= 4 else 0
    if win >= 3:
        sm = savgol_filter(rel.to_numpy(), win, 2, axis=0)
        rate = savgol_filter(rel.to_numpy(), win, 2, deriv=1, delta=1.0 / fps, axis=0)
    else:
        sm, rate = rel.to_numpy(), np.gradient(rel.to_numpy(), 1.0 / fps, axis=0)
    return (pd.DataFrame(sm, index=rel.index, columns=keep),
            pd.DataFrame(rate, index=rel.index, columns=keep), table)


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    s = 1.4826 * np.median(np.abs(x - np.median(x)))
    return s if s > 0 else (np.std(x) or 1e-9)


def detect_events(rel, rate, times, cfg):
    """Expansion events of one chromatophore.  rel and rate are 1-D arrays, times the frame times (s)."""
    # Noise of the rate, measured where the chromatophore is near rest (lowest 30 % of its size), so a recording
    # in which it is active most of the time does not inflate the threshold.
    rest = rel <= np.nanpercentile(rel, 30)
    if cfg.get('noise_from', 'rest') == 'rest' and rest.sum() >= 10:
        sig = robust_sigma(rate[rest])
    else:
        sig = robust_sigma(rate)                    # 'all': the whole recording
    hi, lo = cfg['onset_z'] * sig, sig
    n, events, last_peak = len(rate), [], -np.inf
    above = rate > hi
    starts = np.nonzero(above & ~np.r_[False, above[:-1]])[0]
    for s in starts:
        if times[s] < last_peak + cfg['min_gap_s']:
            continue
        o = s
        while o > 0 and rate[o - 1] > lo:
            o -= 1
        # sub-frame onset: where the rate crossed the low threshold
        if o > 0 and rate[o] != rate[o - 1]:
            f = (lo - rate[o - 1]) / (rate[o] - rate[o - 1])
            onset = times[o - 1] + np.clip(f, 0, 1) * (times[o] - times[o - 1])
        else:
            onset = times[o]
        if o == 0:
            continue        # already expanding when the recording starts: its onset is unknown
        e = s
        while e < n - 1 and rate[e + 1] > 0:
            e += 1
        p = o + int(np.argmax(rel[o:e + 1]))
        amp = rel[p] - rel[o]
        if amp < cfg['min_amplitude']:
            continue
        half = rel[o] + 0.5 * amp
        after = np.nonzero(rel[p:] <= half)[0]
        half_relax = times[p + after[0]] if len(after) else np.nan
        events.append({'onset_s': onset, 'peak_s': times[p], 'half_relax_s': half_relax,
                       'rise_s': times[p] - onset, 'amplitude': amp, 'max_rate_per_s': float(rate[s:e + 1].max()),
                       'noise_sigma_per_s': sig})
        last_peak = times[p]
    return events


def detect_events_prominence(rel, times, cfg):
    """Cross-check detector: peaks of the smoothed relative size picked by prominence.

    After Renard, Ukrow et al. (2026), who smoothed each trace, picked peaks by prominence and dropped those under
    15 % of the trace's range.  It uses no expansion rate and no noise estimate, so it fails differently from
    detect_events(); events that both find are the firm ones.  rel is the smoothed relative size (1-D).
    """
    rel = np.asarray(rel, float)
    if len(rel) < 5 or not np.isfinite(rel).all():
        return []
    need = max(cfg['min_amplitude'], cfg['prominence_range_frac'] * float(rel.max() - rel.min()))
    dt = float(np.median(np.diff(times)))
    peaks, props = find_peaks(rel, prominence=need, distance=max(1, int(round(cfg['min_gap_s'] / dt))))
    events = []
    for p, lb, prom in zip(peaks, props['left_bases'], props['prominences']):
        below = np.nonzero(rel[lb:p + 1] <= rel[p] - 0.9 * prom)[0]     # last frame still within 10 % of the base
        o = lb + (int(below[-1]) if len(below) else 0)
        if o == 0:
            continue        # already expanding when the recording starts, as in detect_events()
        events.append({'onset_s': float(times[o]), 'peak_s': float(times[p]), 'amplitude': float(prom)})
    return events


def match_events(a, b, window):
    """Number of times in a that have a time in b within window (each b used once)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    used, n = np.zeros(len(b), bool), 0
    for t in a:
        k = np.searchsorted(b, t)
        best = None
        for kk in (k - 1, k):
            if 0 <= kk < len(b) and not used[kk] and abs(b[kk] - t) <= window:
                if best is None or abs(b[kk] - t) < abs(b[best] - t):
                    best = kk
        if best is not None:
            used[best] = True
            n += 1
    return n


# ── Whole-patch synchrony ───────────────────────────────────────────────────────
def synchrony(rel, rate, times, cfg, rng, max_null_cells=300, max_null_perm=499):
    """Do the chromatophores of the patch move together, and at what rhythm?

    Returns (row, per): row is a dict for the chunk, per a DataFrame indexed like rel's columns with
    sync_with_patch (correlation of a chromatophore's expansion rate with the mean rate of all the others) and
    phase_deg (its phase against the patch mean at the dominant rhythm; only when the rhythm is clear).

    The test statistic is the mean correlation of the expansion rates over ALL pairs.  The null shifts every
    chromatophore's signal circularly by its own random amount, which keeps each signal's shape and rhythm but
    removes any common timing.
    """
    cols = list(rate.columns)
    per = pd.DataFrame({'sync_with_patch': np.nan, 'phase_deg': np.nan}, index=cols)
    row = {'n_chromatophores': len(cols), 'mean_pair_correlation': np.nan, 'null_mean': np.nan, 'p_value': np.nan,
           'median_sync_with_patch': np.nan, 'rhythm_hz': np.nan, 'rhythm_power_share': np.nan, 'rhythm_clear': False,
           'phase_agreement_R': np.nan, 'band_lo_hz': np.nan, 'band_hi_hz': np.nan}
    if len(times) > 1:
        dt0 = float(np.median(np.diff(times)))
        k = int(round(min(float(cfg['sync_trim_s']), 0.1 * len(times) * dt0) / dt0))
        if k > 0:
            rel, rate, times = rel.iloc[k:-k], rate.iloc[k:-k], np.asarray(times)[k:-k]
    r = rate.to_numpy(float)
    T, N = r.shape if r.ndim == 2 else (0, 0)
    if N < 4 or T < 20:
        return row, per
    sd = r.std(axis=0)
    z = np.where(sd > 0, (r - r.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)

    def mean_pair(zz):
        n = zz.shape[1]
        return float((np.mean(zz.sum(axis=1) ** 2) - n) / (n * (n - 1)))

    row['mean_pair_correlation'] = mean_pair(z)
    sub = np.sort(rng.choice(N, max_null_cells, replace=False)) if N > max_null_cells else np.arange(N)
    zs, obs_sub = z[:, sub], mean_pair(z[:, sub])
    base = np.arange(T)[:, None]
    null = np.array([mean_pair(zs[(base + rng.integers(1, T, len(sub))[None, :]) % T, np.arange(len(sub))[None, :]])
                     for _ in range(min(int(cfg['n_permutations']), max_null_perm))])
    row['null_mean'] = float(null.mean())
    row['p_value'] = float((1 + np.sum(null >= obs_sub)) / (1 + len(null)))
    others = (z.sum(axis=1)[:, None] - z) / (N - 1)
    oc = others - others.mean(axis=0)
    den = np.sqrt((z ** 2).sum(axis=0) * (oc ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        per['sync_with_patch'] = np.where(den > 0, (z * oc).sum(axis=0) / den, np.nan)
    row['median_sync_with_patch'] = float(np.nanmedian(per['sync_with_patch']))
    # rhythm of the patch mean, and each chromatophore's phase against it
    dt = float(np.median(np.diff(times)))
    duration = T * dt
    lo = max(float(cfg['rhythm_band_hz'][0]), 3.0 / duration)          # at least three cycles in the recording
    hi = min(float(cfg['rhythm_band_hz'][1]), 0.4 / dt)                # at least 2.5 frames per cycle
    row['band_lo_hz'], row['band_hi_hz'] = lo, hi
    x = rel.to_numpy(float)
    tt = np.arange(T) - (T - 1) / 2
    x = x - x.mean(axis=0) - np.outer(tt, (tt[:, None] * (x - x.mean(axis=0))).sum(axis=0) / (tt ** 2).sum())
    win = np.hanning(T)[:, None]
    F = np.fft.rfft(x * win, axis=0)
    Fm = np.fft.rfft(x.mean(axis=1) * win[:, 0])
    freqs = np.fft.rfftfreq(T, dt)
    band = (freqs >= lo) & (freqs <= hi)
    if hi > lo and band.sum() >= 5:
        power = np.abs(Fm) ** 2
        inb = np.flatnonzero(band)
        k = int(inb[np.argmax(power[band])])
        near = band & (np.abs(freqs - freqs[k]) <= max(0.15 * freqs[k], 2 * (freqs[1] - freqs[0])))
        tot = float(power[band].sum())
        row['rhythm_hz'] = float(freqs[k])
        row['rhythm_power_share'] = float(power[near].sum() / tot) if tot > 0 else np.nan
        row['rhythm_clear'] = bool(k not in (inb[0], inb[-1]) and np.isfinite(row['rhythm_power_share'])
                                   and row['rhythm_power_share'] >= cfg['rhythm_share'])
        if row['rhythm_clear']:
            ph = np.angle(F[k] * np.conj(Fm[k]))
            per['phase_deg'] = np.degrees(ph)
            row['phase_agreement_R'] = float(np.abs(np.mean(np.exp(1j * ph))))
    return row, per


# ── Space ───────────────────────────────────────────────────────────────────────
def neighbours(xy, cfg):
    """Neighbour pairs (i < j index pairs into xy) and the typical spacing (median nearest-neighbour distance, px)."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return np.zeros((0, 2), int), np.nan
    d_nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    spacing = float(np.median(d_nn))
    if cfg['neighbour_radius_mm'] and cfg['px_per_mm']:
        pairs = np.array(sorted(cKDTree(xy).query_pairs(cfg['neighbour_radius_mm'] * cfg['px_per_mm'])), int)
        return pairs.reshape(-1, 2), spacing
    if len(xy) < 4:
        pairs = np.array([(i, j) for i in range(len(xy)) for j in range(i + 1, len(xy))], int)
        return pairs, spacing
    tri = Delaunay(xy)
    e = np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    length = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    return e[length <= cfg['neighbour_factor'] * np.median(length)], spacing


def coincidence_matrix(onsets, window):
    """Share of events that coincide: for i, j, (events of i with a j event within window + the reverse) / (n_i + n_j)."""
    n = len(onsets)
    m = np.full((n, n), np.nan)
    hits = np.zeros((n, n))
    counts = np.array([len(o) for o in onsets], float)
    srt = [np.sort(np.asarray(o, float)) for o in onsets]
    for i in range(n):
        if not len(srt[i]):
            continue
        for j in range(n):
            if i == j or not len(srt[j]):
                continue
            k = np.searchsorted(srt[j], srt[i])
            near = np.zeros(len(srt[i]), bool)
            for kk in (k - 1, k):
                ok = (kk >= 0) & (kk < len(srt[j]))
                near[ok] |= np.abs(srt[j][kk[ok]] - srt[i][ok]) <= window
            hits[i, j] = near.sum()
    tot = counts[:, None] + counts[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(tot > 0, (hits + hits.T) / tot, np.nan)
    np.fill_diagonal(m, np.nan)
    return m


def coupling_test(m, pairs, n_perm, rng):
    """Neighbour coupling against a permutation null (signals shuffled among the positions).

    Statistic: mean of m over neighbour pairs minus mean over all other pairs.  Shuffling the nodes keeps the
    multiset of all pair values, so only the neighbour mean changes from one permutation to the next.
    """
    n = m.shape[0]
    if n < 4 or not len(pairs):
        return {'n_neighbour_pairs': int(len(pairs)), 'statistic': np.nan, 'p_value': np.nan}
    iu = np.triu_indices(n, 1)
    allv = m[iu]
    ok_all = np.isfinite(allv)
    tot_sum, tot_n = allv[ok_all].sum(), ok_all.sum()

    def stat(perm):
        v = m[perm[pairs[:, 0]], perm[pairs[:, 1]]]
        ok = np.isfinite(v)
        if not ok.any() or tot_n - ok.sum() <= 0:
            return np.nan, np.nan, np.nan
        nb = v[ok].mean()
        other = (tot_sum - v[ok].sum()) / (tot_n - ok.sum())
        return nb - other, nb, other

    obs, nb_mean, other_mean = stat(np.arange(n))
    null = np.array([stat(rng.permutation(n))[0] for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    p = (1 + np.sum(null >= obs)) / (1 + len(null)) if np.isfinite(obs) and len(null) else np.nan
    z = (obs - null.mean()) / null.std() if len(null) > 1 and null.std() > 0 else np.nan
    return {'n_neighbour_pairs': int(len(pairs)), 'neighbour_mean': nb_mean, 'other_mean': other_mean,
            'statistic': obs, 'null_mean': float(null.mean()) if len(null) else np.nan, 'z': z, 'p_value': p}


def distance_profile(m_corr, m_coin, xy, spacing, px_per_mm=0.0):
    """Mean coupling against distance, in units of the typical spacing (and in mm when the scale is known)."""
    n = len(xy)
    iu = np.triu_indices(n, 1)
    d = np.linalg.norm(xy[iu[0]] - xy[iu[1]], axis=1) / spacing
    edges = [0, 1.5, 2.5, 4, 6, 10, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        c, k = m_corr[iu][sel], m_coin[iu][sel]
        mm = spacing / px_per_mm if px_per_mm else np.nan
        rows.append({'distance_spacings': f'{lo:g}-{hi:g}', 'n_pairs': int(sel.sum()),
                     'distance_mm_from': lo * mm, 'distance_mm_to': hi * mm if np.isfinite(hi) else np.nan,
                     'spacing_px': spacing, 'spacing_mm': mm,
                     'mean_rate_correlation': float(np.nanmean(c)) if np.isfinite(c).any() else np.nan,
                     'mean_coincidence': float(np.nanmean(k)) if np.isfinite(k).any() else np.nan})
    return pd.DataFrame(rows)


# ── Waves ───────────────────────────────────────────────────────────────────────
def _plane_fit(x, y, t):
    A = np.c_[np.ones_like(x), x, y]
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss = np.sum((t - t.mean()) ** 2)
    r2 = 1 - np.sum((t - pred) ** 2) / ss if ss > 0 else np.nan
    return coef, r2


def _clusters(chrom, onset, nb, max_lag):
    """Union-find labels of events linked through neighbouring chromatophores within max_lag."""
    parent = np.arange(len(onset))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_chrom = {}
    for i, c in enumerate(chrom):
        by_chrom.setdefault(c, []).append(i)
    for i, c in enumerate(chrom):
        for o in nb.get(c, ()):
            for j in by_chrom.get(o, ()):
                if j > i and abs(onset[j] - onset[i]) <= max_lag:
                    parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(onset))])


def _largest_cluster(chrom, labels):
    best = 0
    for lab in np.unique(labels):
        best = max(best, len(set(chrom[labels == lab])))
    return best


def chance_wave_size(ev, nb, cfg, t_range, rng, n=99):
    """Largest group of linked events expected by chance (95th percentile).

    Each chromatophore's own events are shifted by one random offset (circularly within the recording), which keeps
    every chromatophore's event rate and rhythm but breaks any timing relation between chromatophores.
    """
    if not len(ev):
        return 0
    t0, t1 = t_range
    span = t1 - t0
    chrom = ev['chromatophore'].to_numpy()
    onset = ev['onset_s'].to_numpy()
    sizes = []
    for _ in range(n):
        shift = {c: rng.uniform(0, span) for c in np.unique(chrom)}
        sh = t0 + (onset - t0 + np.array([shift[c] for c in chrom])) % span
        sizes.append(_largest_cluster(chrom, _clusters(chrom, sh, nb, cfg['max_lag_s'])))
    return int(np.percentile(sizes, 95))


def waves(ev, pairs_ids, cfg, frame_s, rng, t_range):
    """Group events into waves and fit speed and direction.  ev: events with chunk, chromatophore, onset_s, x_px, y_px.

    A group counts as a wave only if it is larger than both min_wave_size and the largest group that the same events
    form by chance (chance_wave_size).  Returns (events with a 'wave' column, waves, links, chance size).
    """
    nb = {}
    for a, b in pairs_ids:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    ev = ev.sort_values('onset_s').reset_index(drop=True)
    chance = chance_wave_size(ev, nb, cfg, t_range, rng)
    min_size = max(int(cfg['min_wave_size']), chance + 1)
    raw = _clusters(ev['chromatophore'].to_numpy(), ev['onset_s'].to_numpy(), nb, cfg['max_lag_s'])
    ev['wave'] = -1
    wave_rows, links = [], []
    k = 0
    for lab in np.unique(raw):
        g = ev[raw == lab]
        if g['chromatophore'].nunique() < min_size:
            continue
        ev.loc[g.index, 'wave'] = k
        g = g.sort_values('onset_s').drop_duplicates('chromatophore')
        x, y, t = g['x_px'].to_numpy(), g['y_px'].to_numpy(), g['onset_s'].to_numpy()
        coef, r2 = _plane_fit(x, y, t)
        null = [_plane_fit(x, y, rng.permutation(t))[1] for _ in range(199)]
        p_r2 = (1 + np.sum(np.asarray(null) >= r2)) / 200 if np.isfinite(r2) else np.nan
        slow = math.hypot(coef[1], coef[2])            # s per px
        spread = t.max() - t.min()
        travelling = bool(np.isfinite(r2) and r2 >= cfg['min_wave_r2'] and p_r2 < 0.05 and spread > frame_s)
        speed = 1.0 / slow if travelling and slow > 0 else np.nan
        row = {'wave': k, 'start_s': t.min(), 'spread_s': spread, 'n_chromatophores': len(g),
               'first_chromatophore': int(g['chromatophore'].iloc[0]),
               'kind': 'travelling' if travelling else ('simultaneous' if spread <= frame_s else 'unstructured'),
               'speed_px_per_s': speed, 'direction_deg': math.degrees(math.atan2(coef[2], coef[1])) if travelling else np.nan,
               'r2': r2, 'p_r2': p_r2, 'centre_x_px': x.mean(), 'centre_y_px': y.mean()}
        if cfg['px_per_mm']:
            row['speed_mm_per_s'] = speed / cfg['px_per_mm']
        wave_rows.append(row)
        # pathway links: each chromatophore's activator = the neighbour in this wave whose onset came last before it
        pos = {int(r.chromatophore): (r.x_px, r.y_px, r.onset_s) for r in g.itertuples()}
        for c, (xc, yc, tc) in pos.items():
            cands = [(pos[o][2], o) for o in nb.get(c, ()) if o in pos and 0 <= tc - pos[o][2] <= cfg['max_lag_s'] and o != c]
            cands = [cd for cd in cands if cd[0] < tc or (cd[0] == tc and cd[1] < c)]
            if not cands:
                continue
            tp, o = max(cands)
            dist = math.hypot(xc - pos[o][0], yc - pos[o][1])
            lag = tc - tp
            link = {'wave': k, 'from': o, 'to': c, 'lag_s': lag, 'distance_px': dist,
                    'speed_px_per_s': dist / lag if lag > frame_s else np.nan,
                    'simultaneous': lag <= frame_s}
            if cfg['px_per_mm']:
                link['distance_mm'] = dist / cfg['px_per_mm']
                link['speed_mm_per_s'] = link['speed_px_per_s'] / cfg['px_per_mm']
            links.append(link)
        k += 1
    return ev, pd.DataFrame(wave_rows), pd.DataFrame(links), chance


# ── Pathway image ───────────────────────────────────────────────────────────────
def draw_pathways(out_png, table, pairs_ids, links, overlay=None, roi=None):
    import cv2
    used = table[table['used']]
    if overlay is not None and Path(overlay).exists():
        img = cv2.imread(str(overlay))
    else:
        h = int(table['y_px'].max() * 1.05) + 20
        w = int(table['x_px'].max() * 1.05) + 20
        img = np.full((h, w, 3), 255, np.uint8)
    img = (img * 0.6 + 255 * 0.4).astype(np.uint8)          # fade the background
    pos = {int(c): (int(round(r.x_px)), int(round(r.y_px))) for c, r in table.iterrows()}
    for a, b in pairs_ids:
        cv2.line(img, pos[a], pos[b], (200, 200, 200), 1, cv2.LINE_AA)
    for c in used.index:
        cv2.circle(img, pos[int(c)], 3, (90, 90, 90), -1, cv2.LINE_AA)
    if roi:
        x, y, w, h = map(int, roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if len(links):
        lag = links['lag_s'].to_numpy()
        norm = np.clip(lag / max(lag.max(), 1e-9), 0, 1)
        colours = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        for a, b, col in zip(links['from'], links['to'], colours):
            p0, p1 = pos[int(a)], pos[int(b)]
            cv2.arrowedLine(img, p0, p1, tuple(int(v) for v in col), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.imwrite(str(out_png), img)


def draw_wave_map(out_png, table, ev_wave, links, wave_row, overlay=None, roi=None):
    """Activation map of one wave: each chromatophore coloured by its onset time (blue first, red last), arrows
    from activator to activated, and a large arrow for the fitted direction of travel."""
    import cv2
    if overlay is not None and Path(overlay).exists():
        img = cv2.imread(str(overlay))
    else:
        img = np.full((int(table['y_px'].max() * 1.05) + 20, int(table['x_px'].max() * 1.05) + 20, 3), 255, np.uint8)
    img = (img * 0.5 + 255 * 0.5).astype(np.uint8)
    pos = {int(c): (int(round(r.x_px)), int(round(r.y_px))) for c, r in table.iterrows()}
    for c in table[table['used']].index:
        cv2.circle(img, pos[int(c)], 3, (170, 170, 170), -1, cv2.LINE_AA)
    t = ev_wave['onset_s'].to_numpy()
    norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    colour = {int(c): tuple(int(v) for v in cc) for c, cc in zip(ev_wave['chromatophore'], col)}
    for a, b in zip(links['from'], links['to']):
        cv2.arrowedLine(img, pos[int(a)], pos[int(b)], colour[int(b)], 2, cv2.LINE_AA, tipLength=0.3)
    for c, cc in colour.items():
        cv2.circle(img, pos[c], 7, cc, -1, cv2.LINE_AA)
    if roi:
        x, y, w, h = map(int, roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    label = f"wave {int(wave_row['wave'])}: {wave_row['kind']}, {int(wave_row['n_chromatophores'])} chromatophores, " \
            f"spread {wave_row['spread_s']:.2f} s"
    if wave_row['kind'] == 'travelling':
        cx, cy = int(wave_row['centre_x_px']), int(wave_row['centre_y_px'])
        ang = math.radians(wave_row['direction_deg'])
        L = 0.25 * min(img.shape[:2])
        cv2.arrowedLine(img, (cx, cy), (int(cx + L * math.cos(ang)), int(cy + L * math.sin(ang))), (0, 0, 0), 5,
                        cv2.LINE_AA, tipLength=0.25)
        label += f", {wave_row['speed_px_per_s']:.0f} px/s"
        if 'speed_mm_per_s' in wave_row and pd.notna(wave_row.get('speed_mm_per_s')):
            label += f" ({wave_row['speed_mm_per_s']:.2f} mm/s)"
    scale = max(0.5, img.shape[1] / 1400)
    cv2.putText(img, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)


# ── Whole analysis ──────────────────────────────────────────────────────────────
def analyse(dataset, cfg, out_dir=None, log=print):
    folder = resolve_export(dataset)
    data = load_export(folder)
    fps = data['fps']
    dataset_id = folder.name
    out = Path(out_dir) if out_dir else OUT_DIR / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg['seed'])
    all_tables, all_events, all_waves, all_links, all_profiles, tests, nb_rows = [], [], [], [], [], [], []
    all_prom, cross_rows, sync_rows = [], [], []
    colour = None                 # colour class of each chromatophore, when the export has it (extras/colour_classes.csv)
    cc_file = folder / 'extras' / 'colour_classes.csv'
    if cc_file.exists() and cc_file.stat().st_size > 2:
        colour = pd.read_csv(cc_file).set_index(['chunk', 'chromatophore'])['colour_class']
    rng_new = np.random.default_rng(cfg['seed'] + 1)     # its own stream, so the older tests give the numbers they gave before
    for chunk, (wide, times, cent) in data['chunks'].items():
        rel, rate, table = prepare(wide, cent, fps, cfg)
        ids = list(rel.columns)
        xy = table.loc[ids, ['x_px', 'y_px']].to_numpy(float)
        pairs, spacing = neighbours(xy, cfg)
        pairs_ids = [(ids[i], ids[j]) for i, j in pairs]
        ev_rows = []
        for c in ids:
            for e in detect_events(rel[c].to_numpy(), rate[c].to_numpy(), times, cfg):
                ev_rows.append({'chunk': chunk, 'chromatophore': c, **e})
        ev = pd.DataFrame(ev_rows, columns=['chunk', 'chromatophore', 'onset_s', 'peak_s', 'half_relax_s', 'rise_s',
                                            'amplitude', 'max_rate_per_s', 'noise_sigma_per_s'])
        duration = (times[-1] - times[0]) + 1.0 / fps
        n_ev = ev.groupby('chromatophore').size()
        table['chunk'] = chunk
        table['n_events'] = n_ev.reindex(table.index).fillna(0).astype(int)
        table['events_per_min'] = table['n_events'] / duration * 60
        deg = pd.Series(0, index=table.index)
        for a, b in pairs_ids:
            deg[a] += 1
            deg[b] += 1
        table['n_neighbours'] = deg
        # cross-check detector (prominence) and its agreement with the rate detector
        prom_rows = [{'chunk': chunk, 'chromatophore': c, **e} for c in ids
                     for e in detect_events_prominence(rel[c].to_numpy(), times, cfg)]
        prom = pd.DataFrame(prom_rows, columns=['chunk', 'chromatophore', 'onset_s', 'peak_s', 'amplitude'])
        all_prom.append(prom)
        matched = sum(match_events(ev.loc[ev['chromatophore'] == c, 'peak_s'], prom.loc[prom['chromatophore'] == c, 'peak_s'],
                                   cfg['coincidence_s']) for c in ids) if len(ev) and len(prom) else 0
        cross_rows.append({'chunk': chunk, 'rate_events': int(len(ev)), 'prominence_events': int(len(prom)),
                           'found_by_both': int(matched),
                           'share_of_rate_events': matched / len(ev) if len(ev) else np.nan,
                           'share_of_prominence_events': matched / len(prom) if len(prom) else np.nan})
        # whole-patch synchrony and rhythm
        srow, sper = synchrony(rel, rate, times, cfg, rng_new)
        sync_rows.append({'chunk': chunk, **srow})
        table['sync_with_patch'] = sper['sync_with_patch'].reindex(table.index)
        table['phase_deg'] = sper['phase_deg'].reindex(table.index)
        if colour is not None and chunk in colour.index.get_level_values(0):
            table['colour_class'] = colour.xs(chunk, level=0).reindex(table.index)
        all_tables.append(table.reset_index())
        # coupling
        if len(ids) >= 4:
            r = rate.to_numpy()
            with np.errstate(invalid='ignore', divide='ignore'):
                m_corr = np.corrcoef(r, rowvar=False)
            np.fill_diagonal(m_corr, np.nan)
            m_coin = coincidence_matrix([ev.loc[ev['chromatophore'] == c, 'onset_s'].to_numpy() for c in ids],
                                        cfg['coincidence_s'])
            for name, m in (('rate_correlation', m_corr), ('event_coincidence', m_coin)):
                t = coupling_test(m, pairs, cfg['n_permutations'], rng)
                tests.append({'chunk': chunk, 'measure': name, **t})
            if len(prom):
                m_prom = coincidence_matrix([prom.loc[prom['chromatophore'] == c, 'onset_s'].to_numpy() for c in ids],
                                            cfg['coincidence_s'])
                tests.append({'chunk': chunk, 'measure': 'event_coincidence_prominence',
                              **coupling_test(m_prom, pairs, cfg['n_permutations'], rng_new)})
            prof = distance_profile(m_corr, m_coin, xy, spacing, cfg['px_per_mm'])
            prof.insert(0, 'chunk', chunk)
            all_profiles.append(prof)
            for (i, j) in pairs:
                d_px = float(np.linalg.norm(xy[i] - xy[j]))
                nb_rows.append({'chunk': chunk, 'a': ids[i], 'b': ids[j], 'distance_px': d_px,
                                'distance_mm': d_px / cfg['px_per_mm'] if cfg['px_per_mm'] else np.nan,
                                'rate_correlation': m_corr[i, j], 'event_coincidence': m_coin[i, j]})
        # waves
        if len(ev):
            ev = ev.join(table[['x_px', 'y_px']], on='chromatophore')
            ev, wv, lk, chance = waves(ev, pairs_ids, cfg, 1.0 / fps, rng, (times[0], times[-1] + 1.0 / fps))
            tests.append({'chunk': chunk, 'measure': 'chance_wave_size', 'statistic': chance})
            for df in (wv, lk):
                if len(df):
                    df.insert(0, 'chunk', chunk)
            all_events.append(ev)
            all_waves.append(wv)
            all_links.append(lk)
        log(f'chunk {chunk}: {len(ids)} of {wide.shape[1]} chromatophores measured, {len(ev)} events, '
            f'{len(pairs)} neighbour pairs (spacing {spacing:.1f} px)')

    def cat(frames):
        frames = [f for f in frames if f is not None and len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    table, events, wv, lk = cat(all_tables), cat(all_events), cat(all_waves), cat(all_links)
    tests_df, prof = pd.DataFrame(tests), cat(all_profiles)
    table.to_csv(out / 'chromatophores.csv', index=False, float_format='%.4g')
    events.to_csv(out / 'events.csv', index=False, float_format='%.4g')
    cat(all_prom).to_csv(out / 'events_prominence.csv', index=False, float_format='%.4g')
    cross_df, sync_df = pd.DataFrame(cross_rows), pd.DataFrame(sync_rows)
    cross_df.to_csv(out / 'detector_agreement.csv', index=False, float_format='%.4g')
    sync_df.to_csv(out / 'synchrony.csv', index=False, float_format='%.4g')
    pd.DataFrame(nb_rows).to_csv(out / 'neighbour_pairs.csv', index=False, float_format='%.4g')
    prof.to_csv(out / 'distance_profile.csv', index=False, float_format='%.4g')
    tests_df.to_csv(out / 'coupling_tests.csv', index=False, float_format='%.4g')
    wv.to_csv(out / 'waves.csv', index=False, float_format='%.4g')
    lk.to_csv(out / 'wave_links.csv', index=False, float_format='%.4g')
    # the wide tables: one column per chromatophore
    _wide_tables(data, table, cfg, out)
    first_chunk = min(data['chunks']) if data['chunks'] else 0
    if len(table):
        t0 = table[table['chunk'] == first_chunk].set_index('chromatophore')
        p0 = [(a, b) for a, b in zip(*(pd.DataFrame(nb_rows).query('chunk == @first_chunk')[['a', 'b']].T.to_numpy()))] \
            if nb_rows else []
        l0 = lk[lk['chunk'] == first_chunk] if len(lk) else lk
        ovl = folder / 'overlay.png' if data['info'].get('overlay') else None
        draw_pathways(out / 'pathways.png', t0, p0, l0, overlay=ovl, roi=cfg['roi'])
        for old in out.glob('wave_map_*.png'):
            old.unlink()
        if len(wv):
            w0 = wv[wv['chunk'] == first_chunk].sort_values('n_chromatophores', ascending=False).head(10)
            for wr in w0.to_dict('records'):
                ew = events[(events['chunk'] == first_chunk) & (events['wave'] == wr['wave'])] \
                    .sort_values('onset_s').drop_duplicates('chromatophore')
                lw = l0[l0['wave'] == wr['wave']] if len(l0) else l0
                draw_wave_map(out / f"wave_map_{int(wr['wave'])}.png", t0, ew, lw, wr, overlay=ovl, roi=cfg['roi'])
    summary = summarise(dataset_id, data, cfg, table, events, tests_df, wv)
    summary['detector_agreement'] = cross_df.to_dict('records')
    summary['synchrony'] = sync_df.to_dict('records')
    if 'colour_class' in table and table['colour_class'].nunique() > 1:
        used = table[table['used']]
        summary['colour_classes'] = [{'colour_class': int(k), 'chromatophores_measured': int(len(g)),
                                      'events': int(g['n_events'].sum()),
                                      'median_sync_with_patch': float(g['sync_with_patch'].median())}
                                     for k, g in used.dropna(subset=['colour_class']).groupby('colour_class')]
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=_jsonable), encoding='utf-8')
    text = summary_text(summary)
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    log(text)
    log(f'Results in {out}')
    return out, summary


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


def _wide_tables(data, table, cfg, out):
    """areas_wide.csv (frame x chromatophore, mm^2 if the scale is known) and centroids_wide.csv."""
    cols, cents = [], []
    for chunk, (wide, times, cent) in data['chunks'].items():
        used = table[(table['chunk'] == chunk) & table['used']]['chromatophore']
        w = wide[used].replace(0, np.nan)
        if cfg['px_per_mm']:
            w = w / cfg['px_per_mm'] ** 2
        w.columns = [f'C{chunk}_{c}' if len(data['chunks']) > 1 else f'C{c}' for c in w.columns]
        w.index = pd.Index(np.round(times, 4), name='t_s')
        cols.append(w)
        c = cent.loc[used].copy()
        c.index = w.columns
        cents.append(c)
    if cols:
        pd.concat(cols, axis=1).to_csv(out / 'areas_wide.csv', float_format='%.5g')
        pd.concat(cents).T.to_csv(out / 'centroids_wide.csv', float_format='%.2f')


def summarise(dataset_id, data, cfg, table, events, tests, wv):
    s = {'dataset': dataset_id, 'synthetic': bool(data['info'].get('synthetic')), 'fps': data['fps'],
         'px_per_mm': cfg['px_per_mm'] or None, 'roi': cfg['roi'] or None,
         'chromatophores_total': int(len(table)), 'chromatophores_measured': int(table['used'].sum()) if len(table) else 0,
         'events': int(len(events)),
         'chromatophores_with_events': int((table['n_events'] > 0).sum()) if len(table) else 0,
         'coupling_tests': tests.to_dict('records'), 'waves': int(len(wv)),
         'travelling_waves': int((wv['kind'] == 'travelling').sum()) if len(wv) else 0}
    if len(wv) and (wv['kind'] == 'travelling').any():
        tw = wv[wv['kind'] == 'travelling']
        s['median_wave_speed_px_per_s'] = float(tw['speed_px_per_s'].median())
        if cfg['px_per_mm']:
            s['median_wave_speed_mm_per_s'] = float(tw['speed_mm_per_s'].median())
    s['settings'] = cfg
    return s


def summary_text(s):
    lines = [f"Dataset {s['dataset']}" + (' (SYNTHETIC)' if s['synthetic'] else ''),
             f"{s['chromatophores_measured']} of {s['chromatophores_total']} chromatophores measured; "
             f"{s['events']} activation events on {s['chromatophores_with_events']} of them."]
    for t in s['coupling_tests']:
        if t['measure'] == 'chance_wave_size':
            lines.append(f"  chunk {t['chunk']}: groups of up to {int(t['statistic'])} linked chromatophores happen by chance; "
                         f"larger groups count as waves")
            continue
        p = t.get('p_value')
        if p is None or not np.isfinite(p):
            lines.append(f"  chunk {t['chunk']} {t['measure']}: not enough data for the test")
            continue
        verdict = 'neighbours are more coupled than distant pairs' if p < 0.05 else 'no evidence that neighbours are more coupled'
        lines.append(f"  chunk {t['chunk']} {t['measure']}: neighbours {t['neighbour_mean']:.3f} vs others "
                     f"{t['other_mean']:.3f}, p = {p:.3g} -> {verdict}")
    lines.append(f"{s['waves']} wave(s) of linked activations, {s['travelling_waves']} with a clear direction.")
    nb_ok = any(t.get('p_value') is not None and np.isfinite(t['p_value']) and t['p_value'] < 0.05
                for t in s['coupling_tests'] if t['measure'] != 'chance_wave_size')
    for y in s.get('synchrony', []):
        p = y.get('p_value')
        if p is None or not np.isfinite(p):
            lines.append(f"  chunk {y['chunk']} whole-patch synchrony: not enough data for the test")
            continue
        together = p < 0.05
        small = together and y['mean_pair_correlation'] < 0.05
        lines.append(f"  chunk {y['chunk']} whole-patch synchrony: mean rate correlation of all pairs "
                     f"{y['mean_pair_correlation']:.3f} (shifted signals give {y['null_mean']:.3f}), p = {p:.3g} -> "
                     + ('a shared component, but a very small one' if small else 'the patch moves together' if together
                        else 'no evidence that the patch moves together'))
        together = together and not small
        if y.get('rhythm_clear'):
            lines.append(f"  chunk {y['chunk']} rhythm: {y['rhythm_hz']:.2f} Hz ({100 * y['rhythm_power_share']:.0f} % of the "
                         f"band's power); phase agreement between chromatophores R = {y['phase_agreement_R']:.2f} "
                         f"(1 = all in phase)")
        elif y.get('band_lo_hz') is not None and np.isfinite(y.get('band_lo_hz', np.nan)):
            lines.append(f"  chunk {y['chunk']} rhythm: no clear rhythm between {y['band_lo_hz']:.2g} and "
                         f"{y['band_hi_hz']:.2g} Hz")
        if together and not nb_ok:
            lines.append("  Note: the patch moves together but neighbours are not more coupled than distant pairs. That is "
                         "what a whole-patch display (such as flashing) gives; it is a result, not a failed test.")
    for c in s.get('colour_classes', []):
        lines.append(f"  colour class {c['colour_class']}: {c['chromatophores_measured']} chromatophores measured, "
                     f"{c['events']} events, median synchrony with the patch {c['median_sync_with_patch']:.2f}")
    for d in s.get('detector_agreement', []):
        if d['rate_events'] or d['prominence_events']:
            share = f"{100 * d['share_of_rate_events']:.0f} %" if d['rate_events'] else 'n/a'
            lines.append(f"  chunk {d['chunk']} cross-check: the prominence detector finds {d['prominence_events']} events; "
                         f"{d['found_by_both']} of the {d['rate_events']} events above are found by both ({share}).")
    lines.append('Coupling shows that chromatophores change together, not why: shared motor units, a spreading wave and '
                 'mechanical pull between neighbours all give it.')
    if 'median_wave_speed_px_per_s' in s:
        sp = f"{s['median_wave_speed_px_per_s']:.0f} px/s"
        if 'median_wave_speed_mm_per_s' in s:
            sp += f" ({s['median_wave_speed_mm_per_s']:.2f} mm/s)"
        lines.append(f"Median speed of the travelling waves: {sp}.")
    return '\n'.join(lines) + '\n'


# ── Synthetic example with a known answer ───────────────────────────────────────
def make_synthetic(out, coupled=True, seed=3, flash=False):
    """An export-style folder (areas.csv, centroids.csv, info.json, truth.json) with known waves.

    Chromatophores sit on a jittered hexagonal grid (40 px apart).  Each one twitches on its own a few times a
    minute.  With coupled=True two plane waves also sweep the patch: one at 4 s going +x at 400 px/s, one at 12 s
    going towards 135 degrees at 250 px/s.  With coupled=False the same number of extra events happen at random
    times instead (the negative control: no wave, no neighbour coupling).  With flash=True there are no waves:
    every chromatophore instead flashes at 3 Hz between 5 and 15 s, nearly in phase (a whole-patch display).
    """
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fps, dur, sp = 25.0, 20.0, 40.0
    pts = [((c + 0.5 * (r % 2)) * sp + 40, r * sp * 0.866 + 40) for r in range(12) for c in range(15)]
    xy = np.array(pts) + rng.normal(0, 4, (len(pts), 2))
    n = len(xy)
    t = np.arange(int(dur * fps)) / fps
    base = rng.uniform(80, 400, n)
    truth_waves = [{'start_s': 4.0, 'speed_px_per_s': 400.0, 'direction_deg': 0.0},
                   {'start_s': 12.0, 'speed_px_per_s': 250.0, 'direction_deg': 135.0}]

    def pulse(t0, amp):
        u = np.clip(t - t0, 0, None)
        return amp * (1 - np.exp(-u / 0.06)) * np.exp(-u / 0.6) * (t >= t0)

    areas = np.zeros((len(t), n))
    for i in range(n):
        sig = np.zeros_like(t)
        for _ in range(rng.poisson(3 * dur / 60)):
            sig += pulse(rng.uniform(0.5, dur - 1), rng.uniform(0.25, 0.5))
        if flash:
            gate = (t >= 5) & (t < 15)
            sig += gate * rng.uniform(0.25, 0.4) * 0.5 * (1 - np.cos(2 * math.pi * 3.0 * (t - 5) + rng.normal(0, 0.25)))
        for w in ([] if flash else truth_waves):
            if coupled:
                d = np.deg2rad(w['direction_deg'])
                proj = xy @ np.array([math.cos(d), math.sin(d)])        # distance along the direction of travel
                t0 = w['start_s'] + (proj[i] - proj.min()) / w['speed_px_per_s']
            else:
                t0 = rng.uniform(0.5, dur - 1)
            sig += pulse(t0, rng.uniform(0.4, 0.7))
        areas[:, i] = base[i] * (1 + sig) * (1 + rng.normal(0, 0.015, len(t)))
    rows = pd.DataFrame({'chunk': 0, 'frame': np.repeat(np.arange(len(t)), n), 'orig_frame': np.repeat(np.arange(len(t)), n),
                         't_s': np.repeat(t, n), 'chromatophore': np.tile(np.arange(1, n + 1), len(t)),
                         'area': np.round(areas.ravel())})
    rows.to_csv(out / 'areas.csv', index=False)
    pd.DataFrame({'chunk': 0, 'chromatophore': np.arange(1, n + 1), 'x_px': xy[:, 0], 'y_px': xy[:, 1],
                  'territory_px': 900}).to_csv(out / 'centroids.csv', index=False, float_format='%.2f')
    info = {'dataset': str(out), 'fps': fps, 'fps_source': 'synthetic', 'n_chunks': 1, 'n_chromatophores': n,
            'n_frames': len(t), 'chunks': [{'chunk': 0, 'start_frame': 0, 'n_frames': len(t), 'n_chromatophores': n}],
            'overlay': None, 'synthetic': True, 'attrs': {'synthetic': True}}
    (out / 'info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    truth = {'coupled': coupled, 'waves': truth_waves if coupled and not flash else []}
    if flash:
        truth['flash_hz'] = 3.0
    (out / 'truth.json').write_text(json.dumps(truth, indent=2), encoding='utf-8')
    return out


def run_example(cfg, log=print):
    """Run the analysis on a coupled and an uncoupled synthetic dataset and check the answers."""
    ok = True
    for name, coupled in (('synthetic_waves', True), ('synthetic_no_coupling', False)):
        folder = make_synthetic(EXPORT_DIR / name, coupled=coupled)
        log(f'\n=== {name} ===')
        out, s = analyse(folder, cfg, log=log)
        tests = {t['measure']: t['p_value'] for t in s['coupling_tests'] if t['measure'] != 'chance_wave_size'}
        wv = pd.read_csv(out / 'waves.csv') if (out / 'waves.csv').stat().st_size > 1 else pd.DataFrame()
        tw = wv[wv['kind'] == 'travelling'] if len(wv) else wv
        if coupled:
            for w in json.loads((folder / 'truth.json').read_text())['waves']:
                near = tw[(tw['start_s'] - w['start_s']).abs() < 1.5] if len(tw) else tw
                if not len(near):
                    log(f"  CHECK FAILED: wave at {w['start_s']} s not found")
                    ok = False
                    continue
                big = near.sort_values('n_chromatophores').iloc[-1]
                d_ang = (big['direction_deg'] - w['direction_deg'] + 180) % 360 - 180
                good = abs(big['speed_px_per_s'] / w['speed_px_per_s'] - 1) < 0.2 and abs(d_ang) < 20
                ok &= good
                log(f"  wave at {w['start_s']} s: true {w['speed_px_per_s']:.0f} px/s at {w['direction_deg']:.0f} deg, "
                    f"found {big['speed_px_per_s']:.0f} px/s at {big['direction_deg']:.0f} deg on "
                    f"{int(big['n_chromatophores'])} chromatophores -> {'OK' if good else 'CHECK FAILED'}")
            good = all(p < 0.05 for p in tests.values())
        else:
            good = all(p >= 0.05 for p in tests.values()) and len(tw) == 0
        ok &= good
        log(f"  coupling p values {', '.join(f'{k} {v:.3g}' for k, v in tests.items())}; "
            f"{len(tw)} travelling wave(s) -> {'as expected' if good else 'CHECK FAILED'}")
        y = s['synchrony'][0]
        good = (y['p_value'] < 0.05) if coupled else (y['p_value'] >= 0.05 and not y['rhythm_clear'])
        ok &= good
        log(f"  whole-patch synchrony p {y['p_value']:.3g} ({'expected below' if coupled else 'expected at or above'} 0.05) "
            f"-> {'as expected' if good else 'CHECK FAILED'}")
        d = s['detector_agreement'][0]
        good = d['share_of_rate_events'] >= 0.8
        ok &= good
        log(f"  cross-check detector: {d['found_by_both']} of {d['rate_events']} events found by both "
            f"({100 * d['share_of_rate_events']:.0f} %, expected 80 % or more) -> {'as expected' if good else 'CHECK FAILED'}")
    # a whole-patch display: every chromatophore flashes together at 3 Hz
    folder = make_synthetic(EXPORT_DIR / 'synthetic_flash', flash=True)
    log('\n=== synthetic_flash ===')
    out, s = analyse(folder, cfg, log=log)
    y = s['synchrony'][0]
    good = bool(y['p_value'] < 0.05 and y['rhythm_clear'] and abs(y['rhythm_hz'] - 3.0) < 0.3 and y['phase_agreement_R'] > 0.8)
    ok &= good
    log(f"  planted: all chromatophores flashing together at 3 Hz. Found: synchrony p {y['p_value']:.3g}, rhythm "
        f"{y['rhythm_hz']:.2f} Hz, phase agreement R {y['phase_agreement_R']:.2f} -> {'as expected' if good else 'CHECK FAILED'}")
    nbp = {t['measure']: t['p_value'] for t in s['coupling_tests'] if t['measure'] != 'chance_wave_size'}
    log(f"  neighbour-test p values here ({', '.join(f'{k} {v:.3g}' for k, v in nbp.items())}) are reported, not checked: "
        f"a whole-patch display need not give a neighbour excess.")
    log('\nAll checks passed.' if ok else '\nSome checks failed.')
    return ok


# ── Command line ────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyse', aliases=['analyze'], help='analyse an imported CHROMAS dataset')
    a.add_argument('dataset', help='dataset id (folder in processed/chromas_export) or an export folder')
    a.add_argument('--config', help='toml file with an [activation] table (e.g. examples/<name>/example.toml)')
    a.add_argument('--px-per-mm', type=float)
    a.add_argument('--roi', type=float, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    a.add_argument('--out')
    e = sub.add_parser('example', help='check the method on synthetic data with a known answer')
    e.add_argument('--config')
    args = ap.parse_args(argv)
    try:
        if args.cmd in ('analyse', 'analyze'):
            cfg = load_settings(args.config, px_per_mm=args.px_per_mm, roi=list(args.roi) if args.roi else None)
            analyse(args.dataset, cfg, out_dir=args.out)
        else:
            return 0 if run_example(load_settings(args.config)) else 1
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
