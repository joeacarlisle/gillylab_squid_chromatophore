"""Export CHROMAS results (a Zarr dataset) to plain CSV/PNG/JSON files.

This script runs in the CHROMAS environment (gillylab_chromas\\.venv), because only that environment
has zarr 2.x.  It does not import anything from this project.  chromas_bridge.py starts it:

    <chromas python> chromas_export.py DATASET.zarr OUT_DIR

Files written to OUT_DIR:
    areas.csv   chunk, frame, orig_frame, t_s, chromatophore, area   (one row per chromatophore per frame)
    focus.csv   frame, t_s, in_focus, statistic                       (CHROMAS focus check of the whole video)
    centroids.csv chunk, chromatophore, x_px, y_px, territory_px      (centre of each chromatophore's territory)
    territories.png the first chunk's territory map: 16-bit label image, pixel value = chromatophore number (0 = none)
    overlay.png first frame of the first chunk with the chromatophore territories outlined (if available)
    info.json   frame rate, chunk table, counts, the dataset's attributes
    extras/     quality checks and optional CHROMAS steps, when the dataset has them (export_extras):
        coverage.png            share of frames in which each pixel was pigmented (stitching/queenframe), 0-255
        coverage_by_frame.csv   chunk, frame, coverage_pct: share of the frame segmented as chromatophore
        registration_error.csv  src_chunk, trg_chunk, value: CHROMAS's reprojection check between chunks
                                (share of the overlap that maps back within 3 px; only with 2 or more chunks)
        tracking_points.csv     chunk, n_points, n_frames, step_median_px, step_p99_px, step_max_px, drift_max_px,
                                share_steps_over_5px: the points CHROMAS followed to register each chunk.  CHROMAS
                                drops a point that moves 10 px or more between frames, so only kept points are here
        motor_unit_cells.csv    chunk, chromatophore, n_components: independent components kept for each chromatophore
                                (the elbow of its PCA), after `chromas ica -c <chunk>`
        motor_unit_components.csv  chunk, chromatophore, component, unit: the cluster (putative motor unit) of each
                                component, after `chromas analyse cluster ics -c <chunk>`
        motor_unit_slices.csv   chunk, chromatophore, slice, unit: the unit of each wedge's dominant component
                                (-1 = none: the chromatophore's ICA gave no usable result)
        colour_classes.csv      chunk, chromatophore, colour_class, share, n_px: the segmentation class most of a
                                chromatophore's pixels have (class numbers are those of the lookup table's clicks
                                or of the model; 1 when the segmentation has a single chromatophore class)
        segmentation/           sample frames: <chunk>_<frame>_image.jpg and <chunk>_<frame>_mask.png
        motion_markers.csv      chunk, chromatophore, source: small steady chromatophores CHROMAS uses as fixed points
        slices_summary.csv      chunk, chromatophore, slice, rest_px, peak_px, expansion (after `chromas slice`)
        slices_<chunk>.npz      chromatophore ids, slice areas (frame x chromatophore x slice, float16) and each
                                chromatophore's orientation angle in degrees (mean over frames); wedge j then points
                                at 265 - orientation - 10 * j degrees (x = column, y = row of the image)
        clusters_chromas.csv    chunk, chromatophore, cluster (after `chromas analyse cluster areas`)
        extras.json             what was found, with notes

    chromas_export.py DATASET.zarr OUT_DIR --extras-only   writes only extras/ (keeps the other files as they are)

How CHROMAS stores things (read from chromas 0.1.1): group 'chunking' holds the focus check and the chunk table
(chunk_times = first and last frame of each chunk in the original video); groups 'chunk_0', 'chunk_1', ...
each hold 'areas' with dims (frame, chromatophore).  Column 0 of 'areas' is the background label and is dropped.
The area of a chromatophore is the number of pixels in its largest connected segment.
Each chunk's 'cleanqueen' is a label image in the chunk's registered (master-frame) coordinates: label k is the
territory of chromatophore k, the same k as column k of 'areas'.  Its centre gives the chromatophore's position
(x = column, y = row of the image, the same orientation as overlay.png).
"""
import argparse
import json
import sys
from pathlib import Path

FALLBACK_FPS = 20.0


def _frame_rate(attrs, video_candidates):
    """Frame rate: dataset attribute 'fps', else read from the original video, else a fallback."""
    if 'fps' in attrs:
        return float(attrs['fps']), 'dataset attribute'
    try:
        import cv2
        for cand in video_candidates:
            if cand and Path(str(cand)).exists():
                cap = cv2.VideoCapture(str(cand))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return float(fps), f'video {Path(str(cand)).name}'
    except ImportError:
        pass
    return FALLBACK_FPS, 'fallback (original video not found)'


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_overlay(group, out_path):
    """Outline the chromatophore territories on the master frame.  Returns True if an image was written."""
    try:
        import cv2
        import numpy as np
        from skimage.segmentation import mark_boundaries
        master = np.asarray(group['masterframe'][...], dtype=float)
        clq = np.asarray(group['warped_cleanqueen'][...])
        clq = clq[0] if clq.ndim == 3 else clq
        if master.ndim == 3:
            master = master[0]
        master = master / max(float(master.max()), 1e-9)
        img = (mark_boundaries(master, clq.astype(int)) * 255).astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as exc:  # the picture is optional
        print(f'overlay.png not written: {exc}', file=sys.stderr)
        return False


def territory_centroids(labels):
    """Centre (x = column, y = row) and size in pixels of every label > 0 of a 2-D label image."""
    import numpy as np
    import pandas as pd
    lab = np.asarray(labels).astype(np.int64)
    if lab.ndim == 3:
        lab = lab[0]
    rows, cols = np.indices(lab.shape)
    flat = lab.ravel()
    n = int(flat.max()) + 1
    count = np.bincount(flat, minlength=n).astype(float)
    sx = np.bincount(flat, weights=cols.ravel(), minlength=n)
    sy = np.bincount(flat, weights=rows.ravel(), minlength=n)
    keep = np.nonzero(count)[0]
    keep = keep[keep > 0]
    return pd.DataFrame({'chromatophore': keep, 'x_px': sx[keep] / count[keep], 'y_px': sy[keep] / count[keep],
                         'territory_px': count[keep].astype(int)})


def export(dataset, out_dir):
    import numpy as np
    import pandas as pd
    import zarr

    dataset, out = Path(dataset), Path(out_dir)
    if not dataset.exists():
        raise SystemExit(f'Dataset not found: {dataset}')
    out.mkdir(parents=True, exist_ok=True)

    # The Zarr arrays are read directly (not through xarray): CHROMAS writes arrays of one group with
    # dimension names that can disagree in length, which xarray refuses to open as a single dataset.
    root = zarr.open(str(dataset), mode='r')
    chunk_names = sorted((k for k in root.keys() if k.startswith('chunk_') and k.split('_')[1].isdigit()),
                         key=lambda k: int(k.split('_')[1]))

    # Focus check and chunk table.
    attrs, starts, fps, fps_source = {}, {}, FALLBACK_FPS, 'fallback'
    focus_df = pd.DataFrame(columns=['frame', 't_s', 'in_focus', 'statistic'])
    if 'chunking' in root:
        ck = root['chunking']
        attrs = {k: _json_safe(v) for k, v in ck.attrs.asdict().items()}
        chunk_paths = [str(p) for p in ck['chunk_paths'][...]] if 'chunk_paths' in ck else []
        fps, fps_source = _frame_rate(attrs, [attrs.get('original_video')] + chunk_paths[:1])
        if 'chunk_times' in ck:
            ct = np.asarray(ck['chunk_times'][...])
            starts = {int(i): int(ct[i, 0]) for i in range(ct.shape[0])}
        if 'focus' in ck:
            focus = np.asarray(ck['focus'][...]).astype(bool)
            n = focus.shape[0]
            stat = np.asarray(ck['focus_statistic'][...], dtype=float) if 'focus_statistic' in ck else np.full(n, np.nan)
            focus_df = pd.DataFrame({'frame': np.arange(n), 't_s': np.arange(n) / fps, 'in_focus': focus, 'statistic': stat})

    # Areas of every chromatophore in every chunk.
    frames_out, chunk_rows, overlay_done, n_chrom, cent_out = [], [], False, 0, []
    for name in chunk_names:
        i = int(name.split('_')[1])
        group = root[name]
        if 'areas' not in group:
            continue
        areas = np.asarray(group['areas'][...], dtype=float)[:, 1:]  # column 0 = background
        n_frames, n_c = areas.shape
        start = starts.get(i, 0)
        frame = np.repeat(np.arange(n_frames), n_c)
        frames_out.append(pd.DataFrame({
            'chunk': i, 'frame': frame, 'orig_frame': frame + start, 't_s': (frame + start) / fps,
            'chromatophore': np.tile(np.arange(1, n_c + 1), n_frames), 'area': areas.ravel()}))
        chunk_rows.append({'chunk': i, 'start_frame': start, 'n_frames': int(n_frames), 'n_chromatophores': int(n_c)})
        n_chrom = max(n_chrom, n_c)
        if 'cleanqueen' in group:
            if not cent_out:
                import cv2
                lab = np.asarray(group['cleanqueen'][...]).astype(np.int64)
                lab = lab[0] if lab.ndim == 3 else lab
                lab[lab > n_c] = 0
                cv2.imwrite(str(out / 'territories.png'), lab.astype(np.uint16))
            cent = territory_centroids(group['cleanqueen'][...])
            cent.insert(0, 'chunk', i)
            cent_out.append(cent[cent['chromatophore'] <= n_c])
        if not overlay_done and 'masterframe' in group and 'warped_cleanqueen' in group:
            overlay_done = _write_overlay(group, out / 'overlay.png')
    if not frames_out:
        raise SystemExit("No chunk has 'areas'. Run the 'area' step first (chromas run does it).")

    all_areas = pd.concat(frames_out, ignore_index=True)
    all_areas.to_csv(out / 'areas.csv', index=False)
    focus_df.to_csv(out / 'focus.csv', index=False)
    if cent_out:
        pd.concat(cent_out, ignore_index=True).to_csv(out / 'centroids.csv', index=False, float_format='%.2f')
    else:
        print('centroids.csv not written: no chunk has a cleanqueen label image', file=sys.stderr)
    info = {'dataset': str(dataset), 'fps': fps, 'fps_source': fps_source, 'n_chunks': len(chunk_rows),
            'n_chromatophores': int(n_chrom), 'n_frames': int(all_areas[['chunk', 'frame']].drop_duplicates().shape[0]),
            'chunks': chunk_rows, 'centroids': 'centroids.csv' if cent_out else None, 'overlay': 'overlay.png' if overlay_done else None,
            'synthetic': bool(attrs.get('synthetic', False)), 'attrs': attrs}
    (out / 'info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    n_measured = int(all_areas[all_areas['area'] > 0]['chromatophore'].nunique())
    print(f"Exported {len(all_areas)} area values ({n_measured} chromatophores measured of {info['n_chromatophores']} numbered, "
          f"{info['n_chunks']} chunks, {fps:g} fps from {fps_source}) to {out}")


# ── Extras: quality checks and optional CHROMAS steps ───────────────────────────────────────────────────────────
N_SEG_SAMPLES = 12          # segmentation check: frames per chunk
SEG_MAX_WIDTH = 900         # sample pictures are scaled down to this width


def _chunk_video(path_str, dataset):
    """Find a chunk video named in the dataset (paths are stored as CHROMAS was given them, often relative)."""
    p = Path(path_str)
    for cand in (p, dataset.parent / p, dataset.parent.parent / p, dataset.parent / p.name):
        if cand.exists():
            return cand
    return None


def export_extras(dataset, out_dir, log=print):
    """Write extras/ (see the module docstring).  Each part is optional; a part that fails is noted, not fatal."""
    import cv2
    import numpy as np
    import pandas as pd
    import zarr

    dataset, out = Path(dataset), Path(out_dir) / 'extras'
    out.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(dataset), mode='r')
    chunk_names = sorted((k for k in root.keys() if k.startswith('chunk_') and k.split('_')[1].isdigit()),
                         key=lambda k: int(k.split('_')[1]))
    found, notes = {}, {}

    def part(name, fn):
        try:
            found[name] = bool(fn())
        except Exception as exc:                       # one missing part must not stop the others
            found[name] = False
            notes[name] = f'{type(exc).__name__}: {exc}'
            log(f'extras: {name} not written ({exc})')

    def coverage():
        ok = False
        st = root['stitching'] if 'stitching' in root else None
        if st is not None and 'queenframe' in st:
            q = np.nan_to_num(np.asarray(st['queenframe'][...], dtype=float))
            cv2.imwrite(str(out / 'coverage.png'), np.clip(q * 255, 0, 255).astype(np.uint8))
            ok = True
        rows = []
        for name in chunk_names:
            g = root[name]
            if 'segmentation' not in g:
                continue
            seg = g['segmentation']
            for f0 in range(0, seg.shape[0], 32):          # read in blocks to keep memory small
                block = np.asarray(seg[f0:f0 + 32]) > 0
                for k, fr in enumerate(block):
                    rows.append((int(name.split('_')[1]), f0 + k, 100.0 * float(fr.mean())))
        if rows:
            pd.DataFrame(rows, columns=['chunk', 'frame', 'coverage_pct']).to_csv(out / 'coverage_by_frame.csv',
                                                                                  index=False, float_format='%.3f')
            ok = True
        return ok

    def registration():
        st = root['stitching'] if 'stitching' in root else None
        if st is None or 'reprojection_error' not in st:
            notes['registration'] = 'no reprojection check in the dataset (it needs two or more chunks)'
            return False
        m = np.asarray(st['reprojection_error'][...], dtype=float)
        i, j = np.indices(m.shape)
        pd.DataFrame({'src_chunk': i.ravel(), 'trg_chunk': j.ravel(), 'value': m.ravel()}).to_csv(
            out / 'registration_error.csv', index=False, float_format='%.4f')
        return True

    def tracking():
        rows = []
        for name in chunk_names:
            g = root[name]
            if 'points_trajectories' not in g:
                continue
            a = np.asarray(g['points_trajectories'][...], dtype=float)           # frame x point x (x, y)
            if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] < 1:
                continue
            step = np.linalg.norm(np.diff(a, axis=0), axis=-1)
            step = step[np.isfinite(step)]
            drift = np.linalg.norm(a - a[0], axis=-1)
            rows.append((int(name.split('_')[1]), a.shape[1], a.shape[0], float(np.median(step)),
                         float(np.percentile(step, 99)), float(step.max()), float(np.nanmax(drift)),
                         float(np.mean(step > 5))))
        if rows:
            pd.DataFrame(rows, columns=['chunk', 'n_points', 'n_frames', 'step_median_px', 'step_p99_px', 'step_max_px',
                                        'drift_max_px', 'share_steps_over_5px']).to_csv(
                out / 'tracking_points.csv', index=False, float_format='%.4g')
        else:
            notes['tracking'] = 'no tracked points in the dataset (registration has not run)'
        return bool(rows)

    def segmentation():
        paths = [str(p) for p in root['chunking']['chunk_paths'][...]] if 'chunking' in root and \
            'chunk_paths' in root['chunking'] else []
        sd = out / 'segmentation'
        sd.mkdir(exist_ok=True)
        n = 0
        for name in chunk_names:
            g = root[name]
            i = int(name.split('_')[1])
            if 'segmentation' not in g:
                continue
            seg = g['segmentation']
            video = _chunk_video(paths[i], dataset) if i < len(paths) else None
            cap = cv2.VideoCapture(str(video)) if video else None
            for fr in np.unique(np.linspace(0, seg.shape[0] - 1, N_SEG_SAMPLES).round().astype(int)):
                mask = np.asarray(seg[int(fr)])
                scale = min(1.0, SEG_MAX_WIDTH / mask.shape[1])
                size = (int(mask.shape[1] * scale), int(mask.shape[0] * scale))
                cv2.imwrite(str(sd / f'{i}_{fr:05d}_mask.png'),
                            cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST))
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
                    ok, img = cap.read()
                    if ok:
                        cv2.imwrite(str(sd / f'{i}_{fr:05d}_image.jpg'), cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                                    [cv2.IMWRITE_JPEG_QUALITY, 88])
                n += 1
            if cap is None:
                notes['segmentation'] = 'chunk video not found: masks only'
        return n > 0

    def motion_markers():
        rows = []
        for name in chunk_names:
            g = root[name]
            i = int(name.split('_')[1])
            if 'motion_markers' in g:
                rows += [(i, int(c), 'slice step') for c in np.asarray(g['motion_markers'][...])]
            elif all(k in g for k in ('areas', 'masterframe', 'warped_cleanqueen')):
                from chromas.source.slicing.slicing import identify_motion_marker
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mm = identify_motion_marker(np.asarray(g['areas'][...]), np.asarray(g['masterframe'][...]),
                                                np.asarray(g['warped_cleanqueen'][0]))
                rows += [(i, int(c), 'detected at export (CHROMAS defaults)') for c in mm]
        if rows:
            pd.DataFrame(rows, columns=['chunk', 'chromatophore', 'source']).to_csv(out / 'motion_markers.csv', index=False)
        return bool(rows)

    def slices():
        rows, n = [], 0
        for name in chunk_names:
            g = root[name]
            i = int(name.split('_')[1])
            if 'slice_areas' not in g or 'non_motion_markers' not in g:
                continue
            sa = np.asarray(g['slice_areas'][...], dtype=np.float32)            # frame x chromatophore x slice
            ids = np.asarray(g['non_motion_markers'][...]).astype(int)
            extra = {}
            if 'orientation_angles' in g:           # wedge 0 is measured from this direction, not from the image's x axis
                rad = np.deg2rad(np.asarray(g['orientation_angles'][...], dtype=np.float64))
                extra['orientation'] = np.degrees(np.arctan2(np.nanmean(np.sin(rad), axis=0),
                                                             np.nanmean(np.cos(rad), axis=0))).astype(np.float32)
            np.savez_compressed(out / f'slices_{i}.npz', ids=ids, slice_areas=sa.astype(np.float16), **extra)
            rest = np.nanpercentile(sa, 10, axis=0)
            peak = np.nanpercentile(sa, 95, axis=0)
            for k, c in enumerate(ids):
                for s_ in range(sa.shape[2]):
                    r, p = float(rest[k, s_]), float(peak[k, s_])
                    rows.append((i, int(c), s_, r, p, (p / r - 1) if r > 0 else np.nan))
            n += 1
        if rows:
            pd.DataFrame(rows, columns=['chunk', 'chromatophore', 'slice', 'rest_px', 'peak_px', 'expansion']).to_csv(
                out / 'slices_summary.csv', index=False, float_format='%.3f')
        else:
            notes['slices'] = 'no slice areas in the dataset: run `chromas slice` on it'
        return n > 0

    def motor_units():
        cells, comps, wedges = [], [], []

        def _i(x, bad=-1):
            # CHROMAS leaves NaN for chromatophores the ICA could not analyse
            try:
                x = float(x)
            except (TypeError, ValueError):
                return bad
            return bad if x != x else int(x)

        for name in chunk_names:
            i = int(name.split('_')[1])
            gname = f'ica_chunk_{i}'
            if gname not in root or 'elbow_point' not in root[gname]:
                continue
            g = root[gname]
            ids = np.asarray(g['chrom_id'][...]).astype(int)
            cells += [(i, int(c), _i(k, 0)) for c, k in zip(ids, np.asarray(g['elbow_point'][...]))]
            if 'IC_labels' in g and 'slice_labels' in g:
                ids2 = np.asarray(g['chroms2analyse'][...]).astype(int) if 'chroms2analyse' in g else ids
                il, sl = np.asarray(g['IC_labels'][...]), np.asarray(g['slice_labels'][...])
                for r, c in enumerate(ids2):
                    comps += [(i, int(c), k, _i(u)) for k, u in enumerate(il[r]) if _i(u) >= 0]
                    wedges += [(i, int(c), k, _i(u)) for k, u in enumerate(sl[r])]
        if cells:
            pd.DataFrame(cells, columns=['chunk', 'chromatophore', 'n_components']).to_csv(
                out / 'motor_unit_cells.csv', index=False)
        if comps:
            pd.DataFrame(comps, columns=['chunk', 'chromatophore', 'component', 'unit']).to_csv(
                out / 'motor_unit_components.csv', index=False)
            pd.DataFrame(wedges, columns=['chunk', 'chromatophore', 'slice', 'unit']).to_csv(
                out / 'motor_unit_slices.csv', index=False)
        if not cells:
            notes['motor_units'] = 'no ICA in the dataset: run the motor-unit analysis (after the slice step)'
        elif not comps:
            notes['motor_units'] = 'ICA found, but its components are not clustered yet'
        return bool(comps)

    def colour_classes():
        rows = []
        for name in chunk_names:
            g = root[name]
            if 'segmentation' not in g or 'warped_cleanqueen' not in g:
                continue
            seg, wcq = g['segmentation'], g['warped_cleanqueen']
            n = min(seg.shape[0], wcq.shape[0])
            tally = {}
            for fr in np.unique(np.linspace(0, n - 1, 9).round().astype(int)):
                s, w = np.asarray(seg[int(fr)]), np.asarray(wcq[int(fr)])
                if s.shape != w.shape:
                    continue
                m = (s > 0) & (w > 0)
                keys, cnt = np.unique(w[m].astype(np.int64) * 256 + s[m], return_counts=True)
                for k, c in zip(keys, cnt):
                    tally.setdefault(int(k // 256), {}).setdefault(int(k % 256), 0)
                    tally[int(k // 256)][int(k % 256)] += int(c)
            for cid, d in sorted(tally.items()):
                tot = sum(d.values())
                cls = max(d, key=d.get)
                rows.append((int(name.split('_')[1]), cid, cls, d[cls] / tot, tot))
        if rows:
            df = pd.DataFrame(rows, columns=['chunk', 'chromatophore', 'colour_class', 'share', 'n_px'])
            df.to_csv(out / 'colour_classes.csv', index=False, float_format='%.3f')
            if df['colour_class'].nunique() < 2:
                notes['colour_classes'] = 'the segmentation has one chromatophore class (no colour classes)'
        return bool(rows)

    def clusters():
        rows = []
        for name in chunk_names:
            g = root[name]
            if 'area_clustering' in g and 'labels' in g['area_clustering']:
                lab = np.asarray(g['area_clustering']['labels'][...])
                rows += [(int(name.split('_')[1]), k + 1, int(v)) for k, v in enumerate(lab[1:])]   # 0 = background
        if not rows and 'area_clustering' in root and 'labels' in root['area_clustering']:
            lab = np.asarray(root['area_clustering']['labels'][...])
            rows = [(-1, k + 1, int(v)) for k, v in enumerate(lab[1:])]
        if rows:
            pd.DataFrame(rows, columns=['chunk', 'chromatophore', 'cluster']).to_csv(out / 'clusters_chromas.csv', index=False)
        else:
            notes['clusters'] = 'no CHROMAS clustering in the dataset (`chromas analyse cluster areas`)'
        return bool(rows)

    for name, fn in (('coverage', coverage), ('registration', registration), ('tracking', tracking), ('segmentation', segmentation),
                     ('motion_markers', motion_markers), ('slices', slices), ('clusters', clusters),
                     ('motor_units', motor_units), ('colour_classes', colour_classes)):
        part(name, fn)
    (out / 'extras.json').write_text(json.dumps({'found': found, 'notes': notes}, indent=2), encoding='utf-8')
    log('Extras: ' + ', '.join(f'{k} {"yes" if v else "no"}' for k, v in found.items()))
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dataset', help='CHROMAS dataset (.zarr folder)')
    ap.add_argument('out_dir', help='folder for areas.csv, focus.csv, centroids.csv, overlay.png, info.json')
    ap.add_argument('--extras-only', action='store_true', help='write only extras/ (quality checks, slices, ...)')
    a = ap.parse_args(argv)
    if not a.extras_only:
        export(a.dataset, a.out_dir)
    export_extras(a.dataset, a.out_dir)


if __name__ == '__main__':
    main()
