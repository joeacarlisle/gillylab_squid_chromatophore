"""Score CHROMAS segmentation against hand-labelled masks: lookup table, random forest and/or neural network.

Runs in the CHROMAS environment (it needs chromas, torch for the network, scikit-learn for the forest), like
chromas_export.py:

    <chromas python> chromas_score_segmentation.py FOLDER [--lookup lookup_table.npz] [--weights model.pth ...]
                                                   [--forest rf_segmentation_model_wrapper.pkl ...]
                                                   [--classes 2] [--min-px 10] [--out scores.csv]

FOLDER holds images/ and masks/ as made by `chromas_training.py` (0 = background, 1 = chromatophore).  Use frames
that were NOT used for training, or the scores say how well the network memorised them, not how well it works.

For each method and image it reports:
    iou, dice            pixel overlap with the hand mask (1 = perfect)
    precision, recall    share of predicted chromatophore pixels that are right / of true ones that were found
    n_true, n_found      chromatophores (connected regions of at least --min-px pixels) in the hand mask, and how
                         many of them overlap a predicted chromatophore
    n_extra              predicted regions that overlap no true chromatophore (debris, electrode, background)
and a mean per method at the end.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def load_pairs(folder):
    folder = Path(folder)
    pairs = []
    for img in sorted((folder / 'images').glob('*.png')):
        m = folder / 'masks' / img.name
        if not m.exists():
            print(f'{img.name}: no mask, skipped')
            continue
        rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(m), cv2.IMREAD_UNCHANGED)
        mask = (mask if mask.ndim == 2 else mask[..., 2]) == 1
        pairs.append((img.name, rgb, mask))
    if not pairs:
        raise SystemExit(f'No image/mask pairs in {folder}')
    return pairs


def lookup_predictor(path):
    from chromas.source.segmentation.lookup_utils import segment_image_with_lookup
    table = np.load(path, allow_pickle=True)['weights'].flatten()
    return lambda rgb: segment_image_with_lookup(rgb, table) > 0


def forest_predictor(path):
    """A random forest model trained with `chromas train -m rf`; same clean-up as CHROMAS applies after predicting.

    The .pkl is a Python pickle: it can run code when loaded, so only score files you trained yourself."""
    import joblib
    from chromas.source.segmentation.segmentation_randomforest import postprocess
    wrapper = joblib.load(path)
    return lambda rgb: postprocess(np.asarray(wrapper.predict(rgb)).astype(np.uint8)) > 0


def network_predictor(weights, n_classes, arch='fcn_resnet50'):
    import torch
    from chromas.source.segmentation.neuralnet_utils import get_model
    from chromas.source.segmentation.segmentation_neuralnet import predict
    model = get_model(arch, num_classes=n_classes, weights=str(weights))
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def run(rgb):
        t = torch.from_numpy(rgb[None].astype(np.float32))
        return predict(t, model)[0] > 0                 # every class above background counts as chromatophore
    return run


def score(pred, truth, min_px):
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    n_t, lab_t, st_t, _ = cv2.connectedComponentsWithStats(truth.astype(np.uint8), connectivity=8)
    n_p, lab_p, st_p, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    true_ids = [k for k in range(1, n_t) if st_t[k, cv2.CC_STAT_AREA] >= min_px]
    pred_ids = [k for k in range(1, n_p) if st_p[k, cv2.CC_STAT_AREA] >= min_px]
    hit_true = set(np.unique(lab_t[pred & truth]))
    hit_pred = set(np.unique(lab_p[pred & truth]))
    return {'iou': tp / max(tp + fp + fn, 1), 'dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'n_true': len(true_ids), 'n_found': sum(k in hit_true for k in true_ids),
            'n_extra': sum(k not in hit_pred for k in pred_ids)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('folder')
    ap.add_argument('--lookup', help='lookup_table.npz')
    ap.add_argument('--weights', nargs='+', default=[], help='one or more .pth files (network)')
    ap.add_argument('--forest', nargs='+', default=[], help='one or more random forest models (.pkl)')
    ap.add_argument('--classes', type=int, default=2, help='classes of the network (2 for the supplied model)')
    ap.add_argument('--min-px', type=int, default=10, help='smallest region counted as a chromatophore')
    ap.add_argument('--out', help='write the scores to this CSV file')
    a = ap.parse_args(argv)
    methods = {}
    if a.lookup:
        methods[f'lookup ({Path(a.lookup).name})'] = lookup_predictor(a.lookup)
    for f in a.forest:
        methods[f'forest ({Path(f).parent.name}/{Path(f).name})'] = forest_predictor(f)
    for w in a.weights:
        methods[f'network ({Path(w).name})'] = network_predictor(w, a.classes)
    if not methods:
        raise SystemExit('Give --lookup, --forest and/or --weights.')
    rows = []
    for name, rgb, truth in load_pairs(a.folder):
        for method, fn in methods.items():
            s = score(fn(rgb), truth, a.min_px)
            rows.append({'image': name, 'method': method, **s})
            print(f"{name:32} {method:40} IoU {s['iou']:.3f}  Dice {s['dice']:.3f}  precision {s['precision']:.3f}  "
                  f"recall {s['recall']:.3f}  found {s['n_found']}/{s['n_true']}  extra {s['n_extra']}")
    print('\nMean per method:')
    for method in methods:
        r = [x for x in rows if x['method'] == method]
        m = {k: np.mean([x[k] for x in r]) for k in ('iou', 'dice', 'precision', 'recall')}
        found, true, extra = (sum(x[k] for x in r) for k in ('n_found', 'n_true', 'n_extra'))
        print(f"{method:40} IoU {m['iou']:.3f}  Dice {m['dice']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  found {found}/{true}  extra {extra}")
    if a.out:
        import csv
        with open(a.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'Scores written to {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
