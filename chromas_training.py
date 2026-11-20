"""Prepare hand-labelled training data for the CHROMAS segmentation network.

Runs in this project's environment.  Three steps (see docs/TRAINING_THE_NETWORK.md):

    python chromas_training.py frames VIDEO OUT_DIR [--n 10] [--times 2.5 7 15]
        Export frames as PNG into OUT_DIR/images/ (spread over the video, or at the given times in seconds).
        Each file is named <video>_f<frame>.png.

    python chromas_training.py draft OUT_DIR --lookup path/to/lookup_table.npz
        Optional head start: write a draft paint layer for every image into OUT_DIR/painted/, made with a CHROMAS
        lookup table (red = chromatophore, transparent = background).  Correcting a draft is much faster than
        painting from nothing; correct it by hand before the next step.

    python chromas_training.py masks OUT_DIR
        Convert painted masks in OUT_DIR/painted/ into the format CHROMAS trains on, in OUT_DIR/masks/:
        single-channel PNG, 0 = background, 1 = chromatophore.  A painted mask is the paint layer exported on its
        own (same name and size as the image): every pixel that is painted (not transparent and not black) becomes 1.

    python chromas_training.py crops OUT_DIR [--size 512] [--per-frame 1]
        Cut small crops of the most chromatophore-rich parts of each image and mask into OUT_DIR_crops/ (a random
        forest uses every pixel of every training image, so full 1920 x 1080 frames are too big to train on).

    python chromas_training.py check OUT_DIR
        Check that images and masks pair up, have the same size, are at least 512 px on each side (CHROMAS cuts
        512 x 512 windows), and that the masks hold only 0 and 1.  Prints how much of each frame is chromatophore.

The masks format was read from CHROMAS 0.1.1 (training/neural_net_training.py takes the first channel of the mask
as the class number) and from the binary training set in Chromas2025.zip, whose masks are exactly that.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_SIDE = 512                      # CHROMAS training windows are 512 x 512


def export_frames(video, out_dir, n=10, times=None):
    video, out = Path(video), Path(out_dir) / 'images'
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if times:
        idx = sorted({min(total - 1, max(0, int(round(t * fps)))) for t in times})
    else:
        idx = sorted({int(round(x)) for x in np.linspace(0, total - 1, n + 2)[1:-1]})
    written = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            print(f'frame {i}: could not be read, skipped')
            continue
        p = out / f'{video.stem}_f{i:05d}.png'
        cv2.imwrite(str(p), frame)
        written.append(p)
        print(f'{p.name}  ({i / fps:.2f} s)')
    cap.release()
    h, w = frame.shape[:2] if written else (0, 0)
    if written and min(h, w) < MIN_SIDE:
        print(f'WARNING: frames are {w} x {h}; CHROMAS needs at least {MIN_SIDE} px on each side.')
    print(f'{len(written)} frame(s) in {out}. Paint the chromatophores of each one (docs/TRAINING_THE_NETWORK.md).')
    return written


def draft_masks(out_dir, lookup):
    """Draft paint layers from a CHROMAS lookup table (same rule as CHROMAS: label > 0 = chromatophore)."""
    out_dir = Path(out_dir)
    images, painted = out_dir / 'images', out_dir / 'painted'
    painted.mkdir(exist_ok=True)
    table = np.load(lookup, allow_pickle=True)['weights'].flatten()
    for p in sorted(images.glob('*.png')):
        target = painted / p.name
        if target.exists():
            print(f'{p.name}: already has a paint layer, left as it is')
            continue
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        q = (rgb >> 4).astype(np.uint16)                      # CHROMAS quantises each channel to 16 levels
        on = table[q[..., 0] << 8 | q[..., 1] << 4 | q[..., 2]] > 0
        layer = np.zeros((*on.shape, 4), np.uint8)
        layer[on] = (0, 0, 255, 255)                          # red, opaque (BGRA)
        cv2.imwrite(str(target), layer)
        print(f'{p.name}: draft {100 * on.mean():.1f} % chromatophore -> {target}')
    print('Open each image with its draft as a layer, correct the draft (erase debris and the electrode, fill missed '
          'chromatophores), and export the layer over the draft. Then run the masks step.')


def convert_masks(out_dir):
    out_dir = Path(out_dir)
    painted, masks, images = out_dir / 'painted', out_dir / 'masks', out_dir / 'images'
    if not painted.is_dir():
        raise SystemExit(f'{painted} not found: export each paint layer there as <same name as the image>.png')
    masks.mkdir(exist_ok=True)
    n = 0
    for p in sorted(painted.glob('*.png')):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f'{p.name}: not readable, skipped')
            continue
        if m.ndim == 2:
            on = m > 0
        elif m.shape[2] == 4:
            on = (m[..., 3] > 127) & (m[..., :3].max(axis=2) > 0)
        else:
            on = m.max(axis=2) > 0
        img = images / p.name
        if img.exists():
            ih, iw = cv2.imread(str(img)).shape[:2]
            if (ih, iw) != on.shape:
                print(f'{p.name}: mask is {on.shape[1]} x {on.shape[0]}, image is {iw} x {ih}; skipped')
                continue
        else:
            print(f'{p.name}: no image of the same name in {images} (converted anyway)')
        cv2.imwrite(str(masks / p.name), on.astype(np.uint8))
        print(f'{p.name}: {100 * on.mean():.1f} % chromatophore')
        n += 1
    print(f'{n} mask(s) written to {masks} (0 = background, 1 = chromatophore; they look black in a viewer).')


def make_crops(out_dir, size=512, per_frame=1):
    """Cut `per_frame` crops of size x size per image, where the mask holds the most separate chromatophores.

    Writes OUT_DIR_crops/images and OUT_DIR_crops/masks (earlier crops there are replaced); returns the folder."""
    out_dir = Path(out_dir)
    dest = out_dir.parent / (out_dir.name + '_crops')
    for sub in ('images', 'masks'):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for old in (dest / sub).glob('*.png'):
            old.unlink()
    n_written = 0
    for p in sorted((out_dir / 'images').glob('*.png')):
        mp = out_dir / 'masks' / p.name
        if not mp.exists():
            print(f'{p.name}: no mask, skipped')
            continue
        img = cv2.imread(str(p))
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        m = m if m.ndim == 2 else m[..., 2]
        h, w = m.shape
        if min(h, w) < size:
            print(f'{p.name}: smaller than {size} px, skipped')
            continue
        step = max(32, size // 8)
        found = []
        for y in range(0, h - size + 1, step):
            for x in range(0, w - size + 1, step):
                win = (m[y:y + size, x:x + size] == 1).astype(np.uint8)
                n = cv2.connectedComponents(win)[0] - 1
                if n:
                    found.append((n + 50 * float(win.mean()), x, y, n))
        found.sort(reverse=True)
        chosen = []
        for sc, x, y, n in found:
            if all(abs(x - cx) >= size or abs(y - cy) >= size for _, cx, cy, _ in chosen):
                chosen.append((sc, x, y, n))
            if len(chosen) == per_frame:
                break
        for i, (_, x, y, n) in enumerate(chosen):
            name = p.name if per_frame == 1 else f'{p.stem}_c{i}.png'
            cv2.imwrite(str(dest / 'images' / name), img[y:y + size, x:x + size])
            cv2.imwrite(str(dest / 'masks' / name), m[y:y + size, x:x + size])
            print(f'{name}: crop at x {x}, y {y}, {n} chromatophores')
            n_written += 1
    print(f'{n_written} crop(s) of {size} x {size} in {dest}')
    return dest


def check(out_dir):
    out_dir = Path(out_dir)
    images, masks = out_dir / 'images', out_dir / 'masks'
    problems = []
    names_i = {p.name for p in images.glob('*.png')} if images.is_dir() else set()
    names_m = {p.name for p in masks.glob('*.png')} if masks.is_dir() else set()
    for n in sorted(names_i - names_m):
        problems.append(f'{n}: image without a mask')
    for n in sorted(names_m - names_i):
        problems.append(f'{n}: mask without an image')
    rows = []
    for n in sorted(names_i & names_m):
        img = cv2.imread(str(images / n))
        m = cv2.imread(str(masks / n), cv2.IMREAD_UNCHANGED)
        m1 = m if m.ndim == 2 else m[..., 2]          # CHROMAS reads the first (red) channel
        if img.shape[:2] != m1.shape:
            problems.append(f'{n}: image {img.shape[1]} x {img.shape[0]}, mask {m1.shape[1]} x {m1.shape[0]}')
            continue
        if min(img.shape[:2]) < MIN_SIDE:
            problems.append(f'{n}: smaller than {MIN_SIDE} px on a side')
        vals = np.unique(m1)
        if not set(vals.tolist()) <= {0, 1}:
            problems.append(f'{n}: mask values {vals[:6].tolist()}... (must be 0 and 1; run the masks step)')
        rows.append((n, img.shape[1], img.shape[0], 100 * (m1 == 1).mean()))
    for n, w, h, pct in rows:
        print(f'{n}: {w} x {h}, {pct:.1f} % chromatophore')
    print(f'{len(rows)} pair(s).')
    if problems:
        print('Problems:\n  ' + '\n  '.join(problems))
        return 1
    print('Ready for chromas train.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('frames', help='export frames from a video as PNG')
    f.add_argument('video')
    f.add_argument('out_dir')
    f.add_argument('--n', type=int, default=10, help='number of frames, spread over the video (default 10)')
    f.add_argument('--times', type=float, nargs='+', help='times in seconds instead')
    d = sub.add_parser('draft', help='draft paint layers from a lookup table, to correct by hand')
    d.add_argument('out_dir')
    d.add_argument('--lookup', required=True, help='lookup_table.npz')
    m = sub.add_parser('masks', help='convert painted masks to 0/1 masks')
    m.add_argument('out_dir')
    k = sub.add_parser('crops', help='cut small crops of images and masks, to train a random forest on')
    k.add_argument('out_dir')
    k.add_argument('--size', type=int, default=512)
    k.add_argument('--per-frame', type=int, default=1)
    c = sub.add_parser('check', help='check images and masks before training')
    c.add_argument('out_dir')
    a = ap.parse_args(argv)
    if a.cmd == 'frames':
        export_frames(a.video, a.out_dir, a.n, a.times)
    elif a.cmd == 'draft':
        draft_masks(a.out_dir, a.lookup)
    elif a.cmd == 'masks':
        convert_masks(a.out_dir)
    elif a.cmd == 'crops':
        make_crops(a.out_dir, a.size, a.per_frame)
    else:
        return check(a.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
