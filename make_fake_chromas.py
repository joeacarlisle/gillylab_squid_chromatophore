"""Write a SYNTHETIC CHROMAS-style dataset, so the bridge and dashboard can be tried without a real CHROMAS run.

Run it in the CHROMAS environment (it needs xarray and zarr 2.x, which only that environment has):

    cd <chromas folder>
    .venv\\Scripts\\python <squid folder>\\make_fake_chromas.py output\\fake_demo.zarr

then, in the squid project's environment:

    python chromas_bridge.py import <chromas folder>\\output\\fake_demo.zarr

The data mimic the structure chromas 0.1.1 writes (group 'chunking' plus 'chunk_N' groups with 'areas',
'masterframe' and 'warped_cleanqueen'), not real chromatophores: a 6 x 10 grid of cells whose areas expand
and contract in bursts.  Two thirds of the cells flash at about 2.5 Hz in a patch of the skin, the rest breathe slowly.
The dataset carries the attribute synthetic = True, which the dashboard shows as a banner.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

FPS, N_CHUNKS, FRAMES_PER_CHUNK, GAP = 20, 2, 300, 60   # 15 s chunks with a 3 s out-of-focus gap between them
ROWS, COLS, CELL = 6, 10, 16                              # grid of chromatophores, cell size in pixels


def make(path):
    path = Path(path)
    rng = np.random.default_rng(7)
    n = ROWS * COLS
    h, w = ROWS * CELL, COLS * CELL
    label = (np.arange(h)[:, None] // CELL) * COLS + (np.arange(w)[None, :] // CELL) + 1     # labels 1..n
    base = rng.uniform(30, 60, n)
    patch = np.array([(i // COLS) < 4 and (i % COLS) >= 3 for i in range(n)])               # cells that flash
    phase = rng.uniform(0, 0.15, n)                                                          # nearly in phase

    store = zarr.DirectoryStore(str(path))
    total = N_CHUNKS * FRAMES_PER_CHUNK + (N_CHUNKS - 1) * GAP
    starts = [c * (FRAMES_PER_CHUNK + GAP) for c in range(N_CHUNKS)]
    in_focus = np.zeros(total, dtype=bool)
    for s in starts:
        in_focus[s:s + FRAMES_PER_CHUNK] = True
    xr.Dataset(
        {'focus': (['frame'], in_focus), 'focus_statistic': (['frame'], np.where(in_focus, 1.0, 0.002)),
         'chunk_paths': (['chunk'], [f'chunk_{c}.mp4' for c in range(N_CHUNKS)]),
         'chunk_times': (['chunk', 'boundary'], np.array([[s, s + FRAMES_PER_CHUNK - 1] for s in starts])),
         'chunk_durations': (['chunk'], [FRAMES_PER_CHUNK] * N_CHUNKS)},
        coords={'frame': np.arange(total), 'chunk': np.arange(N_CHUNKS)},
        attrs={'synthetic': True, 'fps': FPS, 'original_video': 'fake_demo.mp4', 'focus_threshold': 0.01}
    ).to_zarr(store, group='chunking', mode='w')

    master = rng.normal(0.5, 0.05, (h, w))
    for c in range(N_CHUNKS):
        t = np.arange(FRAMES_PER_CHUNK) / FPS
        burst = ((t > 4) & (t < 11)).astype(float)                       # flashing from 4 s to 11 s of each chunk
        areas = np.zeros((FRAMES_PER_CHUNK, n + 1))                      # column 0 is the background
        for i in range(n):
            slow = 0.05 * np.sin(2 * np.pi * 0.2 * t + i)
            fast = 0.45 * burst * np.clip(np.sin(2 * np.pi * 2.5 * (t + phase[i])), 0, None) if patch[i] else 0
            areas[:, i + 1] = base[i] * (1 + slow + fast) + rng.normal(0, 0.6, FRAMES_PER_CHUNK)
        # Written as separate datasets, like chromas does (its arrays share a group but not all dimension lengths).
        xr.Dataset({'areas': (['frame', 'chromatophore'], areas), 'masterframe': (['x', 'y'], master)}
                   ).to_zarr(store, group=f'chunk_{c}', mode='w')
        xr.Dataset({'warped_cleanqueen': (['cq', 'x', 'y'], label[None].astype(int))}
                   ).to_zarr(store, group=f'chunk_{c}', mode='a')
    print(f'Wrote synthetic CHROMAS dataset: {path} ({n} chromatophores, {N_CHUNKS} chunks, {FPS} fps)')


if __name__ == '__main__':
    make(sys.argv[1] if len(sys.argv) > 1 else 'output/fake_demo.zarr')
