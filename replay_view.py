"""Activation replay for the Activation tab: the analysed chromatophores drawn at their positions and animated over
time, next to the same moment of the video.

Each chromatophore is a dot whose colour and size follow its relative area (grey and small at rest, orange to dark
red as it expands).  When the analysis found an activation onset, a ring flashes around the dot for a moment,
coloured by the wave the onset belongs to (the same colours as the onset plot; white = an event outside any wave).
A real wave looks like a front of rings sweeping across the skin; noise looks like rings blinking at random.

Reads what the analysis wrote (processed/chromas_activation/<dataset>/events.csv and summary.json) and the CHROMAS
export (processed/chromas_export/<dataset>/: areas, centroids, overlay.png).  The video is looked for in
examples/<dataset>/ (the [video] file of its example.toml), then in the CHROMAS data folder.  CHROMAS positions are
in its registered frame, so on skin that moves a lot the dots and the video can drift apart.

Frames are drawn with OpenCV and sent to the page as JPEG pictures, which keeps the replay quick.
"""
import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import panel as pn

import chromas_activation as ca

EXAMPLES_DIR = ca.BASE_DIR / 'examples'
WAVE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2', '#bcbd22']
SINGLE_COLOR = '#b0b0b0'
VIEW_WIDTH = 560          # width of each picture on the page (pixels)
RING_S = 0.4              # how long an onset ring stays visible (s)
MAX_REL = 1.0             # relative area drawn with the darkest colour (1.0 = twice the resting area)
SPEEDS = {'1/8 speed': 0.125, '1/4 speed': 0.25, '1/2 speed': 0.5, 'real time': 1.0}
NUMBER_MODES = ['Off', 'Selected only', 'All']
LINK_FRESH_S = 0.4        # a wave link is drawn bold for this long after it fires
LINK_HOLD_S = 0.5         # and faint until this long after the last onset of its wave
# colours of the chromatophores chosen in the trace list (and of their curves in the trace plot)
CELL_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#9a6324', '#000075',
               '#808000']


def cell_colour(cells, c):
    """Hex colour of chromatophore c, by its place in the chosen list (the same in the plot and the replay)."""
    cells = [int(x) for x in cells]
    return CELL_COLORS[cells.index(int(c)) % len(CELL_COLORS)] if int(c) in cells else '#444444'


def wave_colours(ev):
    """{wave number: hex colour}; waves in numeric order through WAVE_COLORS.  Wave -1 (single events) is grey."""
    waves = sorted(int(w) for w in pd.unique(ev['wave']) if w >= 0) if len(ev) else []
    out = {w: WAVE_COLORS[i % len(WAVE_COLORS)] for i, w in enumerate(waves)}
    out[-1] = SINGLE_COLOR
    return out


def _bgr(hex_colour):
    h = hex_colour.lstrip('#')
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def find_video(dataset, info):
    """Path of the recording behind a CHROMAS dataset, or None."""
    cands = []
    toml = EXAMPLES_DIR / dataset / 'example.toml'
    if toml.exists():
        try:
            f = tomllib.loads(toml.read_text(encoding='utf-8')).get('video', {}).get('file')
            if f:
                cands.append(toml.parent / f)
        except tomllib.TOMLDecodeError:
            pass
    orig = (info.get('attrs') or {}).get('original_video')
    if orig:
        try:
            import chromas_bridge as cb
            cands.append(cb.settings()['data_dir'] / Path(orig).name)
        except Exception:          # CHROMAS not set up: only the examples folder is searched
            pass
        cands.append(ca.BASE_DIR / 'videos' / Path(orig).name)
    return next((p for p in cands if p.exists()), None)


def _colour_for(rel):
    """BGR colour for a relative area: light grey at rest -> orange at MAX_REL/2 -> dark red at MAX_REL."""
    v = float(np.clip(rel / MAX_REL, 0, 1)) if np.isfinite(rel) else 0.0
    stops = [(0.0, (205, 205, 205)), (0.5, (0, 150, 255)), (1.0, (20, 20, 170))]
    for (a, ca_), (b, cb_) in zip(stops, stops[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return tuple(int(round(x + f * (y - x))) for x, y in zip(ca_, cb_))
    return stops[-1][1]


class Replay:
    """Everything needed to draw one chunk of one dataset, frame by frame."""

    def __init__(self, dataset, chunk=None):
        res = ca.OUT_DIR / dataset
        summary = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
        cfg = {**ca.DEFAULTS, **summary.get('settings', {})}
        data = ca.load_export(ca.resolve_export(dataset))
        self.chunks = sorted(data['chunks'])
        self.chunk = chunk if chunk in data['chunks'] else self.chunks[0]
        wide, times, cent = data['chunks'][self.chunk]
        rel, _, table = ca.prepare(wide, cent, data['fps'], cfg)
        self.fps = data['fps']
        self.times = np.asarray(times, float)
        self.rel = rel.to_numpy(float)
        used = table.loc[rel.columns]
        self.ids = [int(c) for c in rel.columns]
        self.xy = used[['x_px', 'y_px']].to_numpy(float)
        self.rest = used['resting_px'].to_numpy(float)
        ev = ca_read_csv(res / 'events.csv')
        ev = ev[ev['chunk'] == self.chunk] if len(ev) else ev
        self.colours = wave_colours(ev)
        self.onsets = ev[['chromatophore', 'onset_s', 'wave']].to_numpy(float) if len(ev) else np.zeros((0, 3))
        wv = ca_read_csv(res / 'waves.csv')
        self.waves = wv[wv['chunk'] == self.chunk] if len(wv) else wv
        self.n = len(self.times)
        # wave links (wave_links.csv): an arrow from the neighbour that activated just before, drawn when the
        # second chromatophore activates and kept, fainter, until its wave is over
        self.links = []
        lk = ca_read_csv(res / 'wave_links.csv')
        if len(lk) and len(ev):
            lk = lk[lk['chunk'] == self.chunk]
            on = {(int(r.chromatophore), int(r.wave)): float(r.onset_s) for r in ev.itertuples() if r.wave >= 0}
            ends = ev[ev['wave'] >= 0].groupby('wave')['onset_s'].max().to_dict()
            for a, b, w in zip(lk['from'], lk['to'], lk['wave']):
                key = (int(b), int(w))
                if key in on:
                    self.links.append((int(a), int(b), int(w), on[key], float(ends.get(w, on[key]))))

        # region shown: the analysis region, else the chromatophores' extent with a margin
        overlay = Path(data['folder']) / data['info']['overlay'] if data['info'].get('overlay') else None
        base = cv2.imread(str(overlay)) if overlay and overlay.exists() else None
        if base is None:
            base = np.full((int(self.xy[:, 1].max()) + 40, int(self.xy[:, 0].max()) + 40, 3), 255, np.uint8)
        H, W = base.shape[:2]
        if cfg.get('roi'):
            x, y, w, h = (int(v) for v in cfg['roi'])
        else:
            pad = 30
            x, y = int(self.xy[:, 0].min()) - pad, int(self.xy[:, 1].min()) - pad
            w, h = int(self.xy[:, 0].max()) + pad - x, int(self.xy[:, 1].max()) + pad - y
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        self.box = (x, y, w, h)
        self.scale = VIEW_WIDTH / w
        size = (VIEW_WIDTH, max(1, int(round(h * self.scale))))
        crop = base[y:y + h, x:x + w]
        self.base = cv2.resize((crop * 0.25 + 255 * 0.75).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
        self.size = size
        self.pos = [(int(round((px - x) * self.scale)), int(round((py - y) * self.scale))) for px, py in self.xy]
        self.rad0 = np.sqrt(np.maximum(self.rest, 1) / np.pi) * self.scale

        self.video = find_video(dataset, data['info'])
        self._cap, self._next, self._cache = None, None, {}

    def frame_of(self, t):
        return int(np.clip(np.searchsorted(self.times, t), 0, self.n - 1))

    # ── drawing ─────────────────────────────────────────────────────────────────
    def _labels(self, img, numbers, selected, outline_r=None):
        """Write chromatophore numbers on img.  numbers: 'Off' / 'Selected only' / 'All'.  Selected chromatophores
        get a ring and their number in their colour; with 'All' the others get a small dark number."""
        if numbers == 'Off':
            return
        sel = [int(c) for c in (selected or [])]
        font = cv2.FONT_HERSHEY_SIMPLEX
        W = img.shape[1]

        def put(text, x, y, r, scale, colour, thick):
            # beside the dot (right, or left near the edge), with a white outline so it reads on any background
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = x + r + 3 if x + r + 3 + tw < W - 2 else x - r - 3 - tw
            ty = max(th + 2, y - r // 2)
            cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
            cv2.putText(img, text, (tx, ty), font, scale, colour, thick, cv2.LINE_AA)

        for k, c in enumerate(self.ids):
            x, y = self.pos[k]
            dot = outline_r[k] if outline_r is not None else int(np.clip(round(self.rad0[k]), 3, 30))
            if c in sel:
                colour = _bgr(cell_colour(sel, c))
                r = dot + 9
                cv2.circle(img, (x, y), r, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(img, (x, y), r, colour, 2, cv2.LINE_AA)
                put(f'#{c}', x, y, r, 0.55, colour, 2)
            elif numbers == 'All':
                put(str(c), x, y, dot, 0.38, (30, 30, 30), 1)

    def _draw_links(self, img, t):
        index = {c: k for k, c in enumerate(self.ids)}
        for a, b, wave, t_on, t_end in self.links:
            if a not in index or b not in index or t < t_on or t > t_end + LINK_HOLD_S:
                continue
            colour = _bgr(self.colours.get(wave, SINGLE_COLOR))
            p0, p1 = self.pos[index[a]], self.pos[index[b]]
            if t - t_on < LINK_FRESH_S:                               # just fired: bold
                cv2.arrowedLine(img, p0, p1, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.25)
                cv2.arrowedLine(img, p0, p1, colour, 3, cv2.LINE_AA, tipLength=0.25)
            else:                                                     # earlier in the same wave: faint
                faint = tuple(int(0.45 * c + 0.55 * 255) for c in colour)
                cv2.arrowedLine(img, p0, p1, faint, 1, cv2.LINE_AA, tipLength=0.2)

    def map_jpeg(self, i, numbers='Off', selected=(), links=True):
        img = self.base.copy()
        t = self.times[i]
        row = self.rel[i]
        radius = []
        for k, (p, r0) in enumerate(zip(self.pos, self.rad0)):
            v = row[k]
            r = int(np.clip(round(r0 * np.sqrt(1 + max(v, 0) if np.isfinite(v) else 1)), 2, 30))
            radius.append(r)
            cv2.circle(img, p, r, _colour_for(v), -1, cv2.LINE_AA)
            cv2.circle(img, p, r, (90, 90, 90), 1, cv2.LINE_AA)
        index = {c: k for k, c in enumerate(self.ids)}
        for c, onset, wave in self.onsets:
            age = t - onset
            if 0 <= age < RING_S and int(c) in index:
                k = index[int(c)]
                r = radius[k] + 6
                colour = (255, 255, 255) if wave < 0 else _bgr(self.colours.get(int(wave), SINGLE_COLOR))
                width = 3 if age < RING_S / 2 else 2
                if wave < 0:
                    cv2.circle(img, self.pos[k], r + 1, (60, 60, 60), width + 2, cv2.LINE_AA)
                cv2.circle(img, self.pos[k], r, colour, width, cv2.LINE_AA)
        if links:
            self._draw_links(img, t)
        self._labels(img, numbers, selected, outline_r=radius)
        label = f't = {t:5.2f} s   frame {i + 1}/{self.n}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (4, 4), (16 + tw, 32), (255, 255, 255), -1)
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def video_jpeg(self, i, numbers='Off', selected=()):
        """The video frame at the time of analysis frame i, cropped to the same region (with numbers), or None."""
        jpg = self._video_frame(i)
        if jpg is None or numbers == 'Off':
            return jpg
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self._labels(img, numbers, selected)
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _video_frame(self, i):
        if self.video is None:
            return None
        idx = max(0, int(round(self.times[i] * self.fps)))
        if idx in self._cache:
            return self._cache[idx]
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video))
            self._next = 0
        if self._next != idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        self._next = idx + 1 if ok else None
        if not ok:
            return None
        x, y, w, h = self.box
        crop = cv2.resize(frame[y:y + h, x:x + w], self.size, interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        if len(self._cache) < 2000:
            self._cache[idx] = jpg
        return jpg


def ca_read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ── The panel ───────────────────────────────────────────────────────────────────
def replay_pane(dataset, cells_sel=None):
    """Replay section for the Activation tab (built when a dataset is shown).  cells_sel: the tab's chromatophore
    list; the ones chosen there are ringed and numbered in the replay, in the colours of their curves."""
    try:
        rp = Replay(dataset)
    except Exception as exc:          # missing files: say so instead of breaking the tab
        return pn.pane.Markdown(f'*Replay not available: {exc}*')
    if rp.n < 2 or not rp.ids:
        return pn.pane.Markdown('*Replay not available: no measured chromatophores.*')
    state = {'rp': rp}

    player = pn.widgets.Player(name='Frame', start=0, end=rp.n - 1, value=0, loop_policy='loop',
                               show_loop_controls=False, width=VIEW_WIDTH)
    speed = pn.widgets.Select(name='Speed', options=list(SPEEDS), value='1/4 speed', width=130)
    numbers = pn.widgets.RadioButtonGroup(name='Numbers', options=NUMBER_MODES, value='Off', button_type='primary',
                                         button_style='outline', width=330)
    jump = pn.widgets.Select(name='Jump to', width=260)
    show_links = pn.widgets.Checkbox(name=f'Wave links ({len(rp.links)} arrows)', value=bool(rp.links),
                                     disabled=not rp.links)
    chunk_sel = pn.widgets.Select(name='Chunk', options=rp.chunks, value=rp.chunk, width=90,
                                  visible=len(rp.chunks) > 1)
    map_pane = pn.pane.JPG(rp.map_jpeg(0), width=VIEW_WIDTH)
    first = rp.video_jpeg(0)
    video_pane = pn.pane.JPG(first, width=VIEW_WIDTH) if first else pn.pane.Markdown(
        '*No video found for this dataset* (looked in `examples/<dataset>/` and the CHROMAS data folder).',
        width=VIEW_WIDTH)

    def _jump_options():
        r = state['rp']
        opts = {'(start)': 0.0}
        for _, w in r.waves.iterrows():
            opts[f"wave {int(w['wave'])} ({w['kind']}, {int(w['n_chromatophores'])} chromatophores) "
                 f"at {w['start_s']:.2f} s"] = float(w['start_s'])
        jump.options = opts
        jump.value = 0.0

    def _interval(*_):
        # milliseconds between frames; the browser cannot redraw much faster than about 20 per second
        player.interval = int(max(50, 1000 / (state['rp'].fps * SPEEDS[speed.value])))

    def _chosen():
        return list(cells_sel.value) if cells_sel is not None else []

    def _show(event=None):
        r = state['rp']
        i = int(player.value)
        map_pane.object = r.map_jpeg(i, numbers.value, _chosen(), show_links.value)
        jpg = r.video_jpeg(i, numbers.value, _chosen())
        if jpg and isinstance(video_pane, pn.pane.JPG):
            video_pane.object = jpg

    def _jump(event):
        r = state['rp']
        player.value = r.frame_of(max(0.0, float(event.new) - 0.2))

    def _chunk(event):
        state['rp'] = Replay(dataset, event.new)
        player.end = state['rp'].n - 1
        player.value = 0
        _jump_options()
        _interval()
        _show()

    _jump_options()
    _interval()
    player.param.watch(_show, 'value')
    speed.param.watch(_interval, 'value')
    jump.param.watch(_jump, 'value')
    chunk_sel.param.watch(_chunk, 'value')
    numbers.param.watch(_show, 'value')
    show_links.param.watch(_show, 'value')
    if cells_sel is not None:
        cells_sel.param.watch(_show, 'value')

    legend = pn.pane.Markdown(
        'Each dot is a chromatophore at its CHROMAS position; **colour and size follow its relative area** '
        '(grey = at rest, orange = 50% larger, dark red = twice its resting area or more). A **ring** flashes for '
        f'{RING_S:.1f} s at each activation onset, coloured by its wave as in the onset plot (white = a single event '
        'outside any wave). A real wave looks like a front of rings sweeping across; noise looks like rings blinking '
        'at random. The video on the right shows the same moment and region. *Frame rate: the dashboard draws at most '
        'about 20 frames per second, so real time skips frames on fast video.* **Numbers**: *Selected only* rings and '
        'numbers the chromatophores chosen in the list above the trace plot (in the colours of their curves), on the '
        'map and the video; *All* also numbers the rest. **Wave links**: an arrow from the neighbour that activated '
        f'just before appears, bold, as each chromatophore of a wave activates, and stays faint until the wave is over: '
        'one connected front looks like a chain growing in one direction; separate starts look like chains growing '
        'from several places at once.', sizing_mode='stretch_width')
    reset_btn = pn.widgets.Button(name='Reset replay', width=110)
    first_chunk = rp.chunk

    def _reset(_):
        # back to how the replay opened: stopped at the start, quarter speed, no numbers, wave links as at first, the
        # first chunk, and the four most active chromatophores chosen
        player.direction = 0
        if chunk_sel.value != first_chunk:
            chunk_sel.value = first_chunk
        player.value = 0
        speed.value = '1/4 speed'
        numbers.value = 'Off'
        show_links.value = bool(state['rp'].links)
        jump.value = 0.0
        if cells_sel is not None:
            cells_sel.value = list(cells_sel.options)[:4]
        _interval()
        _show()
    reset_btn.on_click(_reset)

    return pn.Column(legend, pn.Row(player, speed, jump, chunk_sel, pn.Column(pn.Spacer(height=22), reset_btn)),
                     pn.Row(pn.pane.Markdown('**Chromatophore numbers:**', margin=(14, 5, 0, 10)), numbers,
                            pn.Column(pn.Spacer(height=8), show_links)),
                     pn.Row(map_pane, video_pane), sizing_mode='stretch_width')
