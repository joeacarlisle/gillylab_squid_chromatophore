"""Interactive colour picker for CHROMAS lookup segmentation (part of the Datasets tab; used by chromas_view.py).

CHROMAS (segmentation_lookup.py) makes its lookup table from clicks in a separate window and shows the result only
afterwards.  This picker does the same in the dashboard, with a live preview: click on a video frame, see which pixels
become chromatophore after every click, and only start CHROMAS when the preview looks right.  CHROMAS itself is not
changed or imported: the picker writes the same kind of table (`weights` 16x16x16 uint8 and `color_space`, as in the
lookup_table.npz that CHROMAS saves) and CHROMAS is started with it as the model file, so its own clicking window
never opens.

How CHROMAS uses a table: the colour of a pixel, each of R, G and B divided by 16, indexes the table; label 0 is
background, a label above 0 is chromatophore (areas.py counts segmentation > 0).  CHROMAS's own click window gives every
click its own class and only the first click is background, so a second background click would count as chromatophore.
Here every Background click is class 0 and every Chromatophore click is class 1, so the background can have as many
shades as needed.  Each pixel colour takes the class of the nearest clicked colour (nearest in Lab; CHROMAS first moves
the clicked colours with k-means, so the result can differ a little at the edges).  suggest_clicks() proposes a first
set of clicks from the frame's own colours, so the picker starts from a default and the person only corrects it.
"""
from pathlib import Path

import cv2
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

import chromas_bridge as cb

hv.extension('bokeh')            # also loaded by the dashboard; repeating it is harmless

SHOW_WIDTH = 960               # frames are shown at most this wide; the colours are read from the full-size frame
KINDS = ('Background', 'Chromatophore')
VIEWS = ('Chromatophores (class > 0)', 'Class 0 outline (as CHROMAS shows it)', 'Segmented')
EDGE_COLOUR = {'Background': '#1b9e77', 'Chromatophore': '#e7298a'}


# ── pure functions (numpy only) ──────────────────────────────────────────────────────────────────
def read_frame(video, seconds):
    """A frame of a video as RGB.  `video` is a name as typed in the dashboard (cb.find_video).  Returns a dict with
    rgb (full size), small (for display), width, height, time (of the frame shown), duration, fps and path."""
    path = cb.find_video(video)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0 or n <= 0:
            raise RuntimeError(f'Could not read the frame rate or length of {path.name}.')
        frame = max(0, min(int(round(float(seconds) * fps)), n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Could not read frame {frame} of {path.name}.')
    finally:
        cap.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = rgb if w <= SHOW_WIDTH else cv2.resize(rgb, (SHOW_WIDTH, int(round(h * SHOW_WIDTH / w))),
                                                   interpolation=cv2.INTER_AREA)
    return {'rgb': rgb, 'small': small, 'width': w, 'height': h, 'time': frame / fps, 'frame': frame,
            'duration': n / fps, 'fps': float(fps), 'path': path}


def sample_colour(rgb, col, row, radius):
    """Median colour of the (2*radius+1) square around a pixel, so one noisy pixel does not set the colour."""
    h, w = rgb.shape[:2]
    patch = rgb[max(0, row - radius):min(h, row + radius + 1), max(0, col - radius):min(w, col + radius + 1)]
    return tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))


def _lab(rgb):
    """RGB uint8 array (..., 3) -> CIE Lab (L 0-100), float32.  Distances in Lab follow perceived colour difference,
    which RGB distances do not (pale mauve and grey-white are close in RGB, clearly different in Lab)."""
    arr = np.asarray(rgb, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def build_table(background, chromatophore):
    """16x16x16 uint8 lookup table: 0 where a colour is closest to a Background click, 1 where closest to a
    Chromatophore click (closest = smallest distance in Lab).  Needs at least one of each."""
    if not background or not chromatophore:
        raise ValueError('Click at least one background and one chromatophore.')
    centres = _lab(list(background) + list(chromatophore))
    is_background = np.array([True] * len(background) + [False] * len(chromatophore))
    g = (np.arange(16) * 16 + 8).astype(np.uint8)               # the middle of each of the 16 colour steps
    bins = _lab(np.stack(np.meshgrid(g, g, g, indexing='ij'), axis=-1))
    nearest = ((bins[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return np.where(is_background[nearest], 0, 1).astype(np.uint8).reshape(16, 16, 16)


def suggest_clicks(rgb, groups=8, tolerance=14.0, per_group=2, seed=0):
    """A default set of clicks, chosen from the frame's own colours.  The pixel colours are grouped by k-means (in
    Lab); the biggest group is the background, and any group within `tolerance` (Lab distance) of it is another shade
    of background (glare, haze, a halo); every other group is chromatophore.  Each group gets `per_group` clicks, on
    pixels well inside that group's colour.  Returns [{'kind', 'x', 'y'}] in the coordinates of `rgb`."""
    h, w = rgb.shape[:2]
    lab = _lab(rgb)
    rng = np.random.default_rng(seed)
    sample = lab[rng.choice(len(lab), size=min(len(lab), 80000), replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(int(seed))              # cv2.kmeans draws its start centres from OpenCV's own random generator: without this each call differs
    _, _, centres = cv2.kmeans(sample, int(groups), None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dist = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    sizes = np.bincount(assign, minlength=len(centres))
    biggest = int(sizes.argmax())
    out = []
    for c in np.argsort(-sizes):
        if sizes[c] < 0.0003 * len(lab):                       # a handful of pixels: noise
            continue
        kind = KINDS[0] if np.linalg.norm(centres[c] - centres[biggest]) <= tolerance else KINDS[1]
        mask = (assign == c).reshape(h, w).astype(np.uint8)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))      # keep pixels with a same-group neighbourhood
        if inner.sum() < per_group:
            inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        if inner.sum() < per_group:
            inner = mask
        ys, xs = np.nonzero(inner)
        d = dist.reshape(h, w, -1)[ys, xs, c]
        best = np.argsort(d)[:60]
        picks = [best[0]]
        while len(picks) < per_group and len(best) > len(picks):   # the next click: far from the ones already taken
            far = [(min((xs[b] - xs[p]) ** 2 + (ys[b] - ys[p]) ** 2 for p in picks), b) for b in best if b not in picks]
            picks.append(max(far)[1])
        for b in picks:
            out.append({'kind': kind, 'x': int(xs[b]), 'y': int(ys[b]), 'group': int(c), 'share': float(sizes[c] / len(lab))})
    return out


def classify(rgb, table):
    """Class (0 or 1) of every pixel of an RGB image, the way CHROMAS applies a table (segment_image_with_lookup)."""
    q = (rgb >> 4).astype(np.uint16)
    return table.reshape(-1)[q[..., 0] << 8 | q[..., 1] << 4 | q[..., 2]]


def outline(mask, thickness=1):
    """Boundary pixels of a boolean mask (as a boolean array)."""
    m = mask.astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((thickness, thickness), np.uint8))
    return edge.astype(bool)


def preview_image(rgb, table, view):
    """The right-hand picture: the frame with the result of the table drawn on it."""
    if table is None:
        return rgb
    cls = classify(rgb, table)
    if view == VIEWS[0]:                                        # chromatophore pixels in magenta
        out = rgb.astype(np.float32)
        out[cls > 0] = 0.4 * out[cls > 0] + 0.6 * np.array([255, 0, 255], np.float32)
        return out.astype(np.uint8)
    if view == VIEWS[1]:                                        # what CHROMAS draws: the outline of class 0
        out = rgb.copy()
        out[outline(cls == 0, 2)] = (0, 255, 0)
        return out
    out = np.empty_like(rgb)                                    # the two classes as flat colours
    out[cls == 0] = (225, 235, 245)
    out[cls > 0] = (70, 20, 90)
    return out


def save_table(path, table):
    """Write a table in the form CHROMAS reads (it loads `weights` and `color_space`)."""
    np.savez(str(path), weights=table, color_space=np.array('RGB'))


# ── the dashboard part ───────────────────────────────────────────────────────────────────────────
class ColourPicker(param.Parameterized):
    """`video_in` is the form's video box; `on_run(table_path)` starts CHROMAS with the saved table and returns an
    error message, or None when the run started."""
    version = param.Integer(0)                                  # changes whenever the frame or the clicks change

    def __init__(self, video_in, on_run, name_in=None, **kw):
        super().__init__(**kw)
        self.video_in, self.on_run, self.name_in = video_in, on_run, name_in      # name_in: the form's dataset name box
        self.clicks = []                                        # dicts: kind, time, x, y, rgb
        self.frame = None
        self.table = None
        self._tap = None

        self.load_btn = pn.widgets.Button(name='Load video', button_type='primary', width=130)
        self.slider = pn.widgets.FloatSlider(name='Frame time (s)', start=0, end=1, step=0.04, value=0, width=520,
                                             disabled=True)
        self.kind = pn.widgets.RadioButtonGroup(name='Next click is', options=list(KINDS), value=KINDS[0],
                                                button_type='success', width=300)
        self.radius = pn.widgets.IntSlider(name='Colour from a square of (2 x radius + 1) pixels', start=0, end=8,
                                           value=2, width=300)
        self.view = pn.widgets.RadioButtonGroup(name='Preview', options=list(VIEWS), value=VIEWS[0], width=640)
        self.suggest_btn = pn.widgets.Button(name='Suggest colours (replaces clicks)', width=240)
        self.groups = pn.widgets.IntSlider(name='Colour groups', start=4, end=14, value=8, width=180)
        self.tolerance = pn.widgets.FloatSlider(name='Background tolerance', start=4, end=30, step=1, value=14, width=200)
        self.prev_btn = pn.widgets.Button(name='\u25c0 Previous suggestion', width=200, disabled=True)
        self.next_btn = pn.widgets.Button(name='Next suggestion \u25b6', width=200, disabled=True)
        self.hist_label = pn.pane.Markdown('', width=420)
        self.hist, self.hist_i = [], -1         # suggestions tried on the frame, so you can go back and forth between them
        self.undo_btn = pn.widgets.Button(name='Undo last click', width=130)
        self.remove_btn = pn.widgets.Button(name='Remove ticked', width=130)
        self.clear_btn = pn.widgets.Button(name='Clear all', width=100)
        self.list = pn.widgets.Tabulator(self._clicks_df(), theme='midnight', show_index=False, disabled=True,
                                         selectable='checkbox', height=190, width=520)
        self.msg = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
        self.stats = pn.pane.Markdown('', width=640)
        self.run_btn = pn.widgets.Button(name='Use these colours and run CHROMAS', button_type='primary', disabled=True,
                                         width=300)
        self.plot = pn.Column(pn.pane.Markdown('*Choose or type the video in the Video box above, then press **Load video**.*'))

        self.load_btn.on_click(self._load)
        self.suggest_btn.on_click(lambda _: self._suggest())
        self.prev_btn.on_click(self._prev)
        self.next_btn.on_click(self._next)
        self.slider.param.watch(self._on_slider, 'value_throttled')
        self.undo_btn.on_click(self._undo)
        self.remove_btn.on_click(self._remove)
        self.clear_btn.on_click(self._clear)
        self.run_btn.on_click(self._run)
        self.view.param.watch(lambda *_: self._touch(), 'value')
        self.list.param.watch(lambda *_: self._touch(), 'selection')     # ticking a row rings that click on the frame

        self.panel = pn.Column(
            pn.pane.Markdown(
                '**Click the colours yourself, check the preview, then run.** Press **Load video**, then click on the frame on '
                'the left. Or press **Suggest colours** (it replaces your clicks): the frame\'s colours are sorted into '
                'groups; the biggest group is the background, groups close to it (glare, haze, a halo) are more '
                'background, and the rest are chromatophore. The picture on the right shows the result. Correct it by '
                'clicking on the frame on the left: mark **Background** wherever chromatophore '
                'colour has spread into the background, and **Chromatophore** on any pigment that is missing, in its dark '
                'centre, not its soft edge. More **Colour groups** finds finer colours; a higher **Background tolerance** '
                'counts more of the groups as background. Move the slider to check other frames before you run: the table '
                'is made from the colours, wherever they were clicked.', sizing_mode='stretch_width'),
            pn.Row(self.load_btn, self.slider),
            pn.Row(self.suggest_btn, self.groups, self.tolerance),
            pn.Row(self.prev_btn, self.next_btn, self.hist_label),
            pn.Row(self.kind, pn.Spacer(width=20), self.radius),
            self.view, self.msg, self.plot, self.stats,
            pn.Row(self.undo_btn, self.remove_btn, self.clear_btn),
            self.list, self.run_btn,
            sizing_mode='stretch_width')

    # clicks ----------------------------------------------------------------------------------
    def _clicks_df(self):
        rows = [(i + 1, c['kind'], c.get('by', 'you'), round(c['time'], 2), c['x'], c['y'], *c['rgb'])
                for i, c in enumerate(self.clicks)]
        return pd.DataFrame(rows, columns=['#', 'kind', 'by', 'time (s)', 'x', 'y', 'R', 'G', 'B'])

    def _on_tap(self, x, y):
        if self.frame is None or x is None or y is None:
            return
        w, h = self.frame['width'], self.frame['height']
        col, row = int(np.clip(x, 0, w - 1)), int(np.clip(h - y, 0, h - 1))      # y points up in the plot
        self.clicks.append({'kind': self.kind.value, 'time': self.frame['time'], 'x': col, 'y': row, 'by': 'you',
                            'rgb': sample_colour(self.frame['rgb'], col, row, self.radius.value)})
        self._changed()

    def _undo(self, _):
        if self.clicks:
            self.clicks.pop()
            self._changed()

    def _remove(self, _):
        drop = set(self.list.selection)
        if drop:
            self.clicks = [c for i, c in enumerate(self.clicks) if i not in drop]
            self.list.selection = []
            self._changed()

    def _clear(self, _):
        self.clicks = []
        self.list.selection = []
        self._changed()

    def _changed(self):
        bg = [c['rgb'] for c in self.clicks if c['kind'] == KINDS[0]]
        ch = [c['rgb'] for c in self.clicks if c['kind'] == KINDS[1]]
        self.table = build_table(bg, ch) if bg and ch else None
        self.list.value = self._clicks_df()
        self.run_btn.disabled = self.table is None
        self._touch()

    def _touch(self):
        self._update_stats()
        self.version += 1

    def _update_stats(self):
        nb = sum(c['kind'] == KINDS[0] for c in self.clicks)
        nc = len(self.clicks) - nb
        text = f'{nb} background and {nc} chromatophore clicks.'
        if self.frame is not None and self.table is not None:
            pct = 100.0 * float((classify(self.frame['small'], self.table) > 0).mean())
            text += f' On this frame **{pct:.1f}%** of the pixels are chromatophore (class > 0).'
            if pct < 0.2:
                text += ' Far too little: CHROMAS would find no chromatophores.'
            elif pct > cb.LOOKUP_SANE_LIMIT:
                text += ' Far too much: the chromatophores would merge into huge patches. Is some background marked as chromatophore?'
        elif self.frame is not None:
            text += ' The preview starts when there is at least one of each.'
        self.stats.object = text

    # frame -----------------------------------------------------------------------------------
    def _show(self, text, kind='info'):
        self.msg.object, self.msg.alert_type, self.msg.visible = text, kind, bool(text)

    def _load(self, _):
        try:
            self.frame = read_frame(self.video_in.value.strip(), self.slider.value)
        except Exception as exc:
            self._show(f'Could not load the video: {exc}', 'danger')
            return
        self._show('')
        f = self.frame
        self.slider.disabled = False
        self.slider.end = max(f['duration'] - 1.0 / f['fps'], 0.1)
        self.slider.step = round(1.0 / f['fps'], 4)
        self._build_plot()
        self._changed()
        if not self.clicks:
            self._show('Click the frame on the left: **Background** first, then at least one **Chromatophore**, and the '
                       'preview appears. Or press **Suggest colours** to start from colours found in the frame '
                       '(it replaces any clicks you have made).', 'info')

    def _keep_clicks(self):
        """Store the clicks as they are now (with your own changes) in the suggestion being shown."""
        if 0 <= self.hist_i < len(self.hist):
            self.hist[self.hist_i]['clicks'] = [dict(c) for c in self.clicks]

    def _hist_update(self):
        n = len(self.hist)
        self.prev_btn.disabled = self.hist_i <= 0
        self.next_btn.disabled = self.frame is None
        if 0 <= self.hist_i < n:
            e = self.hist[self.hist_i]
            self.hist_label.object = (f'**Suggestion {self.hist_i + 1} of {n}** (seed {e["seed"]}, {e["groups"]} colour groups, '
                                      f'background tolerance {e["tolerance"]:g}). Your own clicks on it are kept when you move '
                                      'away and come back.')
        else:
            self.hist_label.object = ''

    def _goto(self, i):
        self._keep_clicks()
        self.hist_i = i
        e = self.hist[i]
        self.clicks = [dict(c) for c in e['clicks']]
        self.groups.value, self.tolerance.value = e['groups'], e['tolerance']
        self.list.selection = []
        self._changed()
        self._hist_update()

    def _prev(self, _):
        if self.hist_i > 0:
            self._goto(self.hist_i - 1)

    def _next(self, _):
        if self.frame is None:
            return
        if self.hist_i < len(self.hist) - 1:
            self._goto(self.hist_i + 1)
        else:                                   # at the newest one: try another, with a new random start
            self._suggest(seed=int(np.random.default_rng().integers(1, 1_000_000)), new_history=False)

    def _suggest(self, seed=0, new_history=True):
        """Replace the clicks with a default chosen from the colours of the frame shown.  The same seed, groups and
        tolerance always give the same clicks; 'Next suggestion' tries another seed and keeps the earlier ones."""
        if self.frame is None:
            self._show('Press **Load video** first: colours are suggested from the frame that is shown.', 'warning')
            return
        f = self.frame
        small = f['small']
        scale = f['width'] / small.shape[1]
        try:
            picks = suggest_clicks(small, self.groups.value, self.tolerance.value, seed=seed)
        except Exception as exc:
            self._show(f'Could not suggest colours: {exc}', 'danger')
            return
        self._keep_clicks()
        self.clicks = []
        for p in picks:
            col = min(int(p['x'] * scale), f['width'] - 1)
            row = min(int(p['y'] * scale), f['height'] - 1)
            self.clicks.append({'kind': p['kind'], 'time': f['time'], 'x': col, 'y': row, 'by': 'auto',
                                'rgb': sample_colour(f['rgb'], col, row, self.radius.value)})
        self.list.selection = []
        entry = {'seed': seed, 'groups': int(self.groups.value), 'tolerance': float(self.tolerance.value),
                 'clicks': [dict(c) for c in self.clicks]}
        if new_history:
            self.hist, self.hist_i = [entry], 0
        else:
            self.hist.append(entry)
            self.hist_i = len(self.hist) - 1
        nb = sum(c['kind'] == KINDS[0] for c in self.clicks)
        self._show(f'Suggested {nb} background and {len(self.clicks) - nb} chromatophore clicks from the colours of this '
                   'frame. Check the preview and correct it with your own clicks, or press **Next suggestion** to try another '
                   'start (**Previous suggestion** goes back to the earlier ones).', 'info')
        self._changed()
        self._hist_update()

    def _on_slider(self, event):
        if self.frame is None:
            return
        try:
            keep = self.frame['path']
            self.frame = read_frame(keep, event.new)
        except Exception as exc:
            self._show(f'Could not read that frame: {exc}', 'danger')
            return
        self._show('')
        self._touch()

    # plot ------------------------------------------------------------------------------------
    def _rgb(self, img):
        f = self.frame
        return hv.RGB(np.flipud(img), bounds=(0, 0, f['width'], f['height']), kdims=['x', 'y'])

    def _build_plot(self):
        f = self.frame
        w = 560
        h = max(int(round(w * f['height'] / f['width'])), 120)
        pane_opts = dict(width=w, height=h, xaxis=None, yaxis=None, active_tools=['wheel_zoom'],
                         tools=['wheel_zoom', 'pan', 'reset'], toolbar='above')
        stream = hv.streams.Params(self, ['version'])

        def left(version):
            return self._rgb(self.frame['small']).opts(title='Click here', **pane_opts)

        def marks(version):
            t = self.frame['time']
            h = self.frame['height']
            ticked = set(self.list.selection)
            rows = [(c['x'], h - c['y'], i + 1, c['kind'], c.get('by', 'you'), i in ticked) for i, c in enumerate(self.clicks)
                    if abs(c['time'] - t) < 1e-6]
            df = pd.DataFrame(rows, columns=['x', 'y', 'n', 'kind', 'by', 'ticked'])
            colours = hv.dim('kind').categorize(EDGE_COLOUR)
            rings = hv.Points(df, ['x', 'y'], ['n', 'kind', 'by']).opts(
                size=11, fill_alpha=0, line_width=3, marker='circle', color=colours, line_color=colours,
                tools=['hover'], hover_tooltips=[('click', '@n'), ('kind', '@kind'), ('set by', '@by')])
            numbers = hv.Labels(df, ['x', 'y'], 'n').opts(text_font_size='9pt', text_color='#222222', text_align='left',
                                                          xoffset=int(0.012 * self.frame['width']), yoffset=0)
            layers = rings * numbers
            if df['ticked'].any():                                  # ticked rows of the list: a big gold ring on the frame
                layers = layers * hv.Points(df[df['ticked']], ['x', 'y']).opts(
                    size=26, fill_alpha=0, line_width=4, line_color='#ffb000', marker='circle', tools=[])
            return layers

        def right(version):
            return self._rgb(preview_image(self.frame['small'], self.table, self.view.value)).opts(
                title='Preview: ' + self.view.value, **pane_opts)

        left_map = hv.DynamicMap(left, streams=[stream])
        self._tap = hv.streams.Tap(source=left_map)
        self._tap.add_subscriber(self._on_tap)
        layout = (left_map * hv.DynamicMap(marks, streams=[stream])) + hv.DynamicMap(right, streams=[stream])
        self.plot[:] = [pn.pane.HoloViews(layout.opts(shared_axes=True), linked_axes=True)]

    # run -------------------------------------------------------------------------------------
    def _run(self, _):
        if self.table is None or self.frame is None:
            return
        try:
            video = cb.find_video(self.video_in.value.strip())
        except Exception as exc:
            self._show(f'{exc}', 'danger')
            return
        if video != self.frame['path']:
            self._show('The Video box now names a different video from the one loaded here. Press **Load video** '
                       'again, or put the loaded video\'s name back.', 'warning')
            return
        out = cb.settings()['output_dir']
        out.mkdir(parents=True, exist_ok=True)
        candidate = out / '_picker_candidate.npz'                 # checked first, so a bad table is never kept for reuse
        save_table(candidate, self.table)
        problem = cb.lookup_table_problem(candidate, video, self.frame['frame'])
        if problem:
            self._show(problem, 'danger')
            return
        name = self.name_in.value.strip() if self.name_in is not None else ''
        final = out / f'{name or video.stem}.lookup_table.npz'      # named after the dataset, so another run's table is not replaced
        save_table(final, self.table)
        error = self.on_run(final)
        if error:
            self._show(error, 'warning')
        else:
            self._show(f'Started CHROMAS with these colours. They are saved as {final.name}; the next run of this '
                       'video or dataset name reuses them when you press **Run CHROMAS** with lookup.', 'success')
