"""Train a random forest segmentation model from the Datasets tab: dropdowns and one button per step.

It replaces the command lines of the manual procedure (chromas_training.py and `chromas train -m rf`):

    1 add frames from a video (+ a draft mask made with a lookup table)   -> chromas_training frames + draft
    2 correct the drafts by hand (the one step that cannot be automated)
    3 make the masks and check them                                       -> chromas_training masks + check
    4 cut small crops (a forest uses every pixel of every training image) -> chromas_training crops
    5 train                                                                -> chromas train -m rf
    6 score the model against lookup on frames it has not seen              -> chromas_score_segmentation.py
    7 use the model in the run form above
"""
import contextlib
import os
import re
import threading
from pathlib import Path

import panel as pn

import chromas_bridge as cb
import chromas_training as ct
import dataset_state

BASE = cb.BASE_DIR
TRAIN_ROOT = BASE / 'training'
DEFAULT_SET = 'squid/train'
VIDEO_SKIP = ('_chunk_', 'registered_', '.lookup')
DEFAULT_KEY = 'example_humboldtsquid'     # the video and lookup table whose name holds this are chosen by default


def _pick(options, keep=None):
    """The value to show in a dropdown: the current one if it is still listed, else the entry named DEFAULT_KEY, else
    the first."""
    if keep in options.values():
        return keep
    for label, value in options.items():
        if DEFAULT_KEY in label.lower():
            return value
    return next(iter(options.values()), None)
_print_lock = threading.Lock()


class _LineWriter:
    """Sends everything printed inside `with redirect_stdout(...)` to log, line by line."""

    def __init__(self, log):
        self.log, self.buf = log, ''

    def write(self, text):
        self.buf += text
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip():
                self.log(line.rstrip())
        return len(text)

    def flush(self):
        if self.buf.strip():
            self.log(self.buf.rstrip())
        self.buf = ''


def training_sets():
    """Names (like 'squid/train') of the folders under training/ that hold an images/ folder."""
    names = {DEFAULT_SET}
    if TRAIN_ROOT.is_dir():
        for p in TRAIN_ROOT.glob('*/*'):
            if (p / 'images').is_dir():
                names.add(p.relative_to(TRAIN_ROOT).as_posix())
    return sorted(names)


def find_videos(st):
    out = {}
    folders = [st['data_dir'], BASE / 'videos', BASE / 'examples', *sorted((BASE / 'examples').glob('*'))]
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in cb.VIDEO_EXTS and not any(k in p.name for k in VIDEO_SKIP):
                out[f'{p.parent.name}/{p.name}'] = str(p)
    return out


def find_lookups(st):
    out = {}
    for p in sorted((BASE / 'examples').glob('*/lookup_table.npz')):
        out[f'{p.parent.name}/{p.name}'] = str(p)
    for p in sorted(st['output_dir'].glob('*.lookup_table.npz')) if st['output_dir'].is_dir() else []:
        out[p.name] = str(p)
    return out


def find_models(st):
    runs = cb.training_runs_dir(st)
    return {p.parent.name: str(p) for p in sorted(runs.glob(f'*/{cb.RF_MODEL_NAME}'))} if runs.is_dir() else {}


class TrainingPanel:
    def __init__(self, weights_in, method_in):
        self.weights_in, self.method_in = weights_in, method_in
        self.job = {'running': False, 'log': [], 'status': ''}
        self.lock = threading.Lock()
        st = cb.settings()
        self.problems = cb.check_install(st)

        self.set_in = pn.widgets.Select(name='Training set (folder under training/)', options=training_sets(),
                                        value=DEFAULT_SET, width=260)
        # 1
        self.video_in = pn.widgets.Select(name='Video', options={}, width=330)
        self.lookup_in = pn.widgets.Select(name='Lookup table for the draft', options={}, width=330)
        self.n_in = pn.widgets.Select(name='Frames', options=[1, 2, 3, 4, 6, 8, 10], value=3, width=90)
        self.add_btn = pn.widgets.Button(name='Add frames and draft masks', button_type='primary', width=220)
        # 2
        self.open_btn = pn.widgets.Button(name='Open the folder of drafts', width=200)
        # 3
        self.masks_btn = pn.widgets.Button(name='Make masks and check', button_type='primary', width=200)
        # 4
        self.size_in = pn.widgets.Select(name='Crop size (px)', options=[256, 384, 512], value=512, width=120)
        self.per_in = pn.widgets.Select(name='Crops per frame', options=[1, 2, 3], value=1, width=130)
        self.crops_btn = pn.widgets.Button(name='Make crops', button_type='primary', width=140)
        # 5
        self.train_on_in = pn.widgets.Select(name='Train on', options={'Crops (recommended)': 'crops', 'Full frames': 'full'},
                                             value='crops', width=200)
        self.name_in = pn.widgets.TextInput(name='Model name', value='rf_v1', width=160)
        self.train_btn = pn.widgets.Button(name='Train random forest', button_type='primary', width=200)
        # 6
        self.model_in = pn.widgets.Select(name='Model', options={}, width=240)
        self.test_in = pn.widgets.Select(name='Score on frames in', options=training_sets(), value=DEFAULT_SET, width=260)
        self.lookup_score_in = pn.widgets.Select(name='Lookup table to compare', options={}, width=330)
        self.score_btn = pn.widgets.Button(name='Score model against lookup', button_type='primary', width=240)
        # 7
        self.use_btn = pn.widgets.Button(name='Use this model in the run form', button_type='success', width=240)
        self.refresh_btn = pn.widgets.Button(name='Refresh lists', width=130)
        self.stop_btn = pn.widgets.Button(name='Stop', button_type='danger', width=90)

        self.status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
        self.log_view = pn.widgets.TextAreaInput(value='', disabled=True, height=200, sizing_mode='stretch_width')

        self.add_btn.on_click(lambda _: self._start('add', self._add))
        self.masks_btn.on_click(lambda _: self._start('masks', self._masks))
        self.crops_btn.on_click(lambda _: self._start('crops', self._crops))
        self.train_btn.on_click(lambda _: self._start('train', self._train))
        self.score_btn.on_click(lambda _: self._start('score', self._score))
        self.open_btn.on_click(self._open_folder)
        self.use_btn.on_click(self._use)
        self.refresh_btn.on_click(lambda _: self.refresh())
        self.stop_btn.on_click(lambda _: cb.request_stop())
        self.poller = dataset_state.poll_while_running(self._poll, lambda: self.job['running'], period=1500)
        self.refresh()

        def step(n, title, *widgets, note=None):
            parts = [pn.pane.Markdown(f'**{n}. {title}**', margin=(10, 0, 0, 0))]
            if note:
                parts.append(pn.pane.Markdown(note, sizing_mode='stretch_width', margin=(0, 0, 0, 0)))
            parts.append(pn.Row(*[w if not isinstance(w, tuple) else pn.Column(pn.Spacer(height=22), w[0])
                                  for w in widgets]))
            return pn.Column(*parts, sizing_mode='stretch_width')

        self.panel = pn.Column(
            pn.pane.Markdown(
                'A random forest labels each pixel as chromatophore or background from its local texture and brightness. '
                'CHROMAS supplies no model: you train one on frames you have corrected. It needs no GPU. Try the lookup '
                'table first and train only if the lookup result is not good enough. Work down the steps; each button '
                'runs one step.', sizing_mode='stretch_width'),
            pn.Row(self.set_in, pn.Column(pn.Spacer(height=22), self.refresh_btn),
                   pn.Column(pn.Spacer(height=22), self.stop_btn)),
            step(1, 'Add frames from a video',
                 self.video_in, self.lookup_in, self.n_in, (self.add_btn,),
                 note='Frames are spread over the video; the draft mask is what the lookup table marks. Do this for '
                      'several videos (20 to 30 frames from at least three is a good start).'),
            step(2, 'Correct the drafts', (self.open_btn,),
                 note='In the folder `painted` of the training set, open each draft over its frame in a paint program and '
                      'erase what is not a chromatophore (the electrode, debris), fill what was missed, then save it under '
                      'the same file name. An uncorrected draft only teaches the forest to copy lookup, and a score '
                      'against it cannot show it is better. This step cannot be done by the dashboard.'),
            step(3, 'Make the masks and check them', (self.masks_btn,)),
            step(4, 'Cut small crops', self.size_in, self.per_in, (self.crops_btn,),
                 note='The forest uses every pixel of every training image (a 1920 x 1080 frame is about two million '
                      'pixels), so it trains on small crops of the most chromatophore-rich parts. They go in the same '
                      'folder name with `_crops` added.'),
            step(5, 'Train', self.train_on_in, self.name_in, (self.train_btn,),
                 note='Runs 432 fits, so it takes a while (more than ten minutes on a small machine). The model is saved in '
                      'the `training_runs` folder of CHROMAS under the model name. Ignore the "validation balanced '
                      'accuracy" it prints: it is measured on the training frames and is nearly always high.'),
            step(6, 'Score the model', self.model_in, self.test_in, self.lookup_score_in, (self.score_btn,),
                 note='Score on frames the forest never saw, ideally from other videos: a separate training set. '
                      'Higher Dice is better; compare it with lookup on the same frames.'),
            step(7, 'Use the model', (self.use_btn,),
                 note='Puts the model in the *Model file* box above and chooses *randomforest*; then run CHROMAS on a '
                      'video as usual. The run segments pixel by pixel, which is slow: crop the video first. Small '
                      'chromatophores a few pixels across can disappear in CHROMAS\'s clean-up, so check the overlay '
                      'video. Keep the model with the scikit-learn version of the CHROMAS environment; it is a Python '
                      'pickle, so open only files you made yourself.'),
            self.status, self.log_view, sizing_mode='stretch_width')

    # ── lists ────────────────────────────────────────────────────────────────────
    def refresh_lookups(self):
        opts = find_lookups(cb.settings())
        for w in (self.lookup_in, self.lookup_score_in):
            keep = w.value
            w.options = opts
            w.value = _pick(opts, keep)

    def refresh(self):
        st = cb.settings()
        sets = training_sets()
        for w in (self.set_in, self.test_in):
            keep = w.value
            w.options = sets
            w.value = keep if keep in sets else DEFAULT_SET
        vids = find_videos(st)
        keep = self.video_in.value
        self.video_in.options = vids
        self.video_in.value = _pick(vids, keep)
        self.refresh_lookups()
        models = find_models(st)
        keep = self.model_in.value
        self.model_in.options = models
        self.model_in.value = keep if keep in models.values() else next(iter(models.values()), None)

    # ── job runner ───────────────────────────────────────────────────────────────
    def _log(self, line):
        with self.lock:
            self.job['log'].append(line)

    def _poll(self):
        with self.lock:
            lines, msg, running = list(self.job['log'][-60:]), self.job['status'], self.job['running']
        self.log_view.value = '\n'.join(reversed(lines))       # newest first: the box does not scroll by itself
        self.status.visible = bool(msg)
        self.status.object = msg
        self.status.alert_type = 'info' if running else ('danger' if msg.startswith('Failed') else 'success')
        for b in (self.add_btn, self.masks_btn, self.crops_btn, self.train_btn, self.score_btn):
            b.disabled = running or bool(self.problems and b in (self.train_btn, self.score_btn))
        if not running:
            self.refresh()

    def _start(self, name, fn):
        if self.job['running']:
            return
        with self.lock:
            self.job.update(running=True, log=[], status=f'Running: {name}.')
        cb.clear_stop()
        self._poll()
        self.poller()

        def work():
            try:
                with _print_lock, contextlib.redirect_stdout(_LineWriter(self._log)):
                    msg = fn()
                with self.lock:
                    self.job['status'] = msg or 'Done.'
            except Exception as exc:
                with self.lock:
                    self.job['status'] = f'Failed: {exc}'
            finally:
                with self.lock:
                    self.job['running'] = False
        threading.Thread(target=work, daemon=True).start()

    def _dir(self, name=None):
        return TRAIN_ROOT / (name or self.set_in.value)

    # ── the steps ────────────────────────────────────────────────────────────────
    def _add(self):
        if not self.video_in.value:
            raise RuntimeError('No video found. Put videos in the CHROMAS data folder, videos/ or examples/.')
        if not self.lookup_in.value:
            raise RuntimeError('No lookup table found. Make one first (Segmentation: lookup, above).')
        d = self._dir()
        d.mkdir(parents=True, exist_ok=True)
        written = ct.export_frames(self.video_in.value, d, int(self.n_in.value))
        ct.draft_masks(d, self.lookup_in.value)
        return (f'Done: {len(written)} frame(s) added to {d}. Next: correct the drafts in the folder "painted" (step 2), '
                'or add frames from another video.')

    def _open_folder(self, _):
        p = self._dir() / 'painted'
        if not p.is_dir():
            self.job['status'] = 'Nothing to open yet: add frames first (step 1).'
        elif os.name == 'nt':
            os.startfile(str(p))                                  # the dashboard runs on this computer
            self.job['status'] = f'Opened {p}.'
        else:
            self.job['status'] = f'The drafts are in {p}.'
        self._poll()

    def _masks(self):
        d = self._dir()
        if not (d / 'painted').is_dir():
            raise RuntimeError('No painted drafts in this training set: add frames first (step 1).')
        ct.convert_masks(d)
        if ct.check(d) != 0:
            raise RuntimeError('the check found problems (listed in the output below).')
        return 'Done: masks are ready. Next: cut crops (step 4) and train (step 5).'

    def _crops(self):
        dest = ct.make_crops(self._dir(), int(self.size_in.value), int(self.per_in.value))
        return f'Done: crops are in {dest}. Next: train (step 5).'

    def _train(self):
        if self.problems:
            raise RuntimeError('; '.join(self.problems))
        name = self.name_in.value.strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise RuntimeError('the model name may contain only letters, digits, _ . and -.')
        d = self._dir()
        if self.train_on_in.value == 'crops':
            d = d.parent / (d.name + '_crops')
            if not d.is_dir():
                raise RuntimeError('no crops yet: run step 4, or choose "Full frames".')
        pkl = cb.train_forest(d, name, self._log)
        return f'Done: model saved as {pkl}. Next: score it (step 6) or use it (step 7).'

    def _score(self):
        if self.problems:
            raise RuntimeError('; '.join(self.problems))
        if not self.model_in.value:
            raise RuntimeError('no trained model yet: train one first (step 5).')
        d = self._dir(self.test_in.value)
        if not (d / 'masks').is_dir():
            raise RuntimeError(f'{d} has no masks: choose a set that has them (step 3).')
        cb.score_models(d, self.lookup_score_in.value, [self.model_in.value], self._log)
        return 'Done: scores are in the output below (Dice: higher is better).'

    def _use(self, _):
        if not self.model_in.value:
            self.job['status'] = 'Failed: no trained model yet.'
        else:
            self.weights_in.value = self.model_in.value
            self.method_in.value = 'randomforest'
            self.job['status'] = 'The model is in the Model file box above and randomforest is chosen.'
        self._poll()
