"""The Training tab: teach CHROMAS to segment the lab's own squid skin (docs/TRAINING_THE_NETWORK.md as buttons).

The random forest and the neural network learn from the same hand-painted frames, so the tab shares the first steps:

    0 before you start        is training worth it; is this computer ready (CHROMAS training patch, supplied network, GPU)
    1 training and test sets  folders under training/ with images/, painted/ and masks/
    2 export frames           chromas_training.export_frames
    3 drafts                  chromas_training.draft_masks (a lookup table's guess, to correct by hand)
    4 paint                   by hand in GIMP or Fiji: the one step the dashboard cannot do; it opens the folder,
                              keeps the painting rules and counts the drafts edited
    5 masks and check         chromas_training.convert_masks + check
    6a random forest          chromas_training.make_crops + `chromas train -m rf`   (chromas_bridge.train_forest)
    6b neural network         `chromas train -m fcn_resnet50 -c 2 --weights ...`     (chromas_bridge.train_network)
    7 score on test frames    chromas_score_segmentation.py: lookup, forests and networks on frames they never saw
    8 use the model           puts it in the Run CHROMAS form of the Datasets tab

CHROMAS itself is not changed: models go to <CHROMAS folder>/training_runs/<name>/, where `chromas train` puts them.
"""
import contextlib
import json
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import panel as pn

import chromas_bridge as cb
import chromas_training as ct
import dataset_state
import rf_training_view as rtv

TRAIN_ROOT = rtv.TRAIN_ROOT
DEFAULT_TRAIN, DEFAULT_TEST = 'squid/train', 'squid/test'
DRAFT_TIMES = '.draft_times.json'       # in painted/: when each draft was made, to count the drafts edited since
RULES_FILE = 'PAINTING_RULES.md'        # in the set's folder: what counts as a chromatophore, kept with the frames
NO_START = '(ImageNet only: needs far more frames)'
SET_NAME = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')


def all_sets():
    """Training sets: folders training/<group>/<name> that hold images/ (and the two defaults)."""
    names = {DEFAULT_TRAIN, DEFAULT_TEST}
    if TRAIN_ROOT.is_dir():
        for p in TRAIN_ROOT.glob('*/*'):
            if p.is_dir() and not p.name.endswith('_crops') and (p / 'images').is_dir():
                names.add(p.relative_to(TRAIN_ROOT).as_posix())
    return sorted(names)


def set_progress(name):
    """Counts for one set: images, drafts or paint layers, those edited since drafting, masks."""
    d = TRAIN_ROOT / name
    images = {p.name for p in (d / 'images').glob('*.png')}
    painted = {p.name: p for p in (d / 'painted').glob('*.png')}
    masks = {p.name for p in (d / 'masks').glob('*.png')}
    try:
        drafted = json.loads((d / 'painted' / DRAFT_TIMES).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        drafted = {}
    edited = sum(1 for n, p in painted.items() if n in drafted and p.stat().st_mtime > drafted[n] + 1)
    unknown = sum(1 for n in painted if n not in drafted)          # painted from nothing, or drafted on the command line
    return {'images': len(images), 'painted': len(painted), 'drafted': len([n for n in painted if n in drafted]),
            'edited': edited, 'unknown': unknown, 'masks': len(masks & images)}


def run_results(st=None):
    """Training runs that have result pictures: {run name: folder}."""
    runs = cb.training_runs_dir(st)
    if not runs.is_dir():
        return {}
    return {p.name: str(p) for p in sorted(runs.iterdir())
            if p.is_dir() and ((p / 'metrics_plot.png').exists() or (p / 'predictions.png').exists())}


class TrainingTab:
    def __init__(self):
        st = cb.settings()
        self.problems = cb.check_install(st)
        self.job = {'running': False, 'log': [], 'status': ''}
        self.lock = threading.Lock()
        self.ready = {'checked': False, 'info': None}

        # 0
        self.ready_pane = pn.pane.Markdown('*Checking this computer ...*', sizing_mode='stretch_width')
        self.ready_btn = pn.widgets.Button(name='Check again', width=120)
        # 1
        self.train_set = pn.widgets.Select(name='Training set', options=all_sets(), value=DEFAULT_TRAIN, width=220)
        self.test_set = pn.widgets.Select(name='Test set (never trained on)', options=all_sets(), value=DEFAULT_TEST,
                                          width=220)
        self.new_set = pn.widgets.TextInput(name='New set (group/name)', placeholder='e.g. squid/train2', width=200)
        self.new_btn = pn.widgets.Button(name='Create', width=80)
        self.progress = pn.pane.Markdown('', sizing_mode='stretch_width')
        self.refresh_btn = pn.widgets.Button(name='Refresh lists', width=120)
        self.stop_btn = pn.widgets.Button(name='Stop', button_type='danger', width=80)
        # which set steps 2 to 5 work on
        self.target = pn.widgets.RadioButtonGroup(name='Work on', options=['Training set', 'Test set'],
                                                  value='Training set', button_type='primary', button_style='outline',
                                                  width=260)
        # 2
        self.video_in = pn.widgets.Select(name='Video', options={}, width=340)
        self.how_in = pn.widgets.RadioButtonGroup(name='Which frames', options=['Spread over the video', 'At these seconds'],
                                                  value='Spread over the video', width=330)
        self.n_in = pn.widgets.Select(name='Frames', options=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20], value=5, width=90)
        self.times_in = pn.widgets.TextInput(name='Seconds', placeholder='e.g. 0 3 4.5 10', width=180, visible=False)
        self.frames_btn = pn.widgets.Button(name='Export frames', button_type='primary', width=150)
        # 3
        self.lookup_in = pn.widgets.Select(name='Lookup table for the drafts', options={}, width=340)
        self.draft_btn = pn.widgets.Button(name='Make drafts', button_type='primary', width=140)
        # 4
        self.open_btn = pn.widgets.Button(name='Open the painted folder', width=200)
        self.rules_in = pn.widgets.TextAreaInput(name='Painting rules for this set (saved as PAINTING_RULES.md)',
                                                 height=110, sizing_mode='stretch_width',
                                                 placeholder='e.g. Paint every chromatophore, small ones too, as large as '
                                                             'its pigment in this frame. Partly hidden ones: paint the '
                                                             'visible part. Out of focus: leave out. The electrode, '
                                                             'debris and bubbles: never.')
        self.rules_btn = pn.widgets.Button(name='Save the rules', width=130)
        # 5
        self.masks_btn = pn.widgets.Button(name='Make masks and check', button_type='primary', width=200)
        # 6a
        self.size_in = pn.widgets.Select(name='Crop size (px)', options=[256, 384, 512], value=512, width=120)
        self.per_in = pn.widgets.Select(name='Crops per frame', options=[1, 2, 3], value=1, width=130)
        self.crops_btn = pn.widgets.Button(name='Make crops', button_type='primary', width=130)
        self.rf_on_in = pn.widgets.Select(name='Train on', options={'Crops (recommended)': 'crops', 'Full frames': 'full'},
                                          value='crops', width=190)
        self.rf_name = pn.widgets.TextInput(name='Model name', value='rf_v1', width=150)
        self.rf_btn = pn.widgets.Button(name='Train random forest', button_type='primary', width=190)
        # 6b
        self.start_in = pn.widgets.Select(name='Start from', options={}, width=420)
        self.epochs_in = pn.widgets.IntInput(name='Passes (-e)', value=50, start=1, step=10, width=110)
        self.val_in = pn.widgets.IntInput(name='Validate every (-v)', value=5, start=1, step=1, width=140)
        self.nn_name = pn.widgets.TextInput(name='Run name', value='squid_v1', width=150)
        self.nn_btn = pn.widgets.Button(name='Train the network', button_type='primary', width=170)
        self.nn_cmd = pn.pane.Markdown('', sizing_mode='stretch_width')
        self.run_sel = pn.widgets.Select(name='Results of a run', options={}, width=260)
        self.run_view = pn.Column(sizing_mode='stretch_width')
        # 7
        self.score_lookup = pn.widgets.Select(name='Lookup table', options={}, width=340)
        self.score_forests = pn.widgets.MultiSelect(name='Random forests', options={}, size=4, width=280)
        self.score_nets = pn.widgets.MultiSelect(name='Networks', options={}, size=4, width=420)
        self.score_btn = pn.widgets.Button(name='Score on the test set', button_type='primary', width=190)
        self.score_view = pn.Column(sizing_mode='stretch_width')
        # 8
        self.use_in = pn.widgets.Select(name='Model', options={}, width=420)
        self.use_btn = pn.widgets.Button(name='Use it in Run CHROMAS', button_type='success', width=200)

        self.status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
        self.log_view = pn.widgets.TextAreaInput(name='Output (newest first)', value='', disabled=True, height=220,
                                                 sizing_mode='stretch_width')

        self.ready_btn.on_click(lambda _: self._check_ready())
        self.new_btn.on_click(self._create_set)
        self.refresh_btn.on_click(lambda _: self.refresh())
        self.stop_btn.on_click(lambda _: cb.request_stop())
        self.how_in.param.watch(self._how, 'value')
        self.frames_btn.on_click(lambda _: self._start('Export frames', self._frames))
        self.draft_btn.on_click(lambda _: self._start('Make drafts', self._drafts))
        self.open_btn.on_click(self._open_folder)
        self.rules_btn.on_click(self._save_rules)
        self.masks_btn.on_click(lambda _: self._start('Make masks and check', self._masks))
        self.crops_btn.on_click(lambda _: self._start('Make crops', self._crops))
        self.rf_btn.on_click(lambda _: self._start('Train random forest', self._train_forest))
        self.nn_btn.on_click(lambda _: self._start('Train the network', self._train_network))
        self.score_btn.on_click(lambda _: self._start('Score', self._score))
        self.use_btn.on_click(self._use)
        for w in (self.target, self.train_set, self.test_set):
            w.param.watch(lambda *_: (self._show_set(), self._load_rules()), 'value')
        for w in (self.start_in, self.epochs_in, self.val_in, self.nn_name, self.train_set):
            w.param.watch(lambda *_: self._show_command(), 'value')
        self.run_sel.param.watch(lambda *_: self._show_run(), 'value')
        self.poller = dataset_state.poll_while_running(self._poll, lambda: self.job['running'], period=1500)
        self.ready_poller = dataset_state.poll_while_running(self._show_ready, lambda: not self.ready['checked'],
                                                             period=1000)

        self.refresh()
        self._load_rules()
        self._check_ready()
        self.panel = self._layout()

    # ── layout ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _step(title, *rows, note=None):
        parts = [pn.pane.Markdown(f'#### {title}', margin=(12, 5, 0, 5))]
        if note:
            parts.append(pn.pane.Markdown(note, sizing_mode='stretch_width', margin=(0, 5, 0, 5)))
        for r in rows:
            parts.append(r if not isinstance(r, (list, tuple)) else
                         pn.Row(*[w if not isinstance(w, tuple) else pn.Column(pn.Spacer(height=22), w[0]) for w in r]))
        return pn.Column(*parts, sizing_mode='stretch_width')

    def _layout(self):
        gimp = pn.Card(pn.pane.Markdown(
            '1. Open the frame (`images\\<name>.png`).\n'
            '2. *File > Open as Layers* and choose its draft, `painted\\<name>.png`. Without a draft: *Layer > New '
            'Layer*, filled with *Transparency*.\n'
            '3. On the paint layer use the **Pencil** (N), not the brush, so a pixel is painted or not; any colour (red is '
            'easiest to see). To remove paint use the **Eraser** with *Hard edge* ticked.\n'
            '4. Zoom to 200-400 % and work across the frame: paint every chromatophore completely, as large as its pigment '
            'is in this frame, small ones too; erase paint from the electrode, debris, bubbles and shadows; where two '
            'expanded chromatophores touch, paint both.\n'
            '5. Hide the image layer (eye icon), then *File > Export As...* `painted\\<same name>.png`, over the draft.\n\n'
            'Fiji / ImageJ works too: paint on a new image of the same size and save it as PNG under the same name. If '
            'two people paint, have both paint the same two frames and score one against the other (step 7) to see how '
            'consistent you are.', sizing_mode='stretch_width'),
            title='How to paint in GIMP', collapsed=True, sizing_mode='stretch_width')
        return pn.Column(
            pn.pane.Markdown('### Train a segmentation model on your own frames'),
            pn.pane.Markdown(
                'Segmentation decides, for every pixel, whether it is chromatophore or background. **Lookup** does it by '
                'colour alone. A **random forest** and the **neural network** learn it from frames you paint by hand, so '
                'they can also use shape and texture: a dark chromatophore against the electrode, debris or a shadow. '
                '**Train only if lookup falls short**: live animals, uneven or changing light, chromatophores of several '
                'colours, or a lot of debris. Step 7 tells you whether it was worth it. Both kinds learn from the same '
                'painted frames, so steps 1 to 5 are shared. Guide: `docs/TRAINING_THE_NETWORK.md`.',
                sizing_mode='stretch_width'),
            self._step('0. Before you start', [self.ready_pane, (self.ready_btn,)],
                       note='Painting 20 to 30 frames takes about a day (less when you correct drafts). A random forest '
                            'trains on this computer in minutes. The network trains in about an hour on an NVIDIA GPU and '
                            'much longer on a CPU; no CPU run has been timed on this laptop.'),
            self._step('1. Training and test sets',
                       [self.train_set, self.test_set, self.new_set, (self.new_btn,), (self.refresh_btn,),
                        (self.stop_btn,)], self.progress,
                       note='Keep about 80 % of the frames for training and 20 % for testing, the test frames preferably '
                            'from other videos. The test frames are never trained on: step 7 scores against them. Steps 2 '
                            'to 5 work on the set chosen here:'),
            pn.Row(self.target),
            self._step('2. Export frames', [self.video_in, self.how_in, self.n_in, self.times_in, (self.frames_btn,)],
                       note='Choose frames that cover what the model will meet: several animals and videos, chromatophores '
                            'relaxed and expanded, the electrode, debris, blur, bright and dark light, at the same '
                            'magnification as the videos you will analyse. Frames go to `images\\` and must be at least '
                            '512 px on each side.'),
            self._step('3. Drafts (optional, recommended)', [self.lookup_in, (self.draft_btn,)],
                       note='A lookup table\'s guess for every frame without a paint layer yet, as a red layer in '
                            '`painted\\`. Correcting a draft is much faster than painting from nothing. The colour '
                            'picker in the Datasets tab saves a table for each video it runs.'),
            self._step('4. Paint (in GIMP or Fiji)', [(self.open_btn,)], gimp, self.rules_in, [(self.rules_btn,)],
                       note='The one step the dashboard cannot do. The model learns exactly what you paint, so decide once '
                            'what counts (partly hidden or out-of-focus chromatophores), write it below and apply it '
                            'everywhere. An uncorrected draft only teaches the model to copy lookup. Step 1 counts the '
                            'drafts edited since they were made.'),
            self._step('5. Masks and check', [(self.masks_btn,)],
                       note='Turns each paint layer into the mask CHROMAS trains on (`masks\\`, 0 = background, 1 = '
                            'chromatophore: they look black in a viewer) and checks every pair. It ends with *Ready for '
                            'chromas train*. Do it for the training set and the test set.'),
            pn.layout.Divider(),
            self._step('6a. Random forest (no GPU needed)',
                       [self.size_in, self.per_in, (self.crops_btn,)],
                       [self.rf_on_in, self.rf_name, (self.rf_btn,)],
                       note='A forest uses every pixel of every training image, so it trains on small crops of the most '
                            'chromatophore-rich parts (`<set>_crops`). Training runs 432 fits: more than ten minutes on a '
                            'small machine. Ignore the "validation balanced accuracy" it prints: it is measured on the '
                            'training frames.'),
            self._step('6b. Neural network (fine-tune the supplied one)',
                       [self.start_in], [self.epochs_in, self.val_in, self.nn_name, (self.nn_btn,)], self.nn_cmd,
                       [self.run_sel], self.run_view,
                       note='Starts from the network that comes with CHROMAS (trained on cuttlefish) and trains it further '
                            'on the training set: `chromas train -m fcn_resnet50 -c 2`. CHROMAS cuts each frame into '
                            '512 x 512 windows, keeps 10 % of them for its own validation and saves the passes with the '
                            'lowest validation loss. The first run downloads the ImageNet starting weights (about 135 MB). '
                            'Keep this page open while it runs; **Stop** ends it, and a stopped run saves no network. '
                            '**Passes**: how many times it goes over the training windows (CHROMAS default 100; fewer is '
                            'safer with few frames). **Validate every**: how often it checks itself. After the run, '
                            '`metrics_plot.png` should show the validation scores improving and then levelling off; if '
                            'training keeps improving while validation gets worse, it is memorising the frames: use fewer '
                            'passes or more frames.'),
            pn.layout.Divider(),
            self._step('7. Score on the test set', [self.score_lookup, self.score_forests, self.score_nets],
                       [(self.score_btn,)], self.score_view,
                       note='Segments every frame of the test set with each method and compares it with your painted '
                            'mask. **Dice** and **IoU**: overlap with your mask, 1 = identical. **Precision**: how much of '
                            'what it marks really is chromatophore (low = it marks debris or the electrode). **Recall**: '
                            'how much of the real pigment it finds (low = it misses or shrinks chromatophores). **Found**: '
                            'chromatophores of your mask it touches. **Extra**: regions it marks that are not '
                            'chromatophores. Use a trained model only if it has a clearly higher Dice than lookup, fewer '
                            'extra regions and finds at least as many chromatophores; otherwise keep lookup, which is '
                            'faster and needs no GPU.'),
            self._step('8. Use the model', [self.use_in, (self.use_btn,)],
                       note='Chooses *neuralnet* or *randomforest* in the Run CHROMAS form of the **Datasets** tab and puts '
                            'the model in its *Model file* box; then run CHROMAS on a video as usual. Check the overlay '
                            'video of the first run. A forest model is a Python pickle: open only files you made.'),
            self.status, self.log_view, sizing_mode='stretch_width')

    # ── lists and views ──────────────────────────────────────────────────────────
    def _dir(self, which=None):
        name = self.train_set.value if (which or self.target.value) == 'Training set' else self.test_set.value
        return TRAIN_ROOT / name

    def refresh(self):
        st = cb.settings()
        sets = all_sets()
        for w in (self.train_set, self.test_set):
            keep = w.value
            w.options = sets
            w.value = keep if keep in sets else w.value
        vids = rtv.find_videos(st)
        keep = self.video_in.value
        self.video_in.options = vids
        self.video_in.value = rtv._pick(vids, keep)
        looks = rtv.find_lookups(st)
        for w, empty in ((self.lookup_in, False), (self.score_lookup, True)):
            keep = w.value
            opts = {'(none)': ''} | looks if empty else looks
            w.options = opts
            w.value = keep if keep in opts.values() else rtv._pick(looks, keep) or ''
        forests = rtv.find_models(st)
        nets = cb.network_models(st)
        keep = self.start_in.value
        self.start_in.options = nets | {NO_START: ''}
        self.start_in.value = keep if keep in self.start_in.options.values() else next(iter(nets.values()), '')
        for w, opts in ((self.score_forests, forests), (self.score_nets, nets)):
            keep = [v for v in w.value if v in opts.values()]
            w.options = opts
            w.value = keep
        models = {f'network: {k.split(": ", 1)[-1]}': v for k, v in nets.items()} | \
                 {f'random forest: {k}': v for k, v in forests.items()}
        keep = self.use_in.value
        self.use_in.options = models
        self.use_in.value = keep if keep in models.values() else next(iter(models.values()), None)
        runs = run_results(st)
        keep = self.run_sel.value
        self.run_sel.options = {'(choose a run)': ''} | runs
        self.run_sel.value = keep if keep in runs.values() else ''
        self._show_set()
        self._show_command()

    def _show_set(self):
        rows = []
        for label, name in (('Training set', self.train_set.value), ('Test set', self.test_set.value)):
            c = set_progress(name)
            painted = f"{c['painted']} paint layers"
            if c['drafted']:
                painted += f" ({c['edited']} of {c['drafted']} drafts edited since they were made)"
            rows.append(f"- **{label}** `training/{name}`: {c['images']} frames, {painted}, {c['masks']} masks")
        same = self.train_set.value == self.test_set.value
        warn = '\n\n**The training set and the test set are the same folder**: scores on it would only show memorising.' \
            if same else ''
        self.progress.object = '\n'.join(rows) + warn

    def _load_rules(self):
        try:
            self.rules_in.value = (self._dir() / RULES_FILE).read_text(encoding='utf-8')
        except OSError:
            self.rules_in.value = ''

    def _show_command(self):
        st = cb.settings()
        name = self.nn_name.value.strip() or 'squid_v1'
        cmd = cb.network_command(TRAIN_ROOT / self.train_set.value, name, self.epochs_in.value or 50,
                                 self.val_in.value or 5, self.start_in.value or None, st)
        text = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
        self.nn_cmd.object = ('**On a GPU machine instead** (same CHROMAS install, training set copied there; create the '
                              f'output folder first): `{text}`')

    def _show_run(self):
        d = Path(self.run_sel.value) if self.run_sel.value else None
        if not d or not d.is_dir():
            self.run_view.objects = []
            return
        parts = []
        for f, what in (('metrics_plot.png', 'Loss and scores over the passes (training and validation)'),
                        ('predictions.png', 'Validation windows: frame, your mask, the prediction')):
            if (d / f).exists():
                parts += [pn.pane.Markdown(f'**{what}**'), pn.pane.PNG(str(d / f), sizing_mode='scale_width',
                                                                       max_width=1100)]
        self.run_view.objects = parts

    # ── job runner (one at a time; output in the box below) ─────────────────────────────────────────────────────
    def _log(self, line):
        with self.lock:
            self.job['log'].append(line)

    def _say(self, text):
        with self.lock:
            self.job['status'] = text
        self._poll()

    def _poll(self):
        with self.lock:
            lines, msg, running = list(self.job['log'][-80:]), self.job['status'], self.job['running']
        self.log_view.value = '\n'.join(reversed(lines))
        self.status.visible = bool(msg)
        self.status.object = msg
        self.status.alert_type = 'info' if running else ('danger' if msg.startswith('Failed') else 'success')
        busy = (self.frames_btn, self.draft_btn, self.masks_btn, self.crops_btn, self.rf_btn, self.nn_btn,
                self.score_btn)
        for b in busy:
            b.disabled = running or bool(self.problems and b in (self.rf_btn, self.nn_btn, self.score_btn))
        if not running:
            self.refresh()

    def _start(self, name, fn):
        if self.job['running']:
            return
        with self.lock:
            self.job.update(running=True, log=[], status=f'Running: {name} ...', started=time.time())
        cb.clear_stop()
        self._poll()
        self.poller()

        def work():
            try:
                with rtv._print_lock, contextlib.redirect_stdout(rtv._LineWriter(self._log)):
                    msg = fn()
                with self.lock:
                    self.job['status'] = msg or 'Done.'
            except cb.Stopped as exc:
                with self.lock:
                    self.job['status'] = f'Failed: stopped. {exc}'
            except (Exception, SystemExit) as exc:
                with self.lock:
                    self.job['status'] = f'Failed: {exc}'
            finally:
                with self.lock:
                    self.job['running'] = False
        threading.Thread(target=work, daemon=True).start()

    # ── step 0 ───────────────────────────────────────────────────────────────────
    def _check_ready(self):
        self.ready.update(checked=False, info=None)
        self.ready_pane.object = '*Checking this computer (asks the CHROMAS Python about PyTorch: a few seconds) ...*'

        def work():
            try:
                info = cb.training_readiness()
            except Exception as exc:
                info = {'problems': [str(exc)]}
            self.ready.update(checked=True, info=info)
        self.ready_thread = threading.Thread(target=work, daemon=True)
        self.ready_thread.start()
        self.ready_poller()

    def _show_ready(self):
        if not self.ready['checked']:
            return
        info = self.ready['info'] or {}
        ok, no, warn = '\u2705', '\u274c', '\u26a0\ufe0f'
        lines = [f'- {ok if not self.problems else no} CHROMAS installed'
                 + ('' if not self.problems else ': ' + '; '.join(self.problems))]
        if info.get('torch'):
            lines.append(f'- {ok if info.get("patched") else no} CHROMAS training patch '
                         + ('applied' if info.get('patched') else 'not applied: run `gillylab_chromas\\.venv\\Scripts\\'
                            'python gillylab_chromas\\extras\\apply_patches.py` once'))
        lines.append(f'- {ok if info.get("weights") else no} supplied network '
                     + ('found' if info.get('weights') else 'not found (only needed to fine-tune it, step 6b)'))
        if info.get('torch'):
            gpu, torch_v = info.get('gpu'), info['torch']
            slow = ('the network trains on the CPU, which is slow (for real runs use the command under 6b on a GPU '
                    'machine); the random forest is not affected')
            if gpu:
                lines.append(f'- {ok} NVIDIA GPU: {gpu} (PyTorch {torch_v}, CUDA {info.get("cuda")})')
            elif not info.get('cuda'):
                lines.append(f'- {warn} PyTorch {torch_v} in the CHROMAS environment is the **CPU-only build**: it '
                             'cannot use a GPU even if this computer has one, so ' + slow + '. To use a GPU here, '
                             'install a CUDA build of PyTorch in the CHROMAS environment (pytorch.org).')
            else:
                lines.append(f'- {warn} PyTorch {torch_v} can use an NVIDIA GPU (CUDA {info["cuda"]}), but **none was '
                             'found**: no card, or its driver is missing or too old. So ' + slow + '.')
        else:
            lines += [f'- {no} {p}' for p in info.get('problems', []) if 'PyTorch' in p]
        self.ready_pane.object = '\n'.join(lines)

    # ── steps 1 to 5 ─────────────────────────────────────────────────────────────
    def _create_set(self, _):
        name = self.new_set.value.strip().strip('/').replace('\\', '/')
        if not SET_NAME.fullmatch(name):
            self._say('Failed: a set name is a group and a name, like squid/train2 (letters, digits, _ . and -).')
            return
        (TRAIN_ROOT / name / 'images').mkdir(parents=True, exist_ok=True)
        self.refresh()
        w = self.train_set if self.target.value == 'Training set' else self.test_set
        w.value = name
        self.new_set.value = ''
        self._say(f'Created training/{name}. It is now the {self.target.value.lower()}.')

    def _how(self, *_):
        spread = self.how_in.value == 'Spread over the video'
        self.n_in.visible, self.times_in.visible = spread, not spread

    def _frames(self):
        if not self.video_in.value:
            raise RuntimeError('No video found. Put videos in the CHROMAS data folder, videos/ or examples/.')
        times = None
        if self.how_in.value == 'At these seconds':
            try:
                times = [float(t) for t in self.times_in.value.replace(',', ' ').split()]
            except ValueError:
                raise RuntimeError('the seconds must be numbers, like 0 3 4.5 10.') from None
            if not times:
                raise RuntimeError('enter the seconds of the frames, like 0 3 4.5 10.')
        d = self._dir()
        written = ct.export_frames(self.video_in.value, d, int(self.n_in.value), times)
        return (f'Done: {len(written)} frame(s) in {d / "images"}. Next: drafts (step 3) or paint them (step 4); add '
                'frames from other videos too.')

    def _drafts(self):
        if not self.lookup_in.value:
            raise RuntimeError('no lookup table found: make one with the colour picker in the Datasets tab.')
        d = self._dir()
        if not any((d / 'images').glob('*.png')):
            raise RuntimeError('this set has no frames yet: export some first (step 2).')
        before = {p.name for p in (d / 'painted').glob('*.png')}
        ct.draft_masks(d, self.lookup_in.value)
        times_file = d / 'painted' / DRAFT_TIMES
        try:
            times = json.loads(times_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            times = {}
        new = [p for p in (d / 'painted').glob('*.png') if p.name not in before]
        times.update({p.name: p.stat().st_mtime for p in new})
        times_file.write_text(json.dumps(times, indent=1), encoding='utf-8')
        return f'Done: {len(new)} draft(s) in {d / "painted"}. Next: correct them by hand (step 4).'

    def _open_folder(self, _):
        p = self._dir() / 'painted'
        if not p.is_dir():
            self._say('Nothing to open yet: export frames (step 2) and make drafts (step 3) first.')
        elif os.name == 'nt':
            os.startfile(str(p))                                  # the dashboard runs on this computer
            self._say(f'Opened {p}. The frames are in {p.parent / "images"}.')
        else:
            self._say(f'The paint layers are in {p}; the frames in {p.parent / "images"}.')

    def _save_rules(self, _):
        d = self._dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / RULES_FILE).write_text(self.rules_in.value, encoding='utf-8')
        self._say(f'Saved the painting rules as {d / RULES_FILE}.')

    def _masks(self):
        d = self._dir()
        if not (d / 'painted').is_dir():
            raise RuntimeError('no paint layers in this set: make drafts (step 3) or paint (step 4) first.')
        ct.convert_masks(d)
        if ct.check(d) != 0:
            raise RuntimeError('the check found problems (listed in the output below).')
        return f'Done: masks of {d.name} are ready. Do the same for the other set, then train (6a or 6b).'

    # ── step 6 ───────────────────────────────────────────────────────────────────
    def _need_masks(self, d):
        if not any((d / 'masks').glob('*.png')):
            raise RuntimeError(f'{d.relative_to(TRAIN_ROOT).as_posix()} has no masks yet: make them (step 5).')

    def _crops(self):
        d = TRAIN_ROOT / self.train_set.value
        self._need_masks(d)
        dest = ct.make_crops(d, int(self.size_in.value), int(self.per_in.value))
        return f'Done: crops are in {dest}. Next: train the random forest.'

    def _train_forest(self):
        name = self.rf_name.value.strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise RuntimeError('the model name may contain only letters, digits, _ . and -.')
        d = TRAIN_ROOT / self.train_set.value
        if self.rf_on_in.value == 'crops':
            d = d.parent / (d.name + '_crops')
            if not d.is_dir():
                raise RuntimeError('no crops yet: press Make crops, or choose "Full frames".')
        self._need_masks(d)
        pkl = cb.train_forest(d, name, self._log)
        return f'Done: random forest saved as {pkl}. Next: score it on the test set (step 7).'

    def _train_network(self):
        d = TRAIN_ROOT / self.train_set.value
        self._need_masks(d)
        if ct.check(d) != 0:
            raise RuntimeError('the training set has problems (listed below): fix them in step 5 first.')
        name = self.nn_name.value.strip()
        info = self.ready.get('info') or {}
        if info.get('torch') and not info.get('patched'):
            raise RuntimeError('the CHROMAS training patch is not applied (step 0): chromas train would stop at once.')
        model = cb.train_network(d, name, int(self.epochs_in.value), int(self.val_in.value), self.start_in.value or None,
                                 self._log)
        self.run_sel.options = {'(choose a run)': ''} | run_results()
        self.run_sel.value = str(model.parent)
        return f'Done: network saved as {model}. Its plots are shown under 6b. Next: score it (step 7).'

    # ── step 7 ───────────────────────────────────────────────────────────────────
    def _score(self):
        d = TRAIN_ROOT / self.test_set.value
        self._need_masks(d)
        if self.test_set.value == self.train_set.value:
            self._log('NOTE: the test set is the training set: these scores show memorising, not how well it works.')
        if not (self.score_lookup.value or self.score_forests.value or self.score_nets.value):
            raise RuntimeError('choose a lookup table, a random forest or a network to score.')
        out = d / f'scores_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        cb.score_models(d, self.score_lookup.value or None, list(self.score_forests.value), self._log,
                        networks=list(self.score_nets.value), out=out)
        self._show_scores(out)
        return f'Done: scores below (and in {out.name} in the test set\'s folder).'

    def _show_scores(self, csv):
        try:
            df = pd.read_csv(csv)
        except Exception:
            self.score_view.objects = []
            return
        g = df.groupby('method', sort=False)
        mean = g[['dice', 'iou', 'precision', 'recall']].mean()
        tot = g[['n_found', 'n_true', 'n_extra']].sum()
        table = pd.DataFrame({'Dice': mean['dice'].round(3), 'IoU': mean['iou'].round(3),
                              'precision': mean['precision'].round(3), 'recall': mean['recall'].round(3),
                              'found': tot['n_found'].astype(str) + ' of ' + tot['n_true'].astype(str),
                              'extra': tot['n_extra']}).reset_index()
        best = table.loc[table['Dice'].idxmax(), 'method']
        self.score_view.objects = [
            pn.pane.Markdown(f'**Mean over {df["image"].nunique()} test frames** (highest Dice: {best})'),
            pn.widgets.Tabulator(table, show_index=False, disabled=True, theme='simple', sizing_mode='stretch_width',
                                 height=40 + 34 * len(table))]

    # ── step 8 ───────────────────────────────────────────────────────────────────
    def _use(self, _):
        import chromas_view
        model = self.use_in.value
        form = chromas_view.run_form_widgets()
        if not model:
            self._say('Failed: no trained model yet (6a or 6b).')
            return
        if form is None:
            self._say(f'Failed: the Run CHROMAS form is not open yet. Open the Datasets tab once, then press this again '
                      f'(or put {model} in its Model file box).')
            return
        method_in, weights_in = form
        method_in.value = 'randomforest' if model.endswith('.pkl') else 'neuralnet'
        weights_in.value = model
        self._say(f'Done: Run CHROMAS in the Datasets tab now uses {method_in.value} with {Path(model).name}.')


def training_panel():
    """The Training tab."""
    return TrainingTab().panel
