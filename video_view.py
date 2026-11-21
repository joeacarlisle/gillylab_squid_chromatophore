"""Video tab for the dashboard: prepare a video for CHROMAS.

Runs the ffprobe and ffmpeg steps (video_prep.py): check the video, see the first frame with the crop drawn on it,
make an H.264 .mp4 copy (cropped and/or trimmed if wanted) in the CHROMAS data folder, and check that the copy has the
same frames and frame rate.  The original is never changed.  Runs in a background thread, one job at a time.
"""
import threading
import time
from pathlib import Path

import pandas as pd
import panel as pn

import chromas_bridge as cb
import dataset_state
import video_prep as vp

_job = {'running': False, 'log': [], 'status': '', 'started': 0.0}
_lock = threading.Lock()


def _log(text):
    with _lock:
        _job['log'].extend(l for l in str(text).splitlines() if l.strip())


def video_panel():
    st = cb.settings()
    src_in = pn.widgets.TextInput(name='Original video (file name or full path)', placeholder='original.avi', width=520)
    check_btn = pn.widgets.Button(name='Check video', button_type='primary', width=130)
    crop_in = pn.widgets.TextInput(name='Crop: x y width height (empty = whole frame)', placeholder='1000 0 920 1080', width=300)
    at_in = pn.widgets.FloatSlider(name='Preview time (s)', value=0.0, start=0.0, end=10.0, step=0.05, width=260)
    full_btn = pn.widgets.Button(name='Show full frame', width=140)
    frame_btn = pn.widgets.Button(name='Show frame with crop', width=170)
    grid_btn = pn.widgets.Button(name='Show 4 frames', width=130)
    advice_btn = pn.widgets.Button(name='Check crop', width=110)
    reset_btn = pn.widgets.Button(name='Reset crop and options', width=190)
    click_btn = pn.widgets.Button(name='Make clicking copy at the preview time', width=290)
    rev_in = pn.widgets.Checkbox(name='Reverse (the preview-time frame becomes the first frame)', value=False)
    name_in = pn.widgets.TextInput(name='Output name (empty = automatic)', placeholder='cropped.mp4', width=260)
    crf_in = pn.widgets.IntInput(name='CRF (quality)', value=16, start=0, end=40, step=1, width=110)
    start_in = pn.widgets.FloatInput(name='Start (s)', value=0.0, start=0.0, step=1.0, width=100)
    len_in = pn.widgets.FloatInput(name='Length (s, 0 = to end)', value=0.0, start=0.0, step=1.0, width=150)
    data_in = pn.widgets.Checkbox(name='Save in the CHROMAS data folder', value=True)
    over_in = pn.widgets.Checkbox(name='Replace an existing file', value=False)
    make_btn = pn.widgets.Button(name='Make H.264 copy and check it', button_type='success', width=240)
    status = pn.pane.Alert('', alert_type='info', visible=False, sizing_mode='stretch_width')
    bar = pn.indicators.Progress(name='Working', active=False, visible=False, width=300, bar_color='primary')
    info_md = pn.pane.Markdown('', sizing_mode='stretch_width')
    checks_tbl = pn.widgets.Tabulator(pd.DataFrame(columns=['Check', 'Result', 'Detail']), show_index=False, disabled=True,
                                      visible=False, sizing_mode='stretch_width', layout='fit_data_stretch')
    frame_pane = pn.pane.PNG(None, visible=False, sizing_mode='scale_width', max_width=900)
    advice_md = pn.pane.Markdown('', visible=False, sizing_mode='stretch_width')
    log_view = pn.widgets.TextAreaInput(name='Output (newest first)', value='', disabled=True, height=180, sizing_mode='stretch_width')
    state = {'info': None, 'path': None, 'last': None}

    def _source():
        text = src_in.value.strip().strip('"')
        if not text:
            raise ValueError('Type the name of the video first (a name is looked for in the CHROMAS data folder, then in '
                             'videos/; or give the full path, for example C:\\Users\\me\\Videos\\clip.avi).')
        return cb.find_video(text, st)

    def _crop(info):
        crop = vp.parse_crop(crop_in.value)
        if crop and info:
            vp.check_crop(crop, info)
        return crop

    def _start(fn, label):
        with _lock:
            if _job['running']:
                return
            _job.update(running=True, log=[], status=f'{label} ...', started=time.time())

        def work():
            try:
                msg = f'Done: {fn()}'
            except Exception as exc:
                msg = f'Failed: {exc}'
            with _lock:
                _job.update(running=False, status=msg)
        threading.Thread(target=work, daemon=True).start()
        poller()

    def _check(_):
        def run():
            path = _source()
            info = vp.probe(path, _log)
            state.update(info=info, path=path)
            _set_range(info)
            info_md.object = (f'**{path.name}**  \n{vp.describe(info)}  \n'
                              f'Measured with {info["tool"]}. '
                              + ('This is **not H.264**: make a copy (below) before CHROMAS.' if info['codec'] != 'h264'
                                 else 'Already H.264.'))
            return f'{path.name}: {vp.describe(info)}'
        _start(run, 'Checking the video')

    view_btns = (full_btn, frame_btn, grid_btn, advice_btn)

    def _mark(active):
        """Highlight the view button last used."""
        for b in view_btns:
            b.button_type = 'primary' if b is active else 'default'

    def _at(info):
        """The 'Frame at' time, kept inside the video (the last frame is at duration - 1 frame)."""
        last = max(0.0, info['seconds'] - (1.0 / info['fps'] if info['fps'] else 0.0))
        return min(float(at_in.value or 0.0), last)

    def _set_range(info):
        """The preview slider runs from 0 to the last frame of the video."""
        last = max(0.05, info['seconds'] - (1.0 / info['fps'] if info['fps'] else 0.0))
        at_in.end = round(last, 2)
        if at_in.value > at_in.end:
            at_in.value = at_in.end

    def _info_for(path):
        info = state['info'] if state['path'] == path else vp.probe(path, _log)
        state.update(info=info, path=path)
        _set_range(info)
        return info

    def _frame(_):
        def run():
            path = _source()
            info = _info_for(path)
            crop = _crop(info)
            at = _at(info)
            frame_pane.object = vp.first_frame(path, crop, at)
            frame_pane.visible = True
            return (f'frame at {at:.2f} s; ' + ('the red rectangle is the part that will be kept' if crop else 'the whole frame (no crop given)'))
        _mark(frame_btn); state['last'] = 'frame'
        _start(run, 'Reading a frame')

    def _full(_):
        def run():
            path = _source()
            info = _info_for(path)
            at = _at(info)
            frame_pane.object = vp.first_frame(path, None, at)
            frame_pane.visible = True
            return f'the whole frame, {info["width"]} x {info["height"]} px, at {at:.2f} s (no crop drawn)'
        _mark(full_btn); state['last'] = 'full'
        _start(run, 'Reading a frame')

    def _grid(_):
        def run():
            path = _source()
            info = _info_for(path)
            crop = _crop(info)
            frame_pane.object = vp.frames_grid(path, crop, 4)
            frame_pane.visible = True
            return ('4 frames from the start to the end' + ('; the red rectangle is the crop' if crop else ' (no crop given)'))
        _mark(grid_btn); state['last'] = None
        _start(run, 'Reading frames')

    def _advice(_):
        def run():
            path = _source()
            info = _info_for(path)
            crop = _crop(info)
            icon = {'ok': 'OK', 'warn': 'CHECK', 'info': 'NOTE'}
            advice_md.object = '**Checks of this crop**' + ('' if crop else ' (no crop typed: the whole frame)') + '\n\n' + '\n'.join(
                f'- **{icon[l]}** {t}' for l, t in vp.crop_advice(path, info, crop))
            advice_md.visible = True
            return 'checks are shown below the buttons'
        _mark(advice_btn); state['last'] = None
        _start(run, 'Checking the crop')

    def _show_result(res, hint):
        state.update(info=res['before'])
        checks_tbl.value = pd.DataFrame([(n, r.upper(), t) for n, r, t in res['checks']], columns=['Check', 'Result', 'Detail'])
        checks_tbl.visible = True
        info_md.object = (f'**Copy:** `{res["out"]}`  \n{vp.describe(res["after"])}  \nRecord: `{res["record"]}`  \n' + hint)
        if not res['ok']:
            raise RuntimeError('a check failed (see the table): do not use this copy')

    def _make(_):
        def run():
            path = _source()
            crop = vp.parse_crop(crop_in.value)
            out_dir = st['data_dir'] if data_in.value else path.parent
            res = vp.make_copy(path, name_in.value.strip() or None, crop, int(crf_in.value), float(start_in.value),
                               float(len_in.value), out_dir, over_in.value, log=_log)
            _show_result(res, 'Type this name in the Datasets tab, **Video** box.' if data_in.value else
                         'It is not in the CHROMAS data folder: give its full path in the Datasets tab.')
            return f'{res["out"].name} is ready.'
        _start(run, 'Making the copy')

    def _click(_):
        def run():
            path = _source()
            info = _info_for(path)
            crop = _crop(info)
            out_dir = st['data_dir'] if data_in.value else path.parent
            peak = float(at_in.value)
            res = vp.make_click_copy(path, peak, crop, int(crf_in.value), name_in.value.strip() or None, out_dir, over_in.value,
                                     rev_in.value, log=_log)
            stem = res['out'].stem
            _show_result(res, '**For clicking colours only, not for analysis.** In the Datasets tab put `' + stem + '` in the '
                         '**Video** box, choose **lookup**, and run (CHROMAS opens a window for the clicks). The first frame is the frame at '
                         f'{peak:g} s' + (' (the copy is reversed)' if rev_in.value else '') + '. Afterwards the table is saved in '
                         f'the CHROMAS output folder as `{stem}.lookup_table.npz`: copy it, in that folder, to `<name of the real video>.lookup_table.npz` so the real video uses it.')
            return f'{res["out"].name} is ready ' + ('; ' + ' '.join(res['notes']) if res['notes'] else '')
        _start(run, 'Making the clicking copy')

    def _slider(_):
        """Dragging the preview time shows that frame again, when the last view was a single frame."""
        if state['last'] == 'full':
            _full(None)
        elif state['last'] == 'frame':
            _frame(None)

    def _poll():
        with _lock:
            lines, msg, running, started = list(_job['log'][-80:]), _job['status'], _job['running'], _job['started']
        log_view.value = '\n'.join(reversed(lines))
        status.visible = bool(msg)
        status.object = msg
        status.alert_type = 'info' if running else ('danger' if msg.startswith('Failed') else 'success')
        bar.visible = bar.active = running
        for b in (check_btn, full_btn, frame_btn, grid_btn, advice_btn, make_btn, click_btn):
            b.disabled = running
        if running:
            secs = int(time.time() - started)
            status.object = f'{msg} {secs // 60}:{secs % 60:02d} elapsed'

    poller = dataset_state.poll_while_running(_poll, lambda: _job['running'], period=1000)
    check_btn.on_click(_check)
    full_btn.on_click(_full)
    frame_btn.on_click(_frame)
    grid_btn.on_click(_grid)
    advice_btn.on_click(_advice)
    make_btn.on_click(_make)
    click_btn.on_click(_click)

    def _reset(_):
        # the crop, the preview time and the copy options back to how the tab opened; the video named above and the
        # results of Check video stay
        crop_in.value = ''
        at_in.value = 0.0
        rev_in.value = False
        name_in.value = ''
        crf_in.value, start_in.value, len_in.value = 16, 0.0, 0.0
        data_in.value, over_in.value = True, False
        frame_pane.visible = advice_md.visible = False
    reset_btn.on_click(_reset)
    at_in.param.watch(_slider, 'value_throttled')

    intro = pn.pane.Markdown(
        '### Prepare a video for CHROMAS\n'
        'CHROMAS wants an H.264 `.mp4` with the same frames as the original, cropped to the region of interest. This tab '
        'does the steps with ffmpeg and ffprobe: **Check video** reads the codec, size, frame '
        'rate and the exact number of frames; **Show full frame** shows the frame as it is, **Show frame with crop** draws the crop on it, and **Show 4 frames** shows four times from start to end; '
        '**Make H.264 copy** writes the copy to the CHROMAS data folder and checks it against the original. '
        '**The original is never changed.** A record (the commands, the checksum of the original, before and after) is '
        'saved in `processed/video_prep/`. CRF 16 is a starting value that lost nothing visible on the 2024 video; lower is larger.',
        sizing_mode='stretch_width')
    return pn.Column(
        intro,
        pn.pane.Markdown(f'The copy goes to `{st["data_dir"]}` unless you untick the box.'),
        pn.Row(src_in, pn.Column(pn.Spacer(height=22), check_btn)),
        pn.Row(crop_in, at_in),
        pn.Row(full_btn, frame_btn, grid_btn, advice_btn, reset_btn),
        pn.pane.Markdown('**Preview time (s)** is the time, in seconds from the start of the video, of the frame that **Show full '
                         'frame** and **Show frame with crop** display (the highlighted button is the one last used; dragging the '
                         'slider shows the new frame). It runs from 0 to the end once the video has been checked. Choose a time '
                         'when the chromatophores you want to measure are **largest**, so the crop is placed around them at their '
                         'full size; in a display that is usually near the peak, not at 0 (where they are small). If you are '
                         'unsure, press **Show 4 frames**, which shows four times from start to end, then set the best time here. '
                         'It does not change the copy, which keeps every frame unless you set Start and Length.',
                         sizing_mode='stretch_width'),
        pn.Card(pn.pane.Markdown(vp.CROP_ADVICE, sizing_mode='stretch_width'), title='Recommendations for cropping',
                collapsed=False, sizing_mode='stretch_width'),
        pn.Row(name_in, crf_in, start_in, len_in),
        pn.pane.Markdown('**Clicking copy (for a lookup table).** CHROMAS shows only the first frame of a video to click colours on. '
                         '**Make clicking copy at the preview time** makes a copy whose first frame is the frame at the preview time, '
                         'so you click on large chromatophores (it keeps at least 130 frames, because CHROMAS drops chunks under 100). '
                         'Tick **Reverse** to put the preview-time frame exactly first when it is near the end. Use the copy only to '
                         'click colours, then use the saved table on the real video.', sizing_mode='stretch_width'),
        pn.Row(rev_in, click_btn),
        pn.Row(data_in, over_in, make_btn),
        status, bar, info_md, advice_md, checks_tbl, frame_pane, log_view, sizing_mode='stretch_width')
