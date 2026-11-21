"""Preparing a video for CHROMAS: the ffmpeg / ffprobe steps, without a Command Prompt.

    probe(path)            ffprobe (or, without it, ffmpeg) -> codec, size, frame rate, exact frame count
    make_copy(src, ...)    the original is never changed: an H.264 .mp4 copy, optionally cropped and/or trimmed, written
                           by default to the CHROMAS data folder; then checked against the original
    first_frame(path)      the first frame as a PNG (bytes), optionally with the crop drawn on it

Every command is passed to ``log`` as a line starting "> ", the same as the CHROMAS steps.  A record of what was done
(commands, SHA-256 of the original, before and after) is written to processed/video_prep/<name>.txt.
"""
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import chromas_bridge as cb

RECORD_DIR = cb.BASE_DIR / 'processed' / 'video_prep'
REVERSE_MAX_BYTES = 1.5e9      # ffmpeg's reverse filter holds every frame in memory
CLICK_MIN_FRAMES = 130         # CHROMAS drops chunks under 100 frames; 30 more in case some frames are blurred


def ffmpeg_exe():
    """The ffmpeg to use: the one bundled with imageio-ffmpeg (always there with the dashboard), else one on the PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which('ffmpeg')
        if found:
            return found
    raise RuntimeError('ffmpeg was not found. Install imageio-ffmpeg in the dashboard environment '
                       '(pip install imageio-ffmpeg) or put ffmpeg on the PATH.')


def ffprobe_exe():
    """ffprobe on the PATH or next to ffmpeg, else None (probe() then uses ffmpeg alone)."""
    found = shutil.which('ffprobe')
    if found:
        return found
    try:
        cand = Path(ffmpeg_exe()).with_name('ffprobe' + ('.exe' if Path(ffmpeg_exe()).suffix == '.exe' else ''))
        return str(cand) if cand.is_file() else None
    except RuntimeError:
        return None


def _run(cmd, log=None):
    if log:
        log('> ' + ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding='utf-8', errors='replace')


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _fps(text):
    """'60/1' -> 60.0"""
    try:
        a, _, b = str(text).partition('/')
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path, log=None):
    """{'codec','width','height','rate' (e.g. '60/1'),'fps','frames' (counted by decoding),'seconds','mb','tool'}.

    With ffprobe this is the command:  ffprobe -v error -select_streams v:0 -count_frames
    -show_entries stream=...  ; without it ffmpeg decodes the video once to count the frames."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    info = {'mb': round(path.stat().st_size / 1e6, 2)}
    probe_exe = ffprobe_exe()
    if probe_exe:
        r = _run([probe_exe, '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                  'stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration', '-of', 'json', path], log)
        if r.returncode != 0:
            raise RuntimeError(f'ffprobe could not read the video: {r.stderr.strip()[-300:]}')
        data = json.loads(r.stdout or '{}')
        st = (data.get('streams') or [{}])[0]
        if not st:
            raise RuntimeError('The file has no video stream.')
        info.update(codec=st.get('codec_name', '?'), width=int(st.get('width', 0)), height=int(st.get('height', 0)),
                    rate=st.get('r_frame_rate', '0/1'), frames=int(st.get('nb_read_frames') or 0), tool='ffprobe')
        info['seconds'] = float(data.get('format', {}).get('duration') or 0)
    else:
        ff = ffmpeg_exe()
        head = _run([ff, '-hide_banner', '-i', path], log).stderr          # ffmpeg exits with 1 here: no output file
        m = re.search(r'Video: (\w+)[^\n]*?, (\d{2,5})x(\d{2,5})', head)
        f = re.search(r'([\d.]+) fps', head)
        if not m:
            raise RuntimeError('ffmpeg could not find a video stream in the file.')
        fps = float(f.group(1)) if f else 0.0
        r = _run([ff, '-v', 'error', '-i', path, '-map', '0:v:0', '-f', 'null', '-progress', 'pipe:1', '-nostats', '-'], log)
        counts = re.findall(r'^frame=(\d+)', r.stdout, re.M)
        info.update(codec=m.group(1), width=int(m.group(2)), height=int(m.group(3)),
                    rate=(f'{int(fps)}/1' if fps == int(fps) else f'{fps}/1'), frames=int(counts[-1]) if counts else 0,
                    tool='ffmpeg (ffprobe not found)')
        info['seconds'] = info['frames'] / fps if fps else 0.0
    info['fps'] = _fps(info['rate'])
    if not info['seconds'] and info['fps']:
        info['seconds'] = info['frames'] / info['fps']
    return info


def describe(info):
    return (f"{info['codec']}, {info['width']} x {info['height']}, {info['rate']} frames/s ({info['fps']:g}), "
            f"{info['frames']} frames, {info['seconds']:.2f} s, {info['mb']} MB")


def parse_crop(text):
    """'x y width height' (spaces or commas) -> (x, y, w, h) ints, or None for an empty box.  Raises ValueError."""
    t = (text or '').strip()
    if not t:
        return None
    parts = [p for p in re.split(r'[\s,;:]+', t) if p]
    if len(parts) != 4 or not all(re.fullmatch(r'\d+', p) for p in parts):
        raise ValueError('The crop is four whole numbers: x y width height (pixels from the top-left corner), '
                         'for example  1000 0 920 1080.')
    return tuple(int(p) for p in parts)


def check_crop(crop, info):
    """Raise ValueError if the crop does not fit the frame or has an odd size (H.264 4:2:0 needs even sizes)."""
    x, y, w, h = crop
    if w <= 0 or h <= 0:
        raise ValueError('The crop width and height must be above 0.')
    if x + w > info['width'] or y + h > info['height']:
        raise ValueError(f"The crop reaches outside the frame: x + width = {x + w} (frame width {info['width']}), "
                         f"y + height = {y + h} (frame height {info['height']}).")
    if w % 2 or h % 2:
        raise ValueError(f'The crop width ({w}) and height ({h}) must both be even numbers: the H.264 format used '
                         'cannot store an odd size. Change one pixel.')


def default_name(src, crop, start, seconds):
    stem = Path(src).stem
    tag = ('_cropped' if crop else '') + ('_trim' if (start or seconds) else '')
    return f'{stem}{tag or "_h264"}.mp4'


def make_copy(src, out_name=None, crop=None, crf=16, start=0.0, seconds=0.0, out_dir=None, overwrite=False,
              log=print, reverse=False):
    """Make the H.264 copy and check it.  Returns {'out','before','after','checks','record'}; ``checks`` is a list of
    (name, 'pass'|'fail'|'info', text).  The original is only read."""
    src = Path(src).resolve()
    before = probe(src, log)
    log('Original: ' + describe(before))
    if crop:
        check_crop(crop, before)
    if not 0 <= crf <= 40:
        raise ValueError('CRF must be between 0 and 40 (16 is visually lossless for this use; lower is larger).')
    if start < 0 or seconds < 0:
        raise ValueError('Start and length must not be negative.')
    if reverse:
        n_out = (seconds if seconds else max(before['seconds'] - start, 0)) * before['fps']
        need = n_out * (crop[2] * crop[3] if crop else before['width'] * before['height']) * 1.5
        if need > REVERSE_MAX_BYTES:
            raise ValueError(f'Reversing needs about {need / 1e9:.1f} GB of memory for {n_out:.0f} frames (the limit here is '
                             f'{REVERSE_MAX_BYTES / 1e9:.1f} GB). Use a shorter Length, or a crop.')
    out_dir = Path(out_dir) if out_dir else cb.settings()['data_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / (out_name or default_name(src, crop, start, seconds))).resolve()
    if out.suffix.lower() != '.mp4':
        out = out.with_suffix('.mp4')
    if out == src:
        raise ValueError('The output would overwrite the original. Choose another output name.')
    if out.exists() and not overwrite:
        raise FileExistsError(f'{out.name} already exists in {out.parent}. Tick "Replace an existing file" or choose '
                              'another output name.')
    cmd = [ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error']
    if start:
        cmd += ['-ss', f'{start:g}']
    cmd += ['-i', src]
    if seconds:
        cmd += ['-t', f'{seconds:g}']
    vf = (['crop={}:{}:{}:{}'.format(crop[2], crop[3], crop[0], crop[1])] if crop else []) + (['reverse'] if reverse else [])
    if vf:
        cmd += ['-vf', ','.join(vf)]
    cmd += ['-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-an', out]
    tmp = out.with_name(out.stem + '.part.mp4')
    cmd[-1] = tmp
    cb._stream(cmd, cb.BASE_DIR, log)                       # logs the command and its output; raises on failure
    tmp.replace(out)
    after = probe(out, log)
    log('Copy: ' + describe(after))

    checks = []
    trimmed = bool(start or seconds)
    exp_w, exp_h = (crop[2], crop[3]) if crop else (before['width'], before['height'])
    checks.append(('Size', 'pass' if (after['width'], after['height']) == (exp_w, exp_h) else 'fail',
                   f"{after['width']} x {after['height']} (expected {exp_w} x {exp_h})"))
    checks.append(('Frame rate', 'pass' if abs(after['fps'] - before['fps']) < 1e-3 else 'fail',
                   f"{after['rate']} (original {before['rate']})"))
    if trimmed:
        exp = (min(seconds, before['seconds'] - start) if seconds else before['seconds'] - start) * before['fps']
        checks.append(('Frames', 'info', f"{after['frames']} frames in the trimmed copy (about {exp:.0f} expected); "
                                         'a trimmed copy cannot match the original count'))
    else:
        checks.append(('Frames', 'pass' if after['frames'] == before['frames'] else 'fail',
                       f"{after['frames']} (original {before['frames']}): CHROMAS needs the same frames"))
    checks.append(('Codec', 'pass' if after['codec'] == 'h264' else 'fail', after['codec']))
    if reverse:
        checks.append(('Reversed', 'info', 'the copy plays backwards: use it only to click colours for a lookup table, never for analysis'))
    ratio = before['mb'] / after['mb'] if after['mb'] else 0
    checks.append(('Size on disk', 'info', f"{after['mb']} MB (original {before['mb']} MB; {ratio:.1f} times smaller)"))
    ok = all(c[1] != 'fail' for c in checks)
    for name, res, text in checks:
        log(f'{res.upper():5s} {name}: {text}')

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    rec = RECORD_DIR / (out.stem + '.txt')
    rec.write_text('\n'.join([
        f'Video preparation record - {datetime.now():%Y-%m-%d %H:%M}',
        f'Original:   {src}', f'SHA-256 of the original: {sha256(src)}', f'Original:   {describe(before)}',
        f'Copy:       {out}', f'Copy:       {describe(after)}',
        f'Crop (x y width height): {" ".join(map(str, crop)) if crop else "none"}',
        f'Trim: start {start:g} s, length {seconds:g} s (0 = to the end)' if trimmed else 'Trim: none',
        f'CRF: {crf}', 'Reversed: yes (plays backwards; for clicking colours only)' if reverse else 'Reversed: no', 'Command: ' + ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd[:-1] + [out]),
        'Checks: ' + ('all passed' if ok else 'A CHECK FAILED'),
        *(f'  {res}: {name} - {text}' for name, res, text in checks), ''
    ]), encoding='utf-8')
    return {'out': out, 'before': before, 'after': after, 'checks': checks, 'record': rec, 'ok': ok}


def first_frame(path, crop=None, at=0.0):
    """PNG bytes of the frame at ``at`` seconds; with ``crop`` a red rectangle shows what would be kept."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if at:
            cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError('Could not read a frame from the video.')
    if crop:
        x, y, w, h = crop
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), max(2, frame.shape[1] // 400))
    scale = min(1.0, 1400 / frame.shape[1])
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', frame)
    return buf.tobytes()


def _sample_frames(path, n=9):
    """n frames spread over the video (not the very first or last), as BGR arrays."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = sorted({int(round(x)) for x in __import__('numpy').linspace(0, max(total - 1, 0), n + 2)[1:-1]}) or [0]
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append((i, f))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError('Could not read frames from the video.')
    return frames


def frames_grid(path, crop=None, n=4):
    """PNG bytes: n frames from the start to the end of the video in a grid, the crop drawn in red on each.
    Chromatophores swell and the skin moves during a display, so a crop is judged on several times, not on one frame."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        idx = sorted({int(round(x)) for x in np.linspace(0, max(total - 1, 0), n)})
        tiles = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                continue
            if crop:
                x, y, w, h = crop
                cv2.rectangle(f, (x, y), (x + w, y + h), (0, 0, 255), max(3, f.shape[1] // 300))
            scale = 700 / f.shape[1]
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            label = f'{i / fps:.2f} s (frame {i})' if fps else f'frame {i}'
            cv2.putText(f, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(f, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(f, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    finally:
        cap.release()
    if not tiles:
        raise RuntimeError('Could not read frames from the video.')
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    ok, buf = cv2.imencode('.png', np.vstack(rows))
    return buf.tobytes()


def crop_advice(path, info, crop=None):
    """Plain-language checks of a crop (or of the whole frame when ``crop`` is None).  A list of (level, text),
    level 'ok', 'warn' or 'info'.  The dark-edge check is a rough guide: it counts dark pixels (pigment, but also an
    electrode or shadow) in a thin band inside each edge of the crop, on 9 frames spread over the video."""
    import cv2
    import numpy as np
    W, H = info['width'], info['height']
    x, y, w, h = crop if crop else (0, 0, W, H)
    out = []
    share = 100.0 * w * h / (W * H)
    out.append(('info', f'The crop is {w} x {h} px, {share:.0f} % of the frame ({W} x {H}). CHROMAS time and memory go roughly '
                        'with the number of pixels, so a smaller crop runs faster.'))
    if crop:
        try:
            check_crop(crop, info)
            out.append(('ok', 'Width and height are even and the crop is inside the frame.'))
        except ValueError as exc:
            out.append(('warn', str(exc)))
    if min(w, h) < 512:
        out.append(('warn', f'A side is under 512 px. Frames for training a model must be at least 512 px on each side; '
                            'a crop this small cannot be used for training.'))
    frames = _sample_frames(path)
    band = 4
    worst = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}
    for _, f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark = g < (np.median(g) - 45)
        c = dark[y:y + h, x:x + w]
        for side, strip in (('left', c[:, :band]), ('right', c[:, -band:]), ('top', c[:band, :]), ('bottom', c[-band:, :])):
            worst[side] = max(worst[side], float(strip.any(axis=0 if side in ('top', 'bottom') else 1).mean()))
    cut = [f'{k} ({100 * v:.0f} % of its length)' for k, v in worst.items() if v >= 0.02]
    if cut:
        out.append(('warn', 'Dark pixels touch the crop edge on the ' + ', '.join(cut) + ' in at least one of 9 sampled frames: a '
                            'chromatophore (or the electrode) may be cut by the edge. Look at "Show 4 frames" and move that edge '
                            'outward, or accept it and note it.'))
    else:
        out.append(('ok', 'No dark pixels touch the crop edges in 9 sampled frames. (A rough check: pale or faint chromatophores '
                          'and chromatophores that expand later are not covered. Look at the pictures.)'))
    return out


CROP_ADVICE = """- **Judge the crop on several times, not one frame.** Press **Show 4 frames**: chromatophores swell and the skin moves during a display, so a crop that fits at the start may cut one at the end.
- **Keep every chromatophore you want measured whole, in every frame.** A chromatophore cut by the edge is measured as smaller than it is. Leave a margin around the ones near the edge so they stay inside when they expand.
- **Leave out what is not skin:** the electrode or probe, the frame edge or vignetting, text or a time stamp, a part that is out of focus (CHROMAS drops blurred frames; a blurred area can cost you frames).
- **Crop only; never shrink the picture.** A chromatophore needs about 10 px across when fully expanded (about 50 px for motor units). This tab cuts out a region at the original resolution and does not resize.
- **Smaller is faster, but not below 512 px a side** if you will paint frames from the cropped video to train a model.
- **Use even numbers** for width and height. The crop is x y width height, in pixels from the top-left corner of the full frame.
- **Use the same crop for every video of one session**, so they can be compared, and write it down (the record this tab saves has it). The dashboard shifts positions back to full-frame pixels afterwards.
- **If in doubt, crop less.** A crop can be tightened later; cut-off skin cannot be recovered. Press **Check crop** for a few automatic checks (they do not replace looking).
"""


def click_plan(info, peak_s, reverse=False, min_frames=CLICK_MIN_FRAMES):
    """(start, seconds, notes) for the copy used only to click colours for a lookup table: its first frame is the frame
    at ``peak_s`` (the largest chromatophores).  CHROMAS drops chunks under 100 frames, so at least ``min_frames`` frames
    must follow.  Forward: starts at the peak, or earlier if too few frames follow.  Reverse: plays [0, peak] backwards,
    so the peak is the first frame (needs ``min_frames`` frames before the peak)."""
    fps = info['fps'] or 0.0
    if not fps:
        raise ValueError('The frame rate of the video is unknown.')
    total = info['frames']
    peak_frame = max(0, min(int(round(peak_s * fps)), total - 1))
    notes = []
    if reverse:
        n = peak_frame + 1
        if n < min_frames:
            raise ValueError(f'Only {n} frames come before {peak_s:g} s, and CHROMAS needs at least {min_frames}. '
                             'Choose a later preview time, or make the copy without reversing.')
        return 0.0, n / fps, notes
    if total - peak_frame >= min_frames:
        return peak_frame / fps, 0.0, notes
    first = total - min_frames
    if first < 0:
        raise ValueError(f'The video has only {total} frames; at least {min_frames} are needed for CHROMAS.')
    notes.append(f'Only {total - peak_frame} frames follow {peak_s:g} s, so the copy starts earlier, at {first / fps:.2f} s '
                 f'(frame {first}), to keep {min_frames} frames. Its first frame is not the largest; tick Reverse to start '
                 'exactly at the preview time.')
    return first / fps, 0.0, notes


def make_click_copy(src, peak_s, crop=None, crf=16, out_name=None, out_dir=None, overwrite=False, reverse=False, log=print):
    """The copy for clicking colours (see click_plan).  Named <video>_click.mp4 unless ``out_name`` is given."""
    src = Path(src).resolve()
    info = probe(src, log)
    start, seconds, notes = click_plan(info, peak_s, reverse)
    for n in notes:
        log(n)
    name = out_name or f'{src.stem}_click.mp4'
    res = make_copy(src, name, crop, crf, start, seconds, out_dir, overwrite, log, reverse)
    res['notes'] = notes
    return res
