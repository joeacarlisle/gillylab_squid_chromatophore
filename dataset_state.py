"""The CHROMAS dataset chosen in the dashboard, shared by the Datasets, Chromatophores and Activation tabs.

The sidebar holds one selector; every CHROMAS tab follows it.  A dataset is listed as soon as it exists anywhere:
loaded in the database (gl_chromas.duckdb, for the Chromatophores tab) or exported (processed/chromas_export/<name>/,
for the Activation tab).  status() says how far each one is, so a tab can say what to do instead of failing.

When an import, a CHROMAS run or an analysis finishes, it calls bump(), which tells every open page to refresh its
list (a callback queued on each page's document; no polling).  poll_while_running() polls a job's progress only
while the job runs: Panel shows its busy spinner during every periodic callback, so permanent polling kept the
spinner going all the time.
"""
import threading

import panel as pn

import chromas_activation as ca
import chromas_bridge as cb

_version = {'n': 0, 'last': None}
_lock = threading.Lock()
_listeners = {}               # bokeh document of an open page -> its refresh function


def bump(select=None):
    """Something about the datasets changed (imported, loaded, analysed).  select: a dataset that open pages should
    switch to (a new import)."""
    with _lock:
        _version['n'] += 1
        _version['last'] = select
        listeners = list(_listeners.items())
    for doc, fn in listeners:                 # safe from a worker thread: bokeh queues it on the page's own loop
        try:
            doc.add_next_tick_callback(lambda fn=fn: fn(select=select))
        except Exception:                     # page closed
            with _lock:
                _listeners.pop(doc, None)


def poll_while_running(poll, is_running, period=1500):
    """Returns start(): call it when a job starts.  It runs poll() every `period` ms until is_running() is False,
    then once more, and stops."""
    holder = {'cb': None}

    def tick():
        poll()
        if not is_running() and holder['cb'] is not None:
            holder['cb'].stop()
            holder['cb'] = None

    def start():
        if holder['cb'] is None and pn.state.curdoc is not None:
            holder['cb'] = pn.state.add_periodic_callback(tick, period=period)
    return start


def version():
    with _lock:
        return _version['n']


def status(name):
    """{'in_db', 'exported', 'positions', 'analysed', 'synthetic'} for one dataset."""
    exp = ca.EXPORT_DIR / name if name else None
    return {'in_db': bool(name) and any(r[0] == name for r in cb.list_datasets()),
            'exported': bool(exp) and (exp / 'info.json').exists(),
            'positions': bool(exp) and (exp / 'centroids.csv').exists(),
            'analysed': bool(name) and (ca.OUT_DIR / name / 'summary.json').exists()}


def all_datasets():
    """Every dataset in the database or exported, newest first."""
    times = {}
    for r in cb.list_datasets():                     # (dataset_id, ..., imported_utc)
        times[r[0]] = str(r[6])
    if ca.EXPORT_DIR.exists():
        from datetime import datetime, timezone
        for d in ca.EXPORT_DIR.iterdir():
            if (d / 'info.json').exists():
                t = datetime.fromtimestamp((d / 'info.json').stat().st_mtime, timezone.utc).isoformat(timespec='seconds')
                times[d.name] = max(times.get(d.name, ''), t)
    return sorted(times, key=lambda n: times[n], reverse=True)


def status_markdown(name):
    if not name:
        return '*No CHROMAS dataset yet: add one in the **Datasets** tab.*'
    s = status(name)
    tick = lambda ok: '✓' if ok else '✗'
    lines = [f"Database {tick(s['in_db'])} · positions {tick(s['positions'])} · analysed {tick(s['analysed'])}"]
    if not s['in_db']:
        lines.append('Load it in the **Datasets** tab to see it in **Chromatophores**.')
    if not s['positions']:
        lines.append('Import it again from its `.zarr` in the **Datasets** tab to analyse it in **Activation**.')
    elif not s['analysed']:
        lines.append('Press **Analyse dataset** in the **Activation** tab.')
    return '  \n'.join(lines)


def make_selector():
    """The sidebar selector and its status line.  Returns (selector, status pane, refresh function)."""
    sel = pn.widgets.Select(name='CHROMAS dataset', options=[], width=260)
    info = pn.pane.Markdown('', width=270, margin=(0, 10))
    seen = {'v': -1}

    def refresh(*_, select=None):
        """Re-read the list; keep the choice (or take `select`), and make every tab redraw."""
        names = all_datasets()
        keep = select if select in names else (sel.value if sel.value in names else (names[0] if names else None))
        if list(sel.options) != names:
            sel.options = names
        if sel.value != keep:
            sel.value = keep
        else:
            sel.param.trigger('value')                # same dataset, new content: redraw the tabs
        seen['v'] = version()

    sel.param.watch(lambda e: setattr(info, 'object', status_markdown(e.new)), 'value')
    refresh()
    info.object = status_markdown(sel.value)
    doc = pn.state.curdoc
    if doc is not None:
        with _lock:
            _listeners[doc] = refresh

        def _gone(_ctx, doc=doc):
            with _lock:
                _listeners.pop(doc, None)
        try:
            doc.on_session_destroyed(_gone)
        except Exception:
            pass
    return sel, info, refresh
