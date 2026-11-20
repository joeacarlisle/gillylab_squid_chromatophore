"""A full-screen button and an opaque white background on every Bokeh plot of the dashboard.

White background: the dashboard theme makes a figure's border transparent, so a plot downloaded with the toolbar's
save button had a transparent area around it (a checkerboard in image viewers).  _white_figure() makes the figure
opaque white, with dark labels (the same fix as the CTD dashboard).

Bokeh's FullscreenTool adds a button to a plot's toolbar that expands that plot to fill the screen (Esc returns).
HoloViews has no option for it, so a plot hook adds it.  enable() sets the hook as a default for the element types
the dashboard draws; a plot made of several elements gets one button.  Call enable() after hv.extension('bokeh').
"""
import holoviews as hv
from bokeh.models import FullscreenTool

ELEMENTS = ('Curve', 'Scatter', 'Points', 'Image', 'RGB', 'HeatMap', 'Histogram', 'Bars', 'Polygons', 'Labels',
            'Rectangles', 'VSpan', 'HSpan', 'VLine', 'HLine', 'Bounds', 'QuadMesh', 'Area', 'Spikes', 'Path')
_done = {'on': False}


def _white_figure(fig):
    """Opaque white figure, so a downloaded PNG has no transparent (checkerboard) area."""
    from bokeh.models import ColorBar
    fig.background_fill_color = 'white'
    fig.border_fill_color = 'white'
    # The dashboard theme sets border_fill_alpha = 0.  Bokeh does not send a property that equals its default (1.0),
    # so the theme would win; 0.999 is sent and still exports fully opaque.
    fig.background_fill_alpha = 0.999
    fig.border_fill_alpha = 0.999
    fig.outline_line_color = '#888888'
    for ax in list(fig.xaxis) + list(fig.yaxis):
        ax.axis_label_text_color = '#222222'
        ax.major_label_text_color = '#222222'
    if fig.title is not None:
        fig.title.text_color = '#222222'
    for side in (fig.right, fig.left, fig.above, fig.below):
        for item in side:
            if isinstance(item, ColorBar):
                item.background_fill_color = 'white'
                item.background_fill_alpha = 0.999
                item.major_label_text_color = '#222222'
                item.title_text_color = '#222222'


def _add_fullscreen(plot, element):
    fig = plot.state
    try:
        _white_figure(fig)
    except Exception:                     # never let styling break a plot
        pass
    tools = getattr(fig, 'tools', None)
    if tools is None or getattr(fig, 'toolbar_location', 'right') is None:
        return
    if not any(isinstance(t, FullscreenTool) for t in tools):
        fig.add_tools(FullscreenTool())


def enable():
    if _done['on']:
        return
    opts = []
    for name in ELEMENTS:
        el = getattr(hv.opts, name, None)
        if el is not None:
            opts.append(el(hooks=[_add_fullscreen]))
    hv.opts.defaults(*opts, backend='bokeh')
    _done['on'] = True
