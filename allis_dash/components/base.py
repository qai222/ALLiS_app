import plotly.graph_objects as go


def toggle_pop(n, is_open):
    if n:
        return not is_open
    return is_open


def toggle_popclose(is_open, *args):
    if is_open:
        return [False for _ in args]


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig
