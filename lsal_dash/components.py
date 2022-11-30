import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html


class BtnPop:

    def __init__(self, name: str, button: dbc.Button, popover: dbc.Popover):
        self.popover = popover
        self.button = button
        self.name = name
        self.div_id = f"{name}__btnpop"

    def get_div(self):
        return html.Div([self.button, self.popover], id=self.div_id, className="d-inline")

    @classmethod
    def from_name(cls, name, btn_label: str, popover_header: str, popover_content: html.Div):
        btn_id = f"{name}__btn"
        popover_id = f"{name}__popover"
        btn = dbc.Button(
            children=btn_label,
            id=btn_id,
            color="primary",
            className="mb-3 mx-3",
            n_clicks=0,
        )
        popover = dbc.Popover(
            [
                dbc.PopoverHeader(popover_header),
                dbc.PopoverBody(popover_content),
            ],
            id=popover_id,
            target=btn_id,
            placement="bottom",
            is_open=False,
        )
        return cls(name, btn, popover)


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
