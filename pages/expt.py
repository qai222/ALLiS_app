from dash import Input, Output, get_app
from dash import register_page, dcc
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from lsal_dash import *
from lsal_dash.shared import *

app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Experiment",
)

_expt_campaign_to_label = ExptData['expt_campaign_to_label']
_expt_options = sorted(_expt_campaign_to_label.keys())


def get_card(lab: str) -> dbc.Card:
    img = html.Img(
        src=f"assets/svg/{lab}.svg",
        className="img-fluid",
    )

    try:
        df_expt = pd.read_parquet(get_expt_path(lab))
    except FileNotFoundError:
        return None

    df_expt = df_expt[df_expt["ReactionType"] == "real"]

    scatter_fom = go.Scatter(
        x=df_expt['LigandAmount'], y=df_expt['FigureOfMerit'], mode="markers", name="FOM",
        marker=dict(color=DEFAULT_PLOTLY_COLORS[0])
        # marker=dict(color=DEFAULT_PLOTLY_COLORS[icolor + 1]),
    )
    scatter_od = go.Scatter(
        x=df_expt['LigandAmount'], y=df_expt['OpticalDensity'], mode="markers", name="OD",
        marker=dict(color=DEFAULT_PLOTLY_COLORS[1])
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(scatter_fom, secondary_y=False)
    fig.add_trace(scatter_od, secondary_y=True)

    fom_mu = df_expt['RefFOMs_mu'].tolist()[0]
    od_mu = df_expt['RefODs_mu'].tolist()[0]
    fom_std = df_expt['RefFOMs_std'].tolist()[0]
    od_std = df_expt['RefODs_std'].tolist()[0]

    fig.add_hrect(
        y0=fom_std + fom_mu, y1=fom_mu - fom_std, line_width=0, fillcolor=DEFAULT_PLOTLY_COLORS[0], opacity=0.1,
        secondary_y=False, name="FOM ref"
    )
    fig.add_hrect(
        y0=od_std + od_mu, y1=od_mu - od_std, line_width=0, fillcolor=DEFAULT_PLOTLY_COLORS[1], opacity=0.1,
        secondary_y=True, name="OD ref"
    )

    fig.add_hline(y=fom_mu, line_dash="dash", line_color=DEFAULT_PLOTLY_COLORS[0])
    fig.add_hline(y=od_mu, line_dash="dash", line_color=DEFAULT_PLOTLY_COLORS[1], secondary_y=True, )

    fig.update_xaxes(type="log")
    fig.update_layout(
        yaxis_range=[0, 2.5],
        hovermode="x unified",
        autosize=True,
        margin={
            "t": 0,
            "b": 0,
            "l": 0,
            "r": 0,
        }
    )

    content = dbc.Row(
        [
            html.Div(img, className="col-3"),
            html.Div(
                dcc.Graph(
                    figure=fig,
                    style={"width": "100%", "height": "100%"},
                ),
                className="col-9",
            )
        ]
    )

    row = dbc.Card(
        [
            dbc.CardHeader(
                [
                    lab,
                ]
            ),
            dbc.CardBody(content),
        ],
        id=f"Expt__card-{lab}",
        className="mb-3",
        style={"max-height": "300px"},
    )
    return row


def get_tab(expt_name: str) -> dbc.Tab:
    tab_id = f"Expt__tab_{expt_name}"
    tab = dbc.Tab(
        label=expt_name,
        tab_id=tab_id,
    )
    return tab


def get_content_by_tab(tab_id: str):
    expt_name = tab_id.replace("Expt__tab_", "")
    rows = []
    for lab in _expt_campaign_to_label[expt_name]:
        rows.append(get_card(lab))
    return rows


TAB_LIST = [get_tab(expt) for expt in _expt_campaign_to_label]
TAB_ID_TO_CONETENT = {tab.tab_id: get_content_by_tab(tab.tab_id) for tab in TAB_LIST}


@app.callback(Output("Expt__content", "children"), [Input("Expt__tabs", "active_tab")])
def expt_switch_tab(at):
    return TAB_ID_TO_CONETENT[at]


tabs = html.Div(
    [
        dbc.Tabs(
            TAB_LIST,
            id="Expt__tabs",
            active_tab=TAB_LIST[0].tab_id,
        ),
        html.Div(id="Expt__content"),
    ]
)

layout = html.Div(
    [
        html.H2("Experiments"),
        html.Hr(),
        tabs,
    ]
)
