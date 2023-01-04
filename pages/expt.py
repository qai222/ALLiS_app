import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, get_app, register_page, dcc, html
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from allis_dash.mongo_connections import COLL_LIGAND, COLL_REACTION, COLL_CAMPAIGN
from allis_dash.utils import get_basename, sort_and_group

app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Experiment",
)

_cid_tabs = "cid_expt_tabs"
_cid_content = "cid_expt_content"


def get_figure_per_ligand(df: pd.DataFrame) -> go.Figure:
    scatter_fom = go.Scatter(
        x=df['LigandAmount'], y=df['FigureOfMerit'], mode="markers", name="FOM",
        marker=dict(color=DEFAULT_PLOTLY_COLORS[0])
    )
    scatter_od = go.Scatter(
        x=df['LigandAmount'], y=df['OpticalDensity'], mode="markers", name="OD",
        marker=dict(color=DEFAULT_PLOTLY_COLORS[1])
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(scatter_fom, secondary_y=False)
    fig.add_trace(scatter_od, secondary_y=True)

    fom_mu = df['RefFOMs_mu'].tolist()[0]
    od_mu = df['RefODs_mu'].tolist()[0]
    fom_std = df['RefFOMs_std'].tolist()[0]
    od_std = df['RefODs_std'].tolist()[0]

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
    return fig


def generate_cards(reaction_ids: list[str]):
    projection = {
        'LigandAmount': 1, 'FigureOfMerit': 1, 'OpticalDensity': 1, "LigandLabel": 1,
        'RefODs_mu': 1, 'RefFOMs_mu': 1, 'RefODs_std': 1, 'RefFOMs_std': 1
    }
    reaction_docs = COLL_REACTION.find({"_id": {"$in": reaction_ids}, "ReactionType": "real"}, projection)
    cards = []
    ligand_labels, reaction_groups = sort_and_group(reaction_docs, keyf=lambda x: x['LigandLabel'])

    ligand_docs = COLL_LIGAND.find({"_id": {"$in": ligand_labels}}, {"SuggestedBy": 1})

    def get_suggestion_badges(lig_doc):
        try:
            suggs = lig_doc['SuggestedBy']
        except KeyError:
            return []
        badges = []
        for sugg in suggs:
            s = f"{sugg['model_id']} @ {sugg['directed_u_score']}"
            b = dbc.Badge(s, color="success", className="mx-1")
            badges.append(b)
        return badges

    lig_lab_to_badges = {d['_id']: get_suggestion_badges(d) for d in ligand_docs}

    for ligand_label, reactions in zip(ligand_labels, reaction_groups):
        img = html.Img(src=f"assets/svg/{ligand_label}.svg", className="img-fluid", )
        df = pd.DataFrame.from_records(reactions)
        fig = get_figure_per_ligand(df)

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

        header = html.Span(
            [
                ligand_label,
                *lig_lab_to_badges[ligand_label],
            ]
        )

        card = dbc.Card(
            [
                dbc.CardHeader(header),
                dbc.CardBody(content),
            ],
            id=f"Expt__card-{ligand_label}",
            className="mb-3",
            style={"max-height": "500px"},
        )
        cards.append(card)
    return cards


@app.callback(
    Output(_cid_content, "children"),
    Input(_cid_tabs, "active_tab"),
)
def expt_switch_tab(active_tab_id):
    # `active_tab_id` is also campaign doc `_id`
    campaign_doc = COLL_CAMPAIGN.find_one({"_id": active_tab_id})
    reaction_ids = campaign_doc["reactions"]
    cards = generate_cards(reaction_ids)
    tags = [
        dbc.Badge(f"{len(reaction_ids)} reactions", color="primary", className="mx-1 mb-3 mt-3"),
        dbc.Badge(f"{len(cards)} ligands", className="mx-1 mb-3 mt-3", color="primary"),
    ]
    if campaign_doc['model_folder'] is None:
        tags.append(
            dbc.Badge(f"extra campaign", className="mx-1 mb-3 mt-3", color="warning"),
        )
    if campaign_doc['round_index'] == 0:
        tags.append(
            dbc.Badge(f"seed dataset", className="mx-1 mb-3 mt-3"),
        )
    return [
        html.Div(tags),
        *cards,
    ]


def layout():
    doc_campaigns = COLL_CAMPAIGN.find({})
    doc_campaigns = sorted(doc_campaigns, key=lambda x: x['round_index'])

    # tab list for clicking
    tab_list = [dbc.Tab(label=doc["_id"], tab_id=doc["_id"]) for doc in doc_campaigns]
    master = html.Div(
        [
            html.H2("Experiment campaigns"),
            html.Hr(),
            dbc.Tabs(
                id=_cid_tabs,
                children=tab_list,
                active_tab=tab_list[0].tab_id
            ),
            html.Div(id=_cid_content)
        ]
    )

    return master
