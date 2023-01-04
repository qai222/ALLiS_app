import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, get_app, html
from dash import register_page, dash_table, dcc, State, no_update

from allis_dash.components.base import toggle_pop, blank_fig
from allis_dash.components.btnpop import get_z_components
from allis_dash.mongo_connections import COLL_MODEL, COLL_LIGAND, COLL_CFPOOL
from allis_dash.path import get_svg_path

register_page(__name__, path='/cfpool', description="Counterfactual")
app = get_app()

_cid_store_suggestion_df_records = "sugg_records"
_cid_left_card = "card_left"
_cid_left_card_model_checklist = "cfpool__model_checklist"
_cid_left_card_uscore_radio_id = "cfpool__uscore_radio"
_cid_left_card_btnpop_name = "cfpool__btnpop_suggestion"
_cid_left_card_table = "cfpool__left_table"
_cid_middle_card = "card_middle"
_cid_right_card = "card_right"
_cid_right_card_cf_graph = "cf_graph"
_cid_right_card_cf_graph_tooltip = "cf_tooltip"


def layout():
    content = dbc.Row(
        [
            dcc.Store(id=_cid_store_suggestion_df_records),
            dbc.Row(
                [
                    html.Div(
                        children=[
                            BTNPOP_SUGG.get_div(),
                            SUGGESTION_TABLE,
                        ],
                        style={"max-width": "180px"},
                        className='col-lg-2 mb-3',
                    ),
                    html.Div(
                        dbc.Card(id=_cid_middle_card, style={"height": "100%", }, className="px-4"),
                        style={"max-width": "200px"},
                        className="col-lg-3 mb-3",
                    ),
                    html.Div(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Counterfactual"),
                                    dcc.Graph(id=_cid_right_card_cf_graph, clear_on_unhover=True),
                                    dcc.Tooltip(id=_cid_right_card_cf_graph_tooltip, direction="left",
                                                background_color="rgba(255,255,255,1.0)"),
                                ],
                                id=_cid_right_card,
                                style={"height": "100%"}, className="px-2"
                            ),
                        ],
                        className="col-lg-7 mb-3",
                    ),
                ],
                className="mt-3"
            )
        ],
    )

    main = html.Div(
        [
            html.H2("Counterfactual"),
            html.Hr(),
            content,
            dbc.Row(html.Hr(), className="mt-3")
        ]
    )
    return main


def get_left_card_components():
    # btnpop for selecting suggestion list
    checklist_model, radio_uscore, btnpop_sugg = get_z_components(
        model_checklist_id=_cid_left_card_model_checklist,
        uscore_radio_id=_cid_left_card_uscore_radio_id,
        btnpop_name=_cid_left_card_btnpop_name,
        model_as_radio=True,
    )
    btnpop_sugg.button.children = "Suggestions"
    app.callback(
        Output(btnpop_sugg.popover.id, 'is_open'),
        Input(btnpop_sugg.button.id, 'n_clicks'),
        State(btnpop_sugg.popover.id, 'is_open')
    )(toggle_pop)

    # datatable
    selection_table = dash_table.DataTable(
        id=_cid_left_card_table,
        markdown_options={'html': True},
        columns=[
            {
                'id': c, 'name': c,
                'presentation': 'markdown',
            }
            for c in (
                "Label",
            )
        ],
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_selectable=False,
        row_deletable=False,
        page_action='native',
        page_current=0,
        page_size=5,
        style_table={'height': '100%', 'width': '100%', 'minWidth': '100px'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_filter_conditional=[
            {
                'if': {'column_id': 'Structure'},
                'pointer-events': 'None',
            },
        ]
    )

    return checklist_model, radio_uscore, btnpop_sugg, selection_table


CHECKLIST_MODEL, RADIO_USCORE, BTNPOP_SUGG, SUGGESTION_TABLE = get_left_card_components()


@app.callback(
    Output(_cid_left_card_table, "data"),
    Input(CHECKLIST_MODEL, "value"),
    Input(RADIO_USCORE, "value"),
)
def update_sugg_df(model_id, utility_score):
    list_keyname = f"{utility_score} @ top"
    doc = COLL_MODEL.find_one({"_id": model_id}, {'suggestions': 1})
    ligand_ids = doc['suggestions'][list_keyname]
    df = pd.DataFrame({'Label': ligand_ids})
    df['id'] = df['Label']  # this is important for the magic of `active_cell`
    return df.to_dict(orient="records")


@app.callback(
    Output(_cid_right_card_cf_graph, "figure"),
    Input(_cid_left_card_table, 'active_cell'),
    Input(_cid_left_card_model_checklist, "value"),
    Input(_cid_left_card_uscore_radio_id, "value"),
)
def update_cf_card(active_cell, model_id, uscore):
    if active_cell is None:
        return blank_fig()
    lab = active_cell['row_id']  # this magic comes from https://stackoverflow.com/questions/73390277

    cf_cursor = COLL_CFPOOL.find(
        {"model_id": model_id, "rank_method": uscore, "ligand_label_base": lab},
    )
    df = pd.DataFrame.from_records(list(cf_cursor))

    print(df.columns)
    scatter = go.Scatter(
        x=df['similarity'],
        y=df['rank_value_delta'],
        mode="markers",
        customdata=df,
    )
    fig = go.Figure(data=[scatter, ])
    fig.update_layout(
        autosize=True,
        margin={
            "t": 5,
            "b": 5,
            "l": 5,
            "r": 5,
        }
    )
    fig.update_layout(
        yaxis=dict(title=f"\u0394 {uscore}"),
        xaxis=dict(title="Similarity"),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig


@app.callback(
    Output(_cid_middle_card, 'children'),
    Input(_cid_left_card_table, 'active_cell'),
)
def update_refcard(active_cell, ):
    cs = [dbc.CardHeader("Reference"), ]

    if active_cell is None:
        cs.append(
            html.P("Select a reference molecule...")
        )
        return cs
    lab = active_cell['row_id']  # this magic comes from https://stackoverflow.com/questions/73390277
    svg_path = get_svg_path(lab, img=True)
    ligand_doc = COLL_LIGAND.find_one({"_id": lab}, {"smiles": 1, "cas_number": 1})
    smi = ligand_doc['smiles']
    cas = ";".join(ligand_doc['cas_number'])
    cs.append(
        html.Div(
            [
                dbc.CardImg(className="img-fluid", src=svg_path),
                html.Div(
                    [
                        html.B(lab, className="text-center mb-0 pb-0",
                               style={"color": "darkblue", "overflow-wrap": "break-word", "width": "100%"}),
                        html.P([html.B("SMILES: "), smi], style={"overflow-wrap": "break-word", "width": "100%"}),
                        html.P([html.B("CAS: "), cas], style={"overflow-wrap": "break-word", "width": "100%"}),
                    ],
                    style={"width": "100%"}
                )
            ],
            className="d-flex flex-wrap align-items-center",
            style={"height": "100%"}
        )
    )
    return cs


@app.callback(
    Output(_cid_right_card_cf_graph_tooltip, "show"),
    Output(_cid_right_card_cf_graph_tooltip, "bbox"),
    Output(_cid_right_card_cf_graph_tooltip, "children"),
    Input(_cid_right_card_cf_graph, "hoverData"),
    Input(_cid_left_card_uscore_radio_id, "value"),
)
def display_hover(hoverData, uscore):
    if hoverData is None or uscore is None:
        return False, no_update, no_update
    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    customdata = pt['customdata']

    _, model_id, _, rank_method, ligand_cf, ligand_ref, similarity, rank_value_base, rank_value_cf, rank_value_delta = customdata
    svg_path = get_svg_path(ligand_cf, True)
    img = dbc.CardImg(className="img-fluid", src=svg_path)

    children = html.Div([
        html.H6(ligand_cf, className="text-center mb-0 pb-0",
                style={"color": "darkblue", "overflow-wrap": "break-word"}),
        img,
        html.P(
            [
                html.B("\u0394"),
                " = {:.3f} - {:.3f}".format(rank_value_base, rank_value_cf),
                html.Span(" = {:.3f}".format(rank_value_delta), style={"color": "red"}),
            ]
        ),
        html.P("Similarity: {:.3f}".format(similarity))
    ], style={'width': '250px', 'white-space': 'normal'}),
    return True, bbox, children
