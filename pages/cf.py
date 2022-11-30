import pandas as pd
from dash import Input, Output, get_app
from dash import register_page, dash_table, dcc, State, no_update

from lsal_dash import *
from lsal_dash.shared import SharedData, LigexpData

register_page(__name__, path='/cf', description="Counterfactual")
app = get_app()
_mo_options = LigexpData['mo_options'].copy()
_rp_options = LigexpData['rp_options'].copy()
_label_to_value = LigexpData['label_to_value'].copy()
_value_to_label = {k: v for v, k in _label_to_value.items()}
MO_RADIO = "cf_mo_radio"
RP_RADIO = "cf_rp_radio"

_rp_options = [
    opt for opt in _rp_options if opt['value'] not in [
        'rank_average_pred_mu',
        'rank_average_pred_uci',
        'rank_average_pred_uci_top2%uci',
    ]
]


def get_btnpop_sugglist():
    mo_checklist_div = html.Div(
        dbc.RadioItems(
            id=MO_RADIO,
            className="btn-group flex-wrap",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary mb-2",
            labelCheckedClassName="active",
            options=_mo_options,
            value=_mo_options[0]['value']
        ),
        className="radio-group",
    )

    rp_radio_div = html.Div(
        dbc.RadioItems(
            id=RP_RADIO,
            className="btn-group flex-wrap",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary mb-2",
            labelCheckedClassName="active",
            options=_rp_options,
            value=_rp_options[0]['value']
        ),
        className="radio-group",
    )
    content = html.Div([
        dcc.Markdown(f"**Model**", className="form-check"),
        mo_checklist_div,
        dcc.Markdown(f"**Acquisition**", className="form-check"),
        rp_radio_div,
    ])

    btnpop_z = BtnPop.from_name(
        name="btnpop_sugg",
        btn_label="Suggestion list",
        popover_header="Select a model and an acquisition function (ranking parameter)",
        popover_content=content
    )
    return btnpop_z


BTNPOP_SUGG = get_btnpop_sugglist()
app.callback(
    Output(BTNPOP_SUGG.popover.id, 'is_open'),
    Input(BTNPOP_SUGG.button.id, 'n_clicks'),
    State(BTNPOP_SUGG.popover.id, 'is_open')
)(toggle_pop)

selection_table = dash_table.DataTable(
    id="cf__selection_table",
    markdown_options={'html': True},
    columns=[
        {
            'id': c, 'name': c,
            'presentation': 'markdown',
        }
        for c in (
            "Label",
            # "Structure"
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
    style_cell={'textAlign': 'left', 'padding': '5px'},
    style_filter_conditional=[
        {
            'if': {'column_id': 'Structure'},
            'pointer-events': 'None',
        },
    ]
)


@app.callback(
    Output("sugg_df", "data"),
    Input(MO_RADIO, "value"),
    Input(RP_RADIO, "value"),
)
def update_sugg_df(mo_v, rp_v):
    mo_v = mo_v.replace("MODEL=", "")
    subdir = f"{mo_v}___{rp_v}"
    parquet_paths = sorted(glob.glob(f"{FOLDER_ASSETS_CFPOOL}/{subdir}/*.parquet"))
    labs = []
    for p in parquet_paths:
        lab = get_basename(p)
        labs.append(lab)
    df_ligand = pd.DataFrame({'Label': labs})
    return df_ligand.to_dict(orient="records")


@app.callback(
    Output("cf__selection_table", "data"),
    Input("sugg_df", "data"),
)
def update_table(records):
    df = pd.DataFrame.from_records(records)
    df['id'] = df['Label']
    return df.to_dict(orient="records")


@app.callback(
    Output("cf__scatter_graph", "figure"),
    Input('cf__selection_table', 'active_cell'),
    Input(MO_RADIO, "value"),
    Input(RP_RADIO, "value"),
)
def update_cf_card(active_cell, mo, rp):
    if active_cell is None:
        return blank_fig()
        # cs.append(
        #     html.P("Select a reference molecule...")
        # )
        # return cs
    lab = active_cell['row_id']  # this magic comes from https://stackoverflow.com/questions/73390277
    df = get_cfpool_path(mo.replace("MODEL=", ""), rp, lab)
    try:
        df = pd.read_parquet(df)
    except FileNotFoundError:
        return blank_fig()
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
        # dragmode='select',
        yaxis=dict(title=f"\u0394 {_value_to_label[rp]}"),
        xaxis=dict(title="Similarity"),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
        # width=500,
        # height=250,
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig
    #
    # df['id'] = df['Label']
    # return df.to_dict(orient="records")


@app.callback(
    Output('cf__ref_card', 'children'),
    Input('cf__selection_table', 'active_cell'),
)
def update_refcard(
        active_cell,
):
    cs = [
        dbc.CardHeader("Reference"),
    ]

    if active_cell is None:
        cs.append(
            html.P("Select a reference molecule...")
        )
        return cs
    lab = active_cell['row_id']  # this magic comes from https://stackoverflow.com/questions/73390277
    svg_path = get_svg_path(lab, img=True)
    ligand_info = SharedData['df_ligand'].set_index("LigandLabel").loc[lab]
    smi = ligand_info['LigandSmiles']
    cas = ligand_info['LigandCas']
    cs.append(
        html.Div(
            [
                dbc.CardImg(id="cf__img_ref", className="img-fluid", src=svg_path),
                html.Div(
                    [
                        html.B(lab, className="text-center mb-0 pb-0",
                               style={"color": "darkblue", "overflow-wrap": "break-word", "width": "100%"}),
                        html.P([html.B("SMILES:"), smi], style={"overflow-wrap": "break-word", "width": "100%"}),
                        html.P([html.B("CAS:"), cas], style={"overflow-wrap": "break-word", "width": "100%"}),
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
    Output("cf__tooltip", "show"),
    Output("cf__tooltip", "bbox"),
    Output("cf__tooltip", "children"),
    Input("cf__scatter_graph", "hoverData"),
    Input(RP_RADIO, "value"),
)
def display_hover(hoverData, rp_v):
    if hoverData is None or rp_v is None:
        return False, no_update, no_update
    rp_l = _value_to_label[rp_v]
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    # print(pt)
    # print(bbox)

    customdata = pt['customdata']

    cluster_base, is_taught_base, ligand_label_cf, similarity, rank_value_base, rank_value_cf, rank_value_delta = customdata
    svg_path = get_svg_path(ligand_label_cf, True)
    img = dbc.CardImg(id="cf__img_ref", className="img-fluid", src=svg_path)

    children = html.Div([
        html.H6(ligand_label_cf, className="text-center mb-0 pb-0",
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


tab1_content = html.Div(
    [
        dcc.Store(id='sugg_df'),
        dbc.Row(
            [
                html.Div(
                    [
                        BTNPOP_SUGG.get_div(),
                        selection_table,
                    ],
                    className='col-2'
                ),
                html.Div(
                    dbc.Card(
                        id='cf__ref_card',
                        style={"height": "100%"},
                        className="px-4"
                    ),
                    className="col-4"
                ),
                html.Div(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Counterfactual"),
                                dcc.Graph(id="cf__scatter_graph", clear_on_unhover=True),
                                dcc.Tooltip(id="cf__tooltip", direction="left",
                                            background_color="rgba(255,255,255,1.0)"),
                            ],
                            id='cf__cf_card',
                            style={"height": "100%"}, className="px-2"
                        ),
                    ],
                    className="col-6",
                ),
            ],
            className="mt-3"
        )
    ],
    id="pool_content"
)

tab2_content = "Under construction..."


@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    return html.P("This shouldn't ever be displayed...")


tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Pool", tab_id="tab-1"),
                dbc.Tab(label="Mutation", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ]
)

layout = html.Div(
    [
        html.H2("Counterfactual"),
        html.Hr(),
        tabs,
    ]
)
