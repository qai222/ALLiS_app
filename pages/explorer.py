import json

import numpy as np
import pandas as pd
from dash import dcc, register_page, get_app, Output, Input, no_update, State, ctx, dash_table
from plotly.colors import DEFAULT_PLOTLY_COLORS

from allis_dash import get_basename, dbc, html, go, rgb_to_rgba, remove_list_dup
from allis_dash.components.btnpop import BtnPop, get_xy_components, get_z_components, get_mg_components
from allis_dash.mongo_connections import COLL_CAMPAIGN, COLL_LIGAND, COLL_PREDICTION

app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Explorer",
)

_cid_store_scatter_selection_xy = "store_scatter_selection_xy"
_cid_store_scatter_selection = "store_scatter_selection"
_cid_scatter_graph = "scatter_graph"
_cid_hist_graph = "hist_graph"
_cid_tooltip = "scatter_tooltip"
_cid_prediction_rows = "prediction_rows"
_cid_prediction_pagination = "prediction_pagination"
_cid_prediction_pagination_div = "prediction_pagination_div"
_pagination_lim = 5


def get_color_size_components():
    mcolor_radio = dbc.RadioItems(
        id="mcolor_radio",
        className="btn-group mb-3",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-success",
        labelCheckedClassName="active",
        options=[
            {"label": "color=z", "value": "z"},
            {"label": "color=group", "value": "mg"},
        ],
        value="z",
    )

    msizeuni_radio = dbc.RadioItems(
        id="msizeuni_radio",
        className="btn-group mb-3",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-info",
        labelCheckedClassName="active",
        options=[
            {"label": "size=z", "value": "z"},
            {"label": "size=uniform", "value": "uniform"},
        ],
        value="z",
    )

    btnpop_color_and_size = BtnPop.from_name(
        name="btnpop_color_and_size",
        btn_label="Color & size",
        popover_header="Select color/size for the markers",
        popover_content=html.Div(
            [
                mcolor_radio,
                msizeuni_radio,
            ],
            className="radio-group",
        )
    )
    return msizeuni_radio, mcolor_radio, btnpop_color_and_size


def get_sliders_components():
    mzpercent_slider = dcc.RangeSlider(0, 100, id="MZPERCENT_Slider_Id", value=[0, 100],
                                       tooltip={"placement": "bottom", "always_visible": True})
    msizemulti_slider = dcc.Slider(0, 2, step=0.05, value=1, id="MSIZEMULTI_Slider_Id", marks=None)

    content = html.Div(
        [
            dcc.Markdown("**Z percentile**"),
            mzpercent_slider,
            dcc.Markdown("**Marker size multiplier**"),
            msizemulti_slider,
        ],
        className="radio-group",
        style={"min-width": "400px"}
    )
    btnpop_sliders = BtnPop.from_name(
        name="btnpop_sliders",
        btn_label="Sliders",
        popover_header="Sliders",
        popover_content=content
    )
    return mzpercent_slider, msizemulti_slider, btnpop_sliders


def set_btnpop_callbacks():
    def bp_open(*args):
        nbps = int(len(args) / 2)
        outputs = [False, ] * nbps
        for i in range(nbps):
            bp = BTNPOP_COMPONENTS[i]
            if bp.button.id == ctx.triggered_id:
                outputs[i] = not args[i * 2 + 1]
                return outputs
        return outputs

    bp_inputs = []
    bp_outputs = []
    for bp in BTNPOP_COMPONENTS:
        bp_inputs.append(Input(bp.button.id, "n_clicks"))
        bp_inputs.append(State(bp.popover.id, "is_open"))
        bp_outputs.append(Output(bp.popover.id, "is_open"))
    app.callback(bp_outputs, bp_inputs)(bp_open)


CHECKLISTS_DESCRIPTOR, CHECKLIST_DIMRED, BTNPOP_XY = get_xy_components()
CHECKLIST_MODEL, RADIO_USCORE, BTNPOP_Z = get_z_components()
CHECKLIST_MG, BTNPOP_MG = get_mg_components()
RADIO_MSIZEUNI, RADIO_MCOLOR, BTNPOP_COLORSIZE = get_color_size_components()
MZPERCENT_SLIDER, MSIZEMULTI_SLIDER, BTNPOP_SLIDERS = get_sliders_components()
BTNPOP_COMPONENTS = [BTNPOP_XY, BTNPOP_Z, BTNPOP_MG, BTNPOP_COLORSIZE, BTNPOP_SLIDERS]
set_btnpop_callbacks()


def layout():
    main = html.Div(
        children=[
            html.H2("Ligand Explorer"),
            html.Hr(),
            html.Div(
                [
                    BTNPOP_XY.get_div(),
                    BTNPOP_Z.get_div(),
                    BTNPOP_MG.get_div(),
                    BTNPOP_COLORSIZE.get_div(),
                    BTNPOP_SLIDERS.get_div()
                ],
            ),
            dcc.Store(id=_cid_store_scatter_selection_xy),
            dcc.Store(id=_cid_store_scatter_selection),

            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id=_cid_scatter_graph, clear_on_unhover=True, ),
                        className="col-6",
                    ),
                    dbc.Col(
                        dcc.Graph(id=_cid_hist_graph, style={"height": "100%"}),
                        className="col-6",
                    )
                ]
            ),

            html.Div(
                dbc.Pagination(
                    id=_cid_prediction_pagination,
                    max_value=1,
                    active_page=1,
                    min_value=1,
                    fully_expanded=False,
                    first_last=True,
                    previous_next=True,
                ),
                id=_cid_prediction_pagination_div,
                style={"display": "none"}
            ),

            html.Div(id=_cid_prediction_rows),

            dcc.Tooltip(id=_cid_tooltip, direction="right", background_color="rgba(255,255,255,1.0)")
        ]
    )
    return main


# collect selections made by xy components
@app.callback(
    Output(_cid_store_scatter_selection_xy, "data"),
    *[Input(cli.id, 'value') for cli in CHECKLISTS_DESCRIPTOR] + [Input(CHECKLIST_DIMRED.id, "value"), ],
)
def xy_selected(*args):
    n_des_checklist = len(CHECKLISTS_DESCRIPTOR)
    if all(arg is None for arg in args):
        return no_update
    des_value_lists = args[:n_des_checklist]
    dim_value_list = args[-1]

    des_selected = []
    for lst in des_value_lists:
        if lst is not None:
            des_selected += lst
    dim_selected = [] if dim_value_list is None else dim_value_list
    selected = des_selected + dim_selected
    assert len(selected) <= 2
    sel = {"des": des_selected, "dim": dim_selected}
    return sel


# make sure at most 2 of descriptor_checklist can be selected, or 1 of radio_dimred
@app.callback(
    [Output(cli.id, 'options') for cli in CHECKLISTS_DESCRIPTOR] + [Output(CHECKLIST_DIMRED, "options"), ],
    Input(_cid_store_scatter_selection_xy, 'data'),
)
def xy_disable_options(data):
    des_selected = data['des']
    dim_selected = data['dim']
    selected = des_selected + dim_selected

    output_options = []
    for cli in CHECKLISTS_DESCRIPTOR + [CHECKLIST_DIMRED, ]:
        if len(des_selected) >= 2 or len(dim_selected) >= 1:
            new_options = [
                {
                    "label": option["label"],
                    "value": option["value"],
                    "disabled": option["value"] not in selected,
                }
                for option in cli.options
            ]
        elif len(des_selected) == 1:
            if cli.id == CHECKLIST_DIMRED.id:
                new_options = [
                    {
                        "label": option["label"],
                        "value": option["value"],
                        "disabled": option["value"] not in selected,
                    }
                    for option in cli.options
                ]
            else:
                new_options = cli.options.copy()
        else:
            new_options = cli.options.copy()
        output_options.append(new_options)
    return output_options


# only one or two models can be selected for z
@app.callback(
    Output(CHECKLIST_MODEL, "options"),
    Input(CHECKLIST_MODEL, "value"),
)
def mo_options_disable(selected_mos):
    default_options = CHECKLIST_MODEL.options.copy()
    new_opts = default_options
    if len(selected_mos) == 2:
        new_opts = [
            {
                'label': opt['label'],
                'value': opt['value'],
                'disabled': opt['value'] not in selected_mos
            }
            for opt in default_options
        ]
    return new_opts


def mongo_get_scatter_data(
        ligand_ids: list[str],
        x_colname: str,
        y_colname: str,
        uscore: str = "mu_top2%mu",
        model_id: str = "MODEL:SL0519"
):
    docs = COLL_LIGAND.aggregate(
        [
            {"$match": {"_id": {"$in": ligand_ids}}},
            {"$project": {
                "x": f"${x_colname}",
                "y": f"${y_colname}",
                "z": {
                    "$getField": {
                        "field": uscore,
                        "input": {
                            "$first": {
                                "$filter": {
                                    "input": "$utility_scores",
                                    "as": "uscore_dict",
                                    "cond": {
                                        "$eq": ["$$uscore_dict.model_id", model_id]
                                    }
                                }
                            }
                        }
                    }
                }
            }}
        ]
    )
    doc_list = list(docs)
    return doc_list


# this is the base selection, `MZPERCENT_SLIDER` will *not* affect this
@app.callback(
    Output(_cid_store_scatter_selection, "data"),
    Input(CHECKLIST_MODEL.id, "value"),
    Input(RADIO_USCORE.id, "value"),
    Input(CHECKLIST_MG.id, "value"),
    Input(_cid_store_scatter_selection_xy, "data")
)
def update_selection(model_id, uscore, ligand_groups_selected, selection_xy):
    xy_des = selection_xy['des']
    xy_dim = selection_xy['dim']
    if len(xy_des) == 2:
        x_colname, y_colname = xy_des
    elif len(xy_dim) == 1:
        x_colname, y_colname = xy_dim[0]
    else:
        return None

    dfs = []
    if len(model_id) == 1:
        delta_z = False
        model_id1 = model_id[0]
        model_id2 = None
        z_label = f"{model_id1} @ {uscore}"
        dfs = []
        for lg_json in ligand_groups_selected:
            lg = json.loads(lg_json)
            group_name = lg['group_name']
            ligand_ids = lg['ligand_ids']
            records = mongo_get_scatter_data(ligand_ids, x_colname, y_colname, uscore, model_id1)
            df_group = pd.DataFrame.from_records(records)
            df_group["group_name"] = [group_name, ] * len(df_group)
            dfs.append(df_group)
    elif len(model_id) == 2:
        delta_z = True
        model_id1, model_id2 = model_id
        z_label = f"{model_id1} - {model_id2} @ {uscore}"
        for lg_json in ligand_groups_selected:
            lg = json.loads(lg_json)
            group_name = lg['group_name']
            ligand_ids = lg['ligand_ids']
            records1 = mongo_get_scatter_data(ligand_ids, x_colname, y_colname, uscore, model_id1)
            records2 = mongo_get_scatter_data(ligand_ids, x_colname, y_colname, uscore, model_id2)
            df_1 = pd.DataFrame.from_records(records1)
            df_2 = pd.DataFrame.from_records(records2)
            df_2.drop(columns=["x", "y"], inplace=True)
            df_group = pd.merge(left=df_1, right=df_2, how="inner", on="_id",
                                suffixes=(f"__{model_id1}", f"__{model_id2}"))
            assert len(df_group) == len(df_1) == len(df_2)
            df_group["z"] = df_group[f"z__{model_id1}"] - df_group[f"z__{model_id2}"]
            df_group["group_name"] = [group_name, ] * len(df_group)
            dfs.append(df_group)
    else:
        return None
    if len(dfs) == 0:
        return None
    df = pd.concat(dfs, axis=0)
    # df.drop_duplicates(subset=["_id"], inplace=True)
    df['z_def'] = [z_label, ] * len(df)
    df['model_id1'] = [model_id1, ] * len(df)
    df['model_id2'] = [model_id2, ] * len(df)
    meta = {
        "plotdata": df.to_dict(orient="records"),
        "x_label": x_colname,
        "y_label": y_colname,
        "z_label": z_label,
        "uscore": uscore,
        "delta_z": delta_z,
        "model_id1": model_id1,
        "model_id2": model_id2,
    }
    return meta


@app.callback(
    Output(_cid_scatter_graph, "figure"),
    Output(_cid_hist_graph, "figure"),
    Input(_cid_store_scatter_selection, "data"),
    Input(RADIO_MCOLOR, "value"),
    Input(RADIO_MSIZEUNI, "value"),
    Input(MZPERCENT_SLIDER, "value"),
    Input(MSIZEMULTI_SLIDER, "value"),
)
def update_scatter(data, color_select, size_uni, z_percent_range, size_multi):
    if data is None:
        return no_update

    plot_df = pd.DataFrame.from_records(data['plotdata'])

    z = plot_df['z'].values
    minz = np.percentile(z, z_percent_range[0])
    maxz = np.percentile(z, z_percent_range[1])

    zlim_min = min(z)
    zlim_max = max(z)

    percentile_mask = (plot_df['z'] >= minz) & (plot_df['z'] <= maxz)
    df_scatter_filtered = plot_df[percentile_mask]

    delta_z = data['delta_z']
    colorscale_name = 'RdBu' if delta_z else 'Blues'

    def marker_dict(
            z_array, cbar=True, colorz=True, unisize=size_uni, size_multiplier=size_multi, deltaz=delta_z,
    ):
        if deltaz:
            ss = [abs(v) for v in z_array]
        else:
            ss = z_array
        d = dict(
            colorscale=colorscale_name, color=z_array, size=ss,
            line={"color": "#444"},
            sizeref=2. * max(z_array) / (20. ** 1.5),
            sizemode="diameter",
            opacity=0.6,
            cmin=zlim_min,
            cmax=zlim_max,
        )
        if not cbar:
            try:
                del d['colorbar']
            except KeyError:
                pass
        if not colorz:
            del d['color']
        d["size"] = [siz * size_multiplier for siz in ss]
        if unisize == "uniform":
            d["size"] = [1.0 * size_multiplier for _ in ss]
        return d

    traces = []
    if color_select == "z":
        trace = go.Scattergl(
            x=df_scatter_filtered['x'].values, y=df_scatter_filtered['y'].values, mode="markers",
            marker=marker_dict(df_scatter_filtered['z'], cbar=True, colorz=True, unisize=size_uni),
            customdata=df_scatter_filtered, name="Selected ligands",
            showlegend=True,
        )
        traces.append(trace)

    elif color_select == "mg":
        group_names = remove_list_dup(df_scatter_filtered['group_name'].tolist())
        # for group_name, df_group in df_scatter_filtered.groupby("group_name"):
        for group_name in group_names:
            df_group = df_scatter_filtered.loc[df_scatter_filtered['group_name'] == group_name]
            trace = go.Scattergl(
                x=df_group['x'].values, y=df_group['y'].values, mode="markers",
                marker=marker_dict(df_group['z'], cbar=False, colorz=False, unisize=size_uni),
                customdata=df_group, name=group_name,
                showlegend=True,
            )
            traces.append(trace)
    else:
        raise RuntimeError

    fig = go.Figure(data=traces)
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        dragmode='select',
        legend=dict(
            orientation="v",
            itemwidth=30,
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
            itemsizing='constant',
            bgcolor="rgba(255, 255, 255, 0.3)",
        ),
        legend_title_text=f"{len(df_scatter_filtered)} ligands from {len(traces)} group(s)",
        xaxis=dict(title=data['x_label']),
        yaxis=dict(title=data['y_label']),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
    )

    counts, bins = np.histogram(df_scatter_filtered['z'], bins=min(len(df_scatter_filtered['z']) // 5 + 1, 20))
    bins = 0.5 * (bins[:-1] + bins[1:])
    hist_traces = [go.Bar(x=bins, y=counts, marker={'color': bins, 'colorscale': colorscale_name})]

    hist_fig = go.Figure(data=hist_traces)

    hist_fig.update_layout(
        xaxis=dict(title=data['z_label']),
        yaxis=dict(title="Count"),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
    )

    return fig, hist_fig


@app.callback(
    Output(_cid_tooltip, "show"),
    Output(_cid_tooltip, "bbox"),
    Output(_cid_tooltip, "children"),
    Input(_cid_scatter_graph, "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    customdata = pt['customdata']

    try:
        lab, x, y, z, group_name, z_label, model_id1, model_id2 = customdata
        z_show = "{:.3f}".format(z)
        z = html.Span(f" {z_label} = {z_show}", style={"color": "red"})
    except ValueError:
        lab, x, y, oz1, oz2, dz, group_name, z_label, model_id1, model_id2 = customdata
        z_show = "{:.3f} = {:.3f} - {:.3f}".format(dz, oz1, oz2)
        z = html.Span([html.B("\u0394"), f"({z_label}) = {z_show}"], style={"color": "red"})

    x_show = "{:.3f}".format(x)
    y_show = "{:.3f}".format(y)
    xy = f"xy = ({x_show}, {y_show})"

    children = html.Div([
        html.H6(lab, className="text-center mb-0 pb-0", style={"color": "darkblue", "overflow-wrap": "break-word"}),
        html.Img(src=f"assets/svg/{lab}.svg", style={'height': '150px', 'width': '150px'},
                 className="mt-0 mb-0 rounded mx-auto d-block pb-0 pt-0"),
        html.P([xy, html.Br(), z], style={"overflow-wrap": "break-word"}),
    ], style={'width': '200px', 'white-space': 'normal'}),
    return True, bbox, children


def get_figure_in_pred_card(pred_docs: list[dict], model_id_to_round_index: dict[str, int]) -> go.Figure:
    """
    get the figure appears in a prediction card
    the prediction docs should 1. share the same ligand 2. sorted by model's `round_index`
    """
    traces = []
    pred_docs = sorted(pred_docs, key=lambda d: model_id_to_round_index[d['model_id']])
    for i, doc in enumerate(pred_docs):
        x = np.array(doc['ligand_amounts'])
        y = np.array(doc['mu'])
        yerr = np.array(doc['std'])
        scatter_color = DEFAULT_PLOTLY_COLORS[i]
        fill_color = rgb_to_rgba(scatter_color, 0.2)
        scatter = go.Scatter(
            x=x, y=y, mode="lines", name=doc['model_id'],
            line=dict(color=scatter_color),
        )
        band1 = go.Scatter(
            x=x,
            y=y + yerr,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
        )
        band2 = go.Scatter(
            x=x,
            y=y - yerr,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
            fillcolor=fill_color,
            fill='tonexty',
        )
        traces += [scatter, band1, band2]
    fig_pred = go.Figure(data=traces)
    fig_pred.update_xaxes(type="log")
    fig_pred.update_layout(
        autosize=True,
        margin={
            "t": 0,
            "b": 0,
            "l": 0,
            "r": 0,
        }
    )
    return fig_pred


def get_table_in_pred_card(ligand_label: str, model_id_to_round_index: dict[str, int]):
    """
    a simple table showing utility scores by different models
    1. models should be sorted by `round_index`
    2. only `std`, `mu_top2%mu`, `std_top2%mu` are shown
    """
    utility_scores = COLL_LIGAND.find_one({"_id": ligand_label}, {"utility_scores": 1})['utility_scores']
    df = pd.DataFrame.from_records(utility_scores)
    df = df[['model_id', 'std', 'mu_top2%mu', 'std_top2%mu']]
    df = df.round(decimals=4)
    records = df.to_dict('records')
    records = sorted(records, key=lambda x: model_id_to_round_index[x['model_id']])
    for r in records:
        r['model_id'] = r['model_id'].replace("MODEL:", "")
    table = dash_table.DataTable(
        records, [{"name": i, "id": i} for i in df.columns],
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['Date', 'Region']
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        },
        fixed_columns={'headers': True, 'data': 1},
        page_action='none',
        style_table={'minWidth': '100%', 'overflowY': 'auto', 'height': '100%'},
        style_cell={
            # all three widths are needed
            'minWidth': '30%', 'width': '30%', 'maxWidth': '100%',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        }
    )
    return table


def get_prediction_card(ligand_label: str):
    prediction_docs = list(COLL_PREDICTION.find({"ligand_label": ligand_label}))

    # TODO is this a bit extravagant?
    model_id_to_round_index = {d["_id"].replace("CAMPAIGN", "MODEL"): d["round_index"] for d in
                               COLL_CAMPAIGN.find({}, {"round_index": 1})}

    fig_pred = get_figure_in_pred_card(prediction_docs, model_id_to_round_index)
    tab_pred = get_table_in_pred_card(ligand_label, model_id_to_round_index)

    img = html.Img(
        src=f"assets/svg/{ligand_label}.svg",
        className="img-fluid",
    )

    content = dbc.Row(
        [
            html.Div(img, className="col-2"),
            html.Div(tab_pred, className="col-4", ),
            html.Div(
                dcc.Graph(
                    figure=fig_pred,
                    style={"width": "100%", "height": "100%"},
                ),
                className="col-6",
            )
        ]
    )

    card = dbc.Card(
        [
            dbc.CardHeader(ligand_label),
            dbc.CardBody(content),
        ],
        id=f"cardinrow-{ligand_label}",
        className="mb-3",
        style={"max-height": "300px"},
    )
    return card


@app.callback(
    Output(_cid_prediction_pagination, "max_value"),
    Output(_cid_prediction_pagination_div, "style"),
    Input(_cid_scatter_graph, "selectedData"),
)
def display_pagination_div(data):
    if data is None:
        return 1, {"display": "none"}
    else:
        pts = data['points']
        npages = len(pts) // _pagination_lim + 1
        return npages, {}


@app.callback(
    Output(_cid_prediction_rows, "children"),
    Input(_cid_scatter_graph, "selectedData"),
    Input(_cid_prediction_pagination, "active_page")
)
def update_rows(selectedData, active_page):
    if selectedData is None:
        return None
    pts = selectedData['points']
    pts = pts[_pagination_lim * (active_page - 1): _pagination_lim * active_page]

    rows = []
    for pt in pts:
        customdata = pt['customdata']
        lab = customdata[0]
        card = get_prediction_card(lab)
        rows.append(card)
    return rows
