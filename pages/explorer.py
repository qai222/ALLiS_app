import numpy as np
from dash import dcc, register_page, get_app, Output, Input, no_update, State, ctx
from plotly.colors import DEFAULT_PLOTLY_COLORS

from lsal_dash import *
from lsal_dash.shared import *

# region page setup
app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Explorer",
)

# constants
_descriptor_category_to_names = LigexpData['des_category'].copy()
_des_options = LigexpData['des_options'].copy()
_dim_options = LigexpData['dim_options'].copy()
_mo_options = LigexpData['mo_options'].copy()
_rp_options = LigexpData['rp_options'].copy()
_mg_options = LigexpData['mg_options'].copy()

_mg_all_l = f"All groups ({LigexpData['n_ligands']})"
_mg_all_v = "mg_all"
_mg_options.append({"label": _mg_all_l, "value": _mg_all_v})

_label_to_value = LigexpData['label_to_value'].copy()
_label_to_value[_mg_all_l] = _mg_all_v
_value_to_label = {k: v for v, k in _label_to_value.items()}

# ids
MO_Cheklist_Id = "mo_checklist"
RP_Radio_Id = "rp_radio"
MG_Checklist_Id = "mg_checklist"
MCOLOR_Radio_Id = "mcolor_radio"
MSIZEUNI_Radio_Id = "msizeuni_radio"
MZPERCENT_Slider_Id = "mzpercent_slider"
MSIZEMULTI_Slider_Id = "msizemulti_slider"

# intermediates
store_scatter_selection = dcc.Store(id="LigExp__scatter_selection")
store_scatter_selection_xy = dcc.Store(id="LigExp__scatter_selection_x")


# endregion

# region btnpop_xy
def get_checklist_des(des_cat: str):
    des_names = _descriptor_category_to_names[des_cat]
    des_options = [
        {"label": lab, "value": _label_to_value[lab]} for lab in des_names
    ]
    checklist_id = f"checklist-{des_cat}"
    checklist = dbc.Checklist(
        id=checklist_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=des_options,
    )
    return checklist, {checklist_id: des_options.copy()}


def get_checklist_dim():
    dim_options = _dim_options.copy()
    checklist_id = "checklist-dim"
    checklist = dbc.Checklist(
        id=checklist_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=dim_options,
        value=[dim_options[0]['value'], ],
    )
    return checklist, {checklist_id: dim_options.copy()}


def get_btnpop_xy():
    xy_id_to_options = dict()

    des_checklist_ids = []
    content_children = []
    for cat in _descriptor_category_to_names:
        header = dcc.Markdown(f"**{cat}**", className="form-check")
        clst_des, id_to_options = get_checklist_des(cat)
        content_children += [header, clst_des]
        des_checklist_ids.append(clst_des.id)
        xy_id_to_options.update(id_to_options)

    header = dcc.Markdown(f"**dimensionality reduction**", className="form-check")
    checklist_dim, id_to_options = get_checklist_dim()
    content_children += [header, checklist_dim]
    xy_id_to_options.update(id_to_options)

    content = html.Div(content_children, className="radio-group", )

    btnpop_xy = BtnPop.from_name(
        name="btnpop_xy",
        btn_label="XY axes",
        popover_header="Select any two descriptors, or one parameter set for dimensionality reduction",
        popover_content=content
    )
    return btnpop_xy, des_checklist_ids, checklist_dim.id, xy_id_to_options


BtnPop_XY, Des_Checklist_Ids, Checklist_Dim_Id, Checklist_Id_To_Options = get_btnpop_xy()


@app.callback(
    Output(store_scatter_selection_xy.id, "data"),
    *[Input(cli, 'value') for cli in Des_Checklist_Ids] + [Input(Checklist_Dim_Id, "value"), ],
)
def xy_selected(*args):
    n_des_checklist = len(Des_Checklist_Ids)
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
    return {"des": des_selected, "dim": dim_selected}


# make sure at most 2 of des_checklist can be selected, or 1 of radio_dim
@app.callback(
    [Output(cli, 'options') for cli in Des_Checklist_Ids] + [Output(Checklist_Dim_Id, "options"), ],
    Input(store_scatter_selection_xy, 'data'),
)
def xy_disable_options(data):
    des_selected = data['des']
    dim_selected = data['dim']
    selected = des_selected + dim_selected

    output_options = []
    for cli in Des_Checklist_Ids + [Checklist_Dim_Id, ]:
        if len(des_selected) >= 2 or len(dim_selected) >= 1:
            new_options = [
                {
                    "label": option["label"],
                    "value": option["value"],
                    "disabled": option["value"] not in selected,
                }
                for option in Checklist_Id_To_Options[cli]
            ]
        elif len(des_selected) == 1:
            if cli == Checklist_Dim_Id:
                new_options = [
                    {
                        "label": option["label"],
                        "value": option["value"],
                        "disabled": option["value"] not in selected,
                    }
                    for option in Checklist_Id_To_Options[cli]
                ]
            else:
                new_options = Checklist_Id_To_Options[cli]
        else:
            new_options = Checklist_Id_To_Options[cli]
        output_options.append(new_options)
    return output_options


# endregion

# region btnpop_z


def get_btnpop_z():
    mo_checklist_div = html.Div(
        dbc.Checklist(
            id=MO_Cheklist_Id,
            className="btn-group flex-wrap",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary mb-2",
            labelCheckedClassName="active",
            options=_mo_options,
            value=[_mo_options[0]['value'], ]
        ),
        className="radio-group",
    )

    rp_radio_div = html.Div(
        dbc.RadioItems(
            id=RP_Radio_Id,
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
        name="btnpop_z",
        btn_label="Z value",
        popover_header="Select a model and an acquisition function (ranking parameter)",
        popover_content=content
    )
    return btnpop_z


BtnPop_Z = get_btnpop_z()


# only one or two models can be selected
@app.callback(
    Output(MO_Cheklist_Id, "options"),
    Input(MO_Cheklist_Id, "value"),
)
def mo_options_disable(selected_mos):
    new_opts = _mo_options
    if len(selected_mos) == 1:
        new_opts = [
            {
                'label': opt['label'],
                'value': opt['value'],
                'disable': opt['value'] in selected_mos
            }
            for opt in _mo_options
        ]
    elif len(selected_mos) == 2:
        new_opts = [
            {
                'label': opt['label'],
                'value': opt['value'],
                'disable': opt['value'] not in selected_mos
            }
            for opt in _mo_options
        ]
    return new_opts


# endregion

# region btnpop_mg
def get_btnpop_mg():
    mg_options = _mg_options
    mg_checklist_div = html.Div(
        dbc.Checklist(
            id=MG_Checklist_Id,
            className="btn-group flex-wrap",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary mb-2",
            labelCheckedClassName="active",
            options=mg_options,
            value=[mg_options[0]["value"], ]
        ),
        className="radio-group",
    )

    btnpop_mg = BtnPop.from_name(
        name="btnpop_mg",
        btn_label="Ligand group",
        popover_header="Select a set of ligand groups",
        popover_content=mg_checklist_div
    )
    return btnpop_mg


BtnPop_MG = get_btnpop_mg()

# endregion

# region btnpop_color_and_size
mcolor_radio = dbc.RadioItems(
    id=MCOLOR_Radio_Id,
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
    id=MSIZEUNI_Radio_Id,
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

# endregion

# region btnpop_sliders
mzpercent_slider = dcc.RangeSlider(0, 100, id=MZPERCENT_Slider_Id, value=[0, 100],
                                   tooltip={"placement": "bottom", "always_visible": True})
msizemulti_slider = dcc.Slider(0, 2, step=0.05, value=1, id=MSIZEMULTI_Slider_Id, marks=None)

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

# endregion

# region btnpop callbacks
btnpop_components = [BtnPop_XY, BtnPop_Z, BtnPop_MG, btnpop_color_and_size, btnpop_sliders]
bp_inputs = []
bp_outputs = []
for bp in btnpop_components:
    bp_inputs.append(Input(bp.button.id, "n_clicks"))
    bp_inputs.append(State(bp.popover.id, "is_open"))
    bp_outputs.append(Output(bp.popover.id, "is_open"))


@app.callback(bp_outputs, bp_inputs)
def bp_open(*args):
    nbps = int(len(args) / 2)
    outputs = [False, ] * nbps
    for i in range(nbps):
        bp = btnpop_components[i]
        if bp.button.id == ctx.triggered_id:
            outputs[i] = not args[i * 2 + 1]
            return outputs
    return outputs


# endregion


@app.callback(
    Output(store_scatter_selection.id, "data"),
    Input(MO_Cheklist_Id, "value"),
    Input(RP_Radio_Id, "value"),
    Input(MG_Checklist_Id, "value"),
    Input(store_scatter_selection_xy.id, "data")
)
def update_selection(mo, rp, mg, xy):
    mg_vals = mg
    if "all" in mg:
        mg_labs = ["all"]
    else:
        mg_labs = [_value_to_label[v] for v in mg]

    xy_des = xy['des']
    xy_dim = xy['dim']
    if len(xy_des) == 2:
        x_val, y_val = xy_des
        x_lab = _value_to_label[x_val]
        y_lab = _value_to_label[y_val]
    elif len(xy_dim) == 1:
        x_val = xy_dim[0] + "_x"
        y_val = xy_dim[0] + "_y"
        x_lab = _value_to_label[xy_dim[0]] + "_x"
        y_lab = _value_to_label[xy_dim[0]] + "_y"
    else:
        return None

    if len(mo) == 1:
        mo = mo[0]
        z_val = "__".join([rp, mo])
        z_lab = _value_to_label[rp].replace("pred_", "") + " @ " + _value_to_label[mo].replace("MODEL=", "")
        z_val1 = None
        z_val2 = None
        delta_z_mode = False
    elif len(mo) == 2:
        mo1, mo2 = mo
        z_val1 = "__".join([rp, mo1])
        z_val2 = "__".join([rp, mo2])
        mo1_lab = _value_to_label[mo1].replace("MODEL=", "")
        mo2_lab = _value_to_label[mo2].replace("MODEL=", "")
        z_lab = _value_to_label[rp].replace("pred_", "")
        z_lab = f"\u0394 {z_lab}({mo1_lab}, {mo2_lab})"
        z_val = None
        delta_z_mode = True
    else:
        return None

    data = dict(
        x_val=x_val,
        y_val=y_val,
        z_val=z_val,
        x_lab=x_lab,
        y_lab=y_lab,
        z_lab=z_lab,
        z_val1=z_val1,
        z_val2=z_val2,
        mg_vals=mg_vals,
        mg_labs=mg_labs,
        delta_z_mode=delta_z_mode,
    )
    return data


@app.callback(
    Output("LigExp__scatter", "figure"),
    Output("LigExp__hist", "figure"),
    Input(store_scatter_selection.id, "data"),
    Input(MCOLOR_Radio_Id, "value"),
    Input(MSIZEUNI_Radio_Id, "value"),
    Input(MZPERCENT_Slider_Id, "value"),
    Input(MSIZEMULTI_Slider_Id, "value"),
)
def update_scatter(data, color_select, size_uni, z_percent_range, size_multi):
    if data is None:
        return no_update

    x_val = data['x_val']
    y_val = data['y_val']
    z_val = data['z_val']
    x_lab = data['x_lab']
    y_lab = data['y_lab']
    z_lab = data['z_lab']
    z_val1 = data['z_val1']
    z_val2 = data['z_val2']
    mg_vals = data['mg_vals']
    delta_z_mode = data['delta_z_mode']
    mg_vals_no_all = [v for v in mg_vals if v != "all"]

    if delta_z_mode:
        z_val = z_lab
        df_z = SharedData["df_rkp"][[z_val1, z_val2]].copy()
        df_z[z_val] = df_z[z_val1] - df_z[z_val2]
    else:
        df_z = SharedData["df_rkp"][[z_val, ]]

    try:
        df_xy = SharedData["df_descriptor"][[x_val, y_val]]
    except KeyError:
        df_xy = SharedData["df_dimred"][[x_val, y_val]]

    df_ligand = SharedData['df_ligand']
    df_meta = LigexpData['df_meta']
    df_meta['mg_all'] = [True, ] * len(df_meta)

    df_mgs = df_meta[mg_vals]

    df_scatter = pd.concat([df_ligand, df_xy, df_z, df_mgs], axis=1)
    bool_idx = df_mgs.any(axis='columns')
    df_scatter = df_scatter[bool_idx]

    z = df_scatter[z_val].values
    minz = np.percentile(z, z_percent_range[0])
    maxz = np.percentile(z, z_percent_range[1])

    zlim_min = min(z)
    zlim_max = max(z)

    percentile_mask = (df_scatter[z_val] >= minz) & (df_scatter[z_val] <= maxz)
    df_scatter_filtered = df_scatter[percentile_mask]

    colorscale_name = 'RdBu' if delta_z_mode else 'Blues'

    def marker_dict(
            z_array, cbar=True, colorz=True, unisize=size_uni, size_multiplier=size_multi, deltaz=delta_z_mode,
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

    if delta_z_mode:
        zvalcols = [z_val, z_val1, z_val2]
    else:
        zvalcols = [z_val, ]
    traces = []
    if color_select == "z":
        df = df_scatter_filtered
        customdata = df[df_ligand.columns.tolist() + zvalcols]
        trace = go.Scattergl(
            x=df[x_val], y=df[y_val], mode="markers",
            marker=marker_dict(df[z_val], cbar=True, colorz=True, unisize=size_uni),
            customdata=customdata, name="Selected ligands",
            showlegend=True,
        )
        traces.append(trace)

    elif color_select == "mg":
        for mg_val in mg_vals:
            bool_idx_mg = df_scatter_filtered[[mg_val]].all(axis='columns')
            df = df_scatter_filtered[bool_idx_mg]
            customdata = df[df_ligand.columns.tolist() + zvalcols]
            if len(df) == 0:
                continue
            trace = go.Scattergl(
                x=df[x_val], y=df[y_val], mode="markers",
                marker=marker_dict(df[z_val], cbar=False, colorz=False, unisize=size_uni),
                customdata=customdata, name=_value_to_label[mg_val],
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
        legend_title_text=f"{len(df_scatter_filtered)} ligands from {len(mg_vals)} group(s)",
        xaxis=dict(title=x_lab),
        yaxis=dict(title=y_lab),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
    )

    fig.update_layout(
        # autosize=False,
        # width=500,
        # height=550,
        # plot_bgcolor='rgba(255,255,255,0.1)'
    )

    counts, bins = np.histogram(df_scatter_filtered[z_val], bins=min(len(df_scatter_filtered[z_val]) // 5 + 1, 20))
    bins = 0.5 * (bins[:-1] + bins[1:])
    hist_traces = [go.Bar(x=bins, y=counts, marker={'color': bins, 'colorscale': colorscale_name})]

    hist_fig = go.Figure(data=hist_traces)

    hist_fig.update_layout(
        # dragmode='select',
        xaxis=dict(title=z_lab),
        yaxis=dict(title="Count"),
        margin=dict(l=10, r=10, b=10, t=10, pad=2),
        # width=500,
        # height=250,
    )

    return fig, hist_fig


@app.callback(
    Output("LigExp__tooltip", "show"),
    Output("LigExp__tooltip", "bbox"),
    Output("LigExp__tooltip", "children"),
    Input("LigExp__scatter", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    pt = hoverData["points"][0]
    bbox = pt["bbox"]

    customdata = pt['customdata']
    delta_z_mode = False
    try:
        lab, smi, inchi, cas, original_z = customdata
        oz1 = None
        oz2 = None
    except ValueError:
        lab, smi, inchi, cas, original_z, oz1, oz2 = customdata
        delta_z_mode = True

    df_pred = pd.read_parquet(f"{FOLDER_ASSETS_PRED}/AL1026/pred_{lab}.parquet")

    tooltip_figure = go.Figure(
        data=[
            go.Scatter(
                x=df_pred["x"],
                y=df_pred["y"],
                error_y=dict(
                    array=df_pred["yerr"].values
                ),
                mode="markers",
            )
        ]
    )
    tooltip_figure.update_xaxes(type="log")

    x_show = "{:.2f}".format(pt['x'])
    y_show = "{:.2f}".format(pt['y'])
    xy = f"xy = ({x_show}, {y_show})"
    if delta_z_mode:
        z_show = "{:.2f} = {:.2f} - {:.2f}".format(original_z, oz1, oz2)
        z = html.Span(f"dz = {z_show}", style={"color": "red"})
    else:
        z_show = "{:.2f}".format(original_z)
        z = html.Span(f"z = {z_show}", style={"color": "red"})

    children = html.Div([
        html.H6(lab, className="text-center mb-0 pb-0", style={"color": "darkblue", "overflow-wrap": "break-word"}),
        html.Img(src=f"assets/svg/{lab}.svg", style={'height': '150px', 'width': '150px'},
                 className="mt-0 mb-0 rounded mx-auto d-block pb-0 pt-0"),
        html.P([f"{smi}", html.Br(), xy, html.Br(), z], style={"overflow-wrap": "break-word"}),
    ], style={'width': '200px', 'white-space': 'normal'}),
    return True, bbox, children


@app.callback(
    Output("LigExp__rows", "children"),
    Input("LigExp__scatter", "selectedData"),
)
def update_rows(selectedData):
    if selectedData is None:
        return None
    pts = selectedData['points']
    if len(pts) > 10:
        rows = [
            dbc.Badge(f"{len(pts)} ligands selected, only display first 10", color="warning", className="mb-3")
        ]
        pts = pts[:10]
    else:
        rows = []
    for pt in pts:
        customdata = pt['customdata']
        delta_z_mode = False
        try:
            lab, smi, inchi, cas, original_z = customdata
            oz1 = None
            oz2 = None
        except ValueError:
            lab, smi, inchi, cas, original_z, oz1, oz2 = customdata
            delta_z_mode = True

        traces = []
        icolor = 0
        for i, model_path in enumerate(get_pred_available_model_paths()):
            pred_parquet_path = model_path + "/" + f"pred_{lab}.parquet"
            df_pred = pd.read_parquet(pred_parquet_path)
            model_name = os.path.basename(os.path.dirname(pred_parquet_path))
            x = df_pred['x']
            y = df_pred['y']
            yerr = df_pred['yerr']
            scatter_color = DEFAULT_PLOTLY_COLORS[i]
            fill_color = rgb_to_rgba(scatter_color, 0.2)
            scatter = go.Scatter(
                x=x, y=y, mode="lines", name=model_name,
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
            icolor += 1

        expt_parquet_path = get_expt_path(lab)
        if file_exists(expt_parquet_path):
            df_expt = pd.read_parquet(expt_parquet_path)
            df_expt = df_expt[df_expt["ReactionType"] == "real"]
            scatter_expt = go.Scatter(
                x=df_expt['LigandAmount'], y=df_expt['FigureOfMerit'], mode="markers", name="EXPT.",
                marker=dict(color=DEFAULT_PLOTLY_COLORS[icolor + 1]),
            )
            traces += [scatter_expt, ]

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
        img = html.Img(
            src=f"assets/svg/{lab}.svg",
            className="img-fluid",
        ),

        content = dbc.Row(
            [
                html.Div(img, className="col-3"),
                html.Div(
                    [
                        html.B(f"SMILES:"),
                        html.Br(),
                        smi,
                        html.Br(),
                        html.B(f"CAS:"),
                        html.Br(),
                        cas,
                    ],
                    className="col-3",
                ),
                html.Div(
                    dcc.Graph(
                        figure=fig_pred,
                        style={"width": "100%", "height": "100%"},
                    ),
                    className="col-6",
                )
            ]
        )

        row = dbc.Card(
            [
                dbc.CardHeader(lab),
                dbc.CardBody(content),
            ],
            id=f"cardinrow-{lab}",
            className="mb-3",
            style={"max-height": "300px"},
        )
        rows.append(row)
    return rows


# region layout
layout = html.Div(
    children=[
        html.H2("Ligand Explorer"),
        html.Hr(),
        html.Div(
            [
                BtnPop_XY.get_div(),
                BtnPop_Z.get_div(),
                BtnPop_MG.get_div(),
                btnpop_color_and_size.get_div(),
                btnpop_sliders.get_div(),
            ],
            # className="text-center",
        ),
        store_scatter_selection,
        store_scatter_selection_xy,

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="LigExp__scatter", clear_on_unhover=True, ),
                    className="col-6",
                ),
                dbc.Col(
                    dcc.Graph(id="LigExp__hist", style={"height": "100%"}),
                    className="col-6",
                )
            ]
        ),

        html.Div(id="LigExp__rows"),

        dcc.Tooltip(id="LigExp__tooltip", direction="right", background_color="rgba(255,255,255,1.0)")
    ]
)
# endregion
