import json
from collections import defaultdict

import dash_bootstrap_components as dbc
from dash import dcc, html

from allis_dash.mongo_connections import COLL_CAMPAIGN, COLL_LIGAND, COLL_MODEL


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


def get_xy_components(
        descriptor_checklist_id_prefix="explorer__descriptor_checklist_id",
        dimred_checklist_id="explorer__dimred_checklist_id",
        btnpop_name="btnpop_xy",
):
    ligand_doc = COLL_LIGAND.find_one()
    category_to_descriptors = defaultdict(list)
    dimred_values = []
    for k in ligand_doc:
        if k.startswith("DESCRIPTOR@"):
            _, category, descriptor_name = k.split("@")
            category_to_descriptors[category].append(descriptor_name)
        elif k.startswith("DIMRED_"):
            _, space, dimred_param, _ = k.split("_")
            dimred_values.append("_".join([space, dimred_param]))
    dimred_values = sorted(set(dimred_values))

    xy_div_content = []

    checklists_descriptor = []
    for category, descriptor_names in category_to_descriptors.items():
        des_options = [
            {"label": descriptor_name, "value": f"DESCRIPTOR@{category}@{descriptor_name}"} for descriptor_name in
            descriptor_names
        ]
        checklist_id = descriptor_checklist_id_prefix + "-" + category
        checklist = dbc.Checklist(
            id=checklist_id,
            className="btn-group flex-wrap",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary mb-2",
            labelCheckedClassName="active",
            options=des_options,
        )
        checklists_descriptor.append(checklist)
        xy_div_content += [dcc.Markdown(f"**{category}**", className="form-check"), checklist]

    dimred_options = [{"label": v, "value": [f"DIMRED_{v}_x", f"DIMRED_{v}_y"]} for v in dimred_values]
    checklist_dimred = dbc.Checklist(
        id=dimred_checklist_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=dimred_options,
        value=[dimred_options[0]['value'], ],
    )
    xy_div_content += [dcc.Markdown(f"**dimensionality reduction**", className="form-check"), checklist_dimred]

    btnpop_xy = BtnPop.from_name(
        name=btnpop_name,
        btn_label="XY axes",
        popover_header="Select any two descriptors, or one parameter set for dimensionality reduction",
        popover_content=html.Div(xy_div_content),
    )
    return checklists_descriptor, checklist_dimred, btnpop_xy


def get_z_components(
        model_checklist_id="explorer__model_checklist",
        uscore_radio_id="explorer__uscore_radio",
        btnpop_name="btnpop_z",
        model_as_radio=False,
):
    model_docs = list(COLL_MODEL.find({}, {"_id": 1, "suggestions": 1}))
    model_options = [{"label": doc["_id"], "value": doc["_id"]} for doc in model_docs]

    uscore_options = []
    known_uscore = []
    for k in model_docs[0]["suggestions"]:
        uscore, direction = k.split(" @ ")
        if uscore not in known_uscore:
            uscore_options.append({"label": uscore, "value": uscore})
            known_uscore.append(uscore)

    uscore_options.reverse()

    if model_as_radio:
        cclass = dbc.RadioItems
        init_value = model_options[0]['value']
    else:
        cclass = dbc.Checklist
        init_value = [model_options[0]['value'], ]
    model_checklist = cclass(
        id=model_checklist_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=model_options,
        value=init_value
    )

    uscore_radio = dbc.RadioItems(
        id=uscore_radio_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=uscore_options,
        value=uscore_options[0]['value']
    )

    content = html.Div([
        dcc.Markdown(f"**Model**", className="form-check"),
        html.Div(model_checklist, className="radio-group", ),
        dcc.Markdown(f"**Utility score**", className="form-check"),
        html.Div(uscore_radio, className="radio-group", ),
    ])

    btnpop_z = BtnPop.from_name(
        name=btnpop_name,
        btn_label="Z value",
        popover_header="Select a model and an acquisition function (ranking parameter)",
        popover_content=content
    )
    return model_checklist, uscore_radio, btnpop_z


def get_mg_components(
        mg_checklist_id="explorer__mg_checklist",
        btnpop_name="btnpop_mg",
):
    # mark groups include:
    # 1. suggested by a model + uscore + directed
    # 2. used in an expt campaign
    # 3. used in fitting a model
    # 4. all
    all_ligand_ids = [d['_id'] for d in COLL_LIGAND.find({}, {"_id": 1})]
    model_docs = list(COLL_MODEL.find({}, {"suggestions": 1, "training_ligands": 1}))
    campaign_docs = list(COLL_CAMPAIGN.find({}, {"ligands": 1}))
    marker_group_options = []

    # all ligands
    label = "all"
    value = {"group_name": "all", "ligand_ids": all_ligand_ids}
    value = json.dumps(value)
    marker_group_options.append({"label": label, "value": value})

    # ligands used/suggested by models
    for doc in model_docs:
        for directed_uscore, ligand_ids in doc['suggestions'].items():
            label = f"{doc['_id']} @ {directed_uscore}"
            value = {"group_name": label, "ligand_ids": ligand_ids}
            value = json.dumps(value)
            marker_group_options.append({"label": label, "value": value})
        fit_label = f"fitting {doc['_id']}"
        fit_value = {"group_name": fit_label, "ligand_ids": doc['training_ligands']}
        fit_value = json.dumps(fit_value)
        marker_group_options.append({"label": fit_label, "value": fit_value})

    # ligands used in expt campaigns
    for doc in campaign_docs:
        label = doc["_id"]
        value = {"group_name": label, "ligand_ids": doc["ligands"]}
        value = json.dumps(value)
        marker_group_options.append({"label": label, "value": value})

    mg_checklist = dbc.Checklist(
        id=mg_checklist_id,
        className="btn-group flex-wrap",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary mb-2",
        labelCheckedClassName="active",
        options=marker_group_options,
        value=[marker_group_options[-1]["value"], ]
    )

    btnpop_mg = BtnPop.from_name(
        name=btnpop_name,
        btn_label="Ligand group",
        popover_header="Select a set of ligand groups",
        popover_content=html.Div(
            mg_checklist,
            className="radio-group",
        ),
    )
    return mg_checklist, btnpop_mg
