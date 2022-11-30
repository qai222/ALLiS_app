from dash import dash_table, register_page, dcc, get_app

from lsal_dash import *
from lsal_dash.shared import *

app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Library",
)

layout = html.Div(
    children=[
        html.H2("Molecule Library"),
        html.Hr(),
        dcc.Markdown(
            r"""
            This table contains all molecular ligands considered in this project.
            - Ligands are labelled with a string of format `<type>-XXXXXXXX` 
                - `<type>` can be either `LIGAND` or `POOL`
                - `LIGAND` indicates the ligand comes from the initial (seed) dataset
                - `POOL` indicates the ligands comes from scraping PubChem
            - A ligand can have multiple `Cas` number, there are delimited by semicolons
            - String data filtering is enabled except for structures
            """
        ),
        dash_table.DataTable(
            id='MolInv__datatable',
            data=MolinvData['df_ligand'].to_dict("records"),
            markdown_options={'html': True},
            columns=[
                {'id': c, 'name': c, 'presentation': 'markdown'}
                for c in (
                    "Label",
                    "Smiles",
                    "Cas",
                    "Structure"
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
            page_size=10,
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_filter_conditional=[
                {
                    'if': {'column_id': 'Structure'},
                    'pointer-events': 'None',
                },
            ]
        )

    ],
)
