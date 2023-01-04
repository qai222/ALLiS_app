import pandas as pd
from dash import dash_table, register_page, dcc, get_app, html

from allis_dash import COLL_LIGAND
from allis_dash.utils import label_to_svg_column, get_basename

app = get_app()

_page_name = get_basename(__file__)
register_page(
    __name__,
    name=get_basename(__file__),
    path=f"/{_page_name}",
    description="Library",
)

COLL_LIGAND.find({}, )


# make layout a function so it reloads after page refresh
def layout():
    projection = {
        "_id": 1,
        "smiles": 1,
        "cas_number": 1,
    }
    ligand_docs = COLL_LIGAND.find({}, projection)
    df = pd.DataFrame.from_records(ligand_docs)
    df.rename(columns={"_id": "Label", "smiles": "Smiles", "cas_number": "CAS"}, inplace=True)
    df["Structure"] = df.apply(lambda row: label_to_svg_column(row["Label"]), axis=1)

    table = dash_table.DataTable(
        id='MolInv__datatable',
        data=df.to_dict("records"),
        markdown_options={'html': True},
        columns=[
            {'id': c, 'name': c, 'presentation': 'markdown'}
            for c in (
                "Label",
                "Smiles",
                "CAS",
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

    master = html.Div(
        [
            html.H2("Molecule Library"),
            html.Hr(),
            dcc.Markdown(
                r"""
                This table contains all molecular ligands considered in this project.
                - Ligands are labelled with a string of format `<type>-XXXXXXXX`:
                    - `<type>` can be either `LIGAND` or `POOL`;
                    - `LIGAND` indicates the ligand comes from the initial (seed) dataset;
                    - `POOL` indicates the ligands comes from scraping PubChem;
                - A ligand can have multiple `Cas` number, there are delimited by semicolons;
                - String data filtering is enabled except for structures. For special characters, such as parenthesis, 
                use `\` to escape, e.g. `\(CN\)`.
                """
            ),
            table,
        ]
    )
    return master
