from dash import register_page, dcc

from allis_dash import FOLDER_APP, html

register_page(__name__, path='/', description="Home")

with open(f"{FOLDER_APP}/readme.md", "r") as f:
    readme = f.read()

layout = html.Div(
    [
        html.H2("Active Learning Ligand Selection"),
        html.Hr(),
        dcc.Markdown(readme),
    ]
)
