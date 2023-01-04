import dash
from dash import Dash

from allis_dash import *
from allis_dash.mongo_connections import MongoHasData

app = Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=MongoHasData,
    assets_folder=FOLDER_ASSETS,
)

dbc_navlinks = [
    dbc.NavLink(
        f"{page['description']}", href=page["relative_path"], active='exact'
    ) for page in dash.page_registry.values()
]

sidebar = html.Div(
    [
        # html.H2("LSAL", className="display-4"),

        dcc.Markdown(
            """
            ### ***A***ctive
            ### ***L***earning
            ### ***Li***gand
            ### ***S***election
            """
        ),
        dbc.Nav(
            dbc_navlinks,
            vertical=True,
            pills=True,
        ),
        dcc.Markdown(id="Home_load_time")
    ],
    style=SIDEBAR_STYLE,
)

if MongoHasData:
    content = html.Div(id="page-content", style=CONTENT_STYLE, children=[dash.page_container])
else:
    content = html.Div("not data found in MongoDB, did you forget to import?")

app.layout = html.Div(
    [
        sidebar,
        content,
        dcc.Markdown("dummy", style={"display": "none"}, id="dummy"),
    ]
)

server = app.server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
