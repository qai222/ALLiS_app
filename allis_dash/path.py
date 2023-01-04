from allis_dash.utils import *

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "10rem",
    "padding": "2rem 1rem",
    "background-color": "#ccccff",
    "z-index": "1",
}

CONTENT_STYLE = {
    "margin-left": "10.5rem",
    "margin-right": "0.5rem",
    "padding": "2rem 1rem",
    "z-index": "0",
}

FOLDER_APP = os.path.abspath("{}/../".format(get_folder(__file__)))

FOLDER_PAGES = f"{FOLDER_APP}/pages"
FOLDER_ASSETS = f"{FOLDER_APP}/assets"
FOLDER_ASSETS_SVG = f"{FOLDER_ASSETS}/svg"

for folder in (FOLDER_PAGES, FOLDER_ASSETS, FOLDER_ASSETS_SVG,):
    if not os.path.isdir(folder):
        createdir(folder)


def get_svg_path(ligand_label: str, img=False) -> FilePath:
    if img:
        return f"assets/svg/{ligand_label}.svg"
    else:
        return f"{FOLDER_ASSETS_SVG}/{ligand_label}.svg"
