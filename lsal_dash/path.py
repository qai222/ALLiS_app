import glob

from lsal_dash.utils import *

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
FOLDER_ASSETS_PRED = f"{FOLDER_ASSETS}/pred"
FOLDER_ASSETS_EXPT = f"{FOLDER_ASSETS}/expt"
FOLDER_ASSETS_CSV = f"{FOLDER_ASSETS}/csv"
FOLDER_ASSETS_JSON = f"{FOLDER_ASSETS}/json"
FOLDER_ASSETS_PKL = f"{FOLDER_ASSETS}/pkl"
FOLDER_ASSETS_CFPOOL = f"{FOLDER_ASSETS}/cfpool"

for folder in (
        FOLDER_PAGES, FOLDER_ASSETS, FOLDER_ASSETS_JSON, FOLDER_ASSETS_PRED,
        FOLDER_ASSETS_CSV, FOLDER_ASSETS_SVG, FOLDER_ASSETS_EXPT, FOLDER_ASSETS_CFPOOL, FOLDER_ASSETS_PKL
):
    if not os.path.isdir(folder):
        createdir(folder)


def get_prediction_path(model_name: str, ligand_label: str) -> FilePath:
    model_dir = f"{FOLDER_ASSETS_PRED}/{model_name}"
    if not os.path.isdir(model_dir):
        createdir(model_dir)
    return f"{model_dir}/pred_{ligand_label}.parquet"


def get_expt_path(ligand_label: str) -> FilePath:
    return f"{FOLDER_ASSETS_EXPT}/expt_{ligand_label}.parquet"


def get_svg_path(ligand_label: str, img=False) -> FilePath:
    if img:
        return f"assets/svg/{ligand_label}.svg"
    else:
        return f"{FOLDER_ASSETS_SVG}/{ligand_label}.svg"


def get_cfpool_folder(m: str, r: str):
    return f"{FOLDER_ASSETS_CFPOOL}/{m}___{r}/"


def get_cfpool_path(model_name: str, rank_method: str, base_label: str):
    cf_folder = get_cfpool_folder(model_name, rank_method)
    return f"{cf_folder}/{base_label}.parquet"


def get_pred_available_model_paths():
    return sorted(glob.glob(f"{FOLDER_ASSETS_PRED}/*/"))


def get_expt_available_model_paths():
    return sorted(glob.glob(f"{FOLDER_ASSETS_EXPT}/*/"))
