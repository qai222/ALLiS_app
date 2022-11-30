from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lsal_dash import *

"""
copy files generated using the `visualize` worker here
"""

data = dict()
for file in glob.glob("visualize/*"):
    if file.endswith(".csv"):
        dfname = get_basename(file)
        data[dfname] = pd.read_csv(file, index_col=0)
    elif file.endswith(".json.gz") or file.endswith(".json"):
        dictname = get_basename(file)
        dictname = dictname.replace(".json", "").replace(".gz", "")
        d = json_load(file)
        if 'label_to_df_pred' in dictname:
            preds = d
            for model_name, label_to_df in tqdm(preds.items()):
                for label, df in tqdm(label_to_df.items(), desc=model_name):
                    df_path = get_prediction_path(model_name, label)
                    if not file_exists(df_path):
                        df.to_parquet(df_path, compression=None, )
        elif 'label_to_df_expt' in dictname:
            expts = d
            for label, df in tqdm(expts.items()):
                df_path = get_expt_path(label)
                if not file_exists(df_path):
                    df.to_parquet(df_path, compression=None, )
        elif 'label_to_svg' in dictname:
            svg = d
            for label, svg_text in tqdm(svg.items()):
                svg_fn = get_svg_path(label)
                if not file_exists(svg_fn):
                    with open(svg_fn, "w") as f:
                        f.write(svg_text)
        elif 'mrb_to_cfpool' in dictname:
            mrb_to_cfpool_path = defaultdict(lambda: defaultdict(dict))
            mrb_to_cfpool = d
            for m in mrb_to_cfpool:
                for r in mrb_to_cfpool[m]:
                    createdir(get_cfpool_folder(m, r))
                    for base_label in mrb_to_cfpool[m][r]:
                        df_path = get_cfpool_path(m, r, base_label)
                        df = mrb_to_cfpool[m][r][base_label]
                        if not file_exists(df_path):
                            df.to_parquet(df_path, compression=None, )
                        mrb_to_cfpool_path[m][r][base_label] = df_path
            data["mrb_to_cfpool_path"] = mrb_to_cfpool_path
    else:
        raise ValueError

if __name__ == '__main__':
    visdata_json = f"{FOLDER_ASSETS_JSON}/VisData.json"
    json_dump(data, visdata_json, gz=False)
    data = json_load(visdata_json)
    pkl_dump(data, f"{FOLDER_ASSETS_PKL}/VisData.pkl")
