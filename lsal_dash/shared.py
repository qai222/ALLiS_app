from collections import Counter
from collections import defaultdict

import pandas as pd

from .path import FOLDER_ASSETS_PKL, FOLDER_ASSETS_JSON
from .utils import pkl_load, critical_step, json_load


def _get_descriptor_category():
    descriptor_explanation = """
    # polarizability
    avgpol axxpol ayypol azzpol molpol dipole SLogP
    # surface
    asa maximalprojectionarea maximalprojectionradius minimalprojectionarea minimalprojectionradius psa vdwsa volume   
    # count
    chainatomcount chainbondcount fsp3 fusedringcount rotatablebondcount acceptorcount accsitecount donorcount donsitecount mass nHeavyAtom fragCpx nC nO nN nP nS nRing
    # topological
    hararyindex balabanindex hyperwienerindex wienerindex wienerpolarity
    """
    cat = dict()
    lines = descriptor_explanation.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            k = line.strip().split()[1]
            des_names = lines[i + 1].strip().split()
            cat[k] = des_names
    return cat


def label_to_svg_column(lab: str) -> str:
    svg_path = f"assets/svg/{lab}.svg"
    return f"![structure]({svg_path})"


@critical_step
def load__shared() -> dict:
    return pkl_load(f"{FOLDER_ASSETS_PKL}/shared_data.pkl")


@critical_step
def load__molinv(shared_data):
    df_ligand = shared_data['df_ligand']
    df_ligand = df_ligand.drop(columns=["LigandIdentifier", ])
    df_ligand["LigandStructure"] = df_ligand.apply(lambda row: label_to_svg_column(row["LigandLabel"]), axis=1)
    ocs = list(df_ligand.columns)
    ncs = [c.replace("Ligand", "") for c in ocs]
    df_ligand = df_ligand.rename(columns=dict(zip(ocs, ncs)))
    return {
        "df_ligand": df_ligand,
    }


@critical_step
def load__ligexp(shared_data):
    # options for descriptors
    df_descriptor = shared_data['df_descriptor']
    des_options = []
    for c in df_descriptor.columns:
        c_label = str(c).replace("DESCRIPTOR_", "")
        c_value = str(c)
        des_options.append({"label": c_label, "value": c_value})

    # categories of descriptors
    des_category = _get_descriptor_category()

    # options for dimred
    df_dimred = shared_data['df_dimred']
    dim_spec = set()
    for c in df_dimred.columns:
        col_name = str(c)
        _, space, param, axis = col_name.split("_")
        dim_spec.add((space, param))

    dim_header = "DIMRED"
    dim_options = []
    for space, param in sorted(dim_spec):
        dim_options.append(
            {"label": f"{space}_{param}", "value": "_".join([dim_header, space, param])}
        )

    # options for scatter color
    mo_options = []
    rp_options = []
    mos = set()
    rps = set()
    df_rkp = shared_data['df_rkp']
    for c in df_rkp.columns:
        col_name = str(c)
        rp_value, mo = col_name.split("__")
        rp_label = rp_value.replace("rank_average_", "")
        if mo not in mos:
            mo_option = {"label": mo, "value": mo}
            mo_options.append(mo_option)
            mos.add(mo)
        if rp_value not in rps:
            rp_option = {"label": rp_label, "value": rp_value}
            rp_options.append(rp_option)
            rps.add(rp_value)

    # options for marker group
    mg_value_to_label = {"is_init": "InSeedDataset"}
    mg_options = []
    df_meta = SharedData['df_meta']
    df_meta: pd.DataFrame
    expt_counter = Counter(df_meta['ExptCampaign'])
    for expt_name in expt_counter:
        if expt_name is None: continue
        df_meta = df_meta.assign(**{expt_name: lambda x: x.ExptCampaign == expt_name})
        mg_value_to_label[expt_name] = expt_name
    df_meta = df_meta.drop(columns=["ExptCampaign"])
    for c in df_meta.columns:
        mg_option_value = str(c)
        size = sum(df_meta[mg_option_value].astype(int))
        try:
            mg_option_label = mg_value_to_label[mg_option_value]
        except KeyError:
            _, m_and_aq, _, direction = mg_option_value.split("__")
            m, aq = m_and_aq.split("@@")
            mg_option_label = f"{m}: {aq} @ {direction}"
        mg_options.append({"label": mg_option_label + f" ({size})", "value": mg_option_value})

    labels = []
    values = []
    for opts in [
        des_options, dim_options, mo_options, rp_options, mg_options
    ]:
        for opt in opts:
            lab = opt['label']
            val = opt['value']
            assert lab not in labels and val not in values
            labels.append(lab)
            values.append(val)

    return {
        "des_options": des_options,
        "des_category": des_category,
        "dim_options": dim_options,
        "mo_options": mo_options,
        "rp_options": rp_options,
        "mg_options": sorted(mg_options, key=lambda x: len(x['label'])),

        "label_to_value": dict(zip(labels, values)),

        "n_ligands": len(df_meta),
        "df_meta": df_meta,
    }


@critical_step
def load__expt(data):
    # find correspondence between ligands and expt campaign
    df_ligand = data['df_ligand']
    df_meta = data['df_meta']
    df = pd.concat([df_ligand, df_meta], axis=1)
    df = df.dropna(axis=0, how="any", subset=['ExptCampaign'])
    label_to_expt_campaign = dict(zip(df['LigandLabel'].tolist(), df['ExptCampaign'].tolist()))
    expt_campaign_to_label = defaultdict(list)
    for k, v in label_to_expt_campaign.items():
        expt_campaign_to_label[v].append(k)

    label_to_reaction_identifiers = json_load(f"{FOLDER_ASSETS_JSON}/dict_label_to_reaction_identifiers.json.gz")
    reaction_record_table = json_load(f"{FOLDER_ASSETS_JSON}/dict_reaction_record_table.json.gz")
    expt_data = dict(
        label_to_expt_campaign=label_to_expt_campaign,
        expt_campaign_to_label=expt_campaign_to_label,
        label_to_reaction_identifiers=label_to_reaction_identifiers,
        reaction_record_table=reaction_record_table,
    )
    return expt_data


SharedData = load__shared()
MolinvData = load__molinv(SharedData)
LigexpData = load__ligexp(SharedData)
ExptData = load__expt(SharedData)
