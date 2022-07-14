import pandas as pd

tipos_atos = [
    "extrato_de_aditamento_contratual",
    "extrato_de_contrato_ou_convenio",
    "aviso_de_licitacao",
    "aviso_de_suspensao_de_licitacao",
    "aviso_de_anulacao_e_revogacao",
]

acts_df = pd.read_parquet(f"../Raw_datasets/reviewed_acts.parquet")
acts_df = acts_df[["dodf_id", "act_type", "raw_text"]]
acts_df["act_type"] = acts_df["act_type"].replace(
    "aviso_de_aditamento_contratual", "extrato_de_aditamento_contratual"
)
acts_df["act_type"] = acts_df["act_type"].replace(
    "aviso_de_revogacao_anulacao_de_licitacao", "aviso_de_anulacao_e_revogacao"
)
acts_df["dodf_id"] = acts_df["dodf_id"].str.replace("_", " ")


for act in set(acts_df["act_type"]):
    filtered = acts_df.loc[acts_df["act_type"] == act]
    filtered[["dodf_id", "raw_text"]].to_parquet(f"atos_{act}.parquet")
