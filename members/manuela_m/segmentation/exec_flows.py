import os

n_jobs = 10

tipos_atos = [
    "extrato_de_aditamento_contratual",
    "extrato_de_contrato_ou_convenio",
    "aviso_de_licitacao",
    "aviso_de_suspensao_de_licitacao",
    "aviso_de_anulacao_e_revogacao",
]


def exec_pre_process_flow(act):
    os.system(
        f"python3 pre_process_flow.py run --act {act} --labeled"
        f" ../Raw_datasets/atos_{act}.parquet --dodf_dir ../dodf_txt --output"
        f" ../Processed_datasets/ --max-workers {n_jobs}"
    )


def exec_segmentation_flow(act):
    os.system(
        f"python3 segmentation_flow.py run "
        f"--act {act} "
        f"--embedding ../Models/cbow_s50_2.txt "
        f"--dataset ../Processed_datasets/segmentacao_{act}.iob "
        f"--output ../Models/ --max-workers {n_jobs}"
    )


for act in tipos_atos:
    exec_pre_process_flow(act)
    exec_segmentation_flow(act)
