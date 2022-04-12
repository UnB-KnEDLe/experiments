import os
from multiprocessing import cpu_count

tipos_atos = [
    "extrato_de_aditamento_contratual",
    "extrato_de_contrato_ou_convenio",
    "aviso_de_licitacao",
    "aviso_de_suspensao_de_licitacao",
    "aviso_de_anulacao_e_revogacao",
]

n_jobs = 10
# n_jobs = cpu_count()

annotators = ["gabriel", "thiago", "vitor", "manuela"]
for act in tipos_atos:
    for labeled in annotators:
        os.system(
            f"python3 pre_process_flow.py run --act {act} --labeled"
            f" ../Raw_datasets/{labeled}_atos.parquet --dodf_dir ../dodf_txt --output"
            f" ../Processed_datasets/ --max-workers {n_jobs}"
        )


def call_flow(act):
    os.system(
        f"python3 segmentation_flow.py run "
        f"--act {act} "
        f"--embedding ../cbow_s50_2.txt "
        f"--dataset ../Processed_datasets/segmentacao_{act}.iob "
        f"--output ../Models/ --max-workers {n_jobs}"
    )


for i in tipos_atos:
    call_flow(i)
