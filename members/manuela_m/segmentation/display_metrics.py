import numpy as np
from tabulate import tabulate

tipos_atos = [
    "extrato_de_aditamento_contratual",
    "extrato_de_contrato_ou_convenio",
    "aviso_de_licitacao",
    "aviso_de_suspensao_de_licitacao",
    "aviso_de_anulacao_e_revogacao",
]


def show(act):
    metrics = np.load(f"../Models/metrics_{act}.npy", allow_pickle=True).tolist()
    metrics[0].insert(0, "Recall")
    metrics[1].insert(0, "Precision")
    metrics[2].insert(0, "F1")

    print(act)
    print(tabulate(metrics, headers=["B", "I", "O"]))
    print("\n")


for act in tipos_atos:
    show(act)
