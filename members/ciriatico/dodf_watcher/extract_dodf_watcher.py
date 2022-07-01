import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import unidecode

def parse_html_text(text):
    if text is not None:
        return unidecode.unidecode(BeautifulSoup(text, 'html.parser').get_text())
    
    return text

page = requests.get("http://164.41.76.30/dodf_watcher/")
html = page.content.decode("utf-8")

soup = BeautifulSoup(html, 'html.parser')

url_paths = soup.findAll('a')
url_paths = [l.get_text() for l in url_paths]
url_paths = [l.replace("\n", "%0A") for l in url_paths if "DODF" in l]

documentos = {"dodf": [],
              "link_jornal": [],
              "secao": [],
              "orgao": [],
              "documento": []}

secao = "Seção III"

for dodf in url_paths:
    r = requests.get("http://164.41.76.30/dodf_watcher/" + dodf)
    r_json = r.json()["json"]
    
    if secao in r_json["INFO"]:
        for orgao in r_json["INFO"][secao]:
            for doc in r_json["INFO"][secao][orgao]["documentos"]:
                documentos["dodf"].append(dodf)
                documentos["link_jornal"].append(r_json["linkJornal"])
                documentos["secao"].append(secao)
                documentos["orgao"].append(orgao)
                documentos["documento"].append(r_json["INFO"][secao][orgao]["documentos"][doc])
    
    print(dodf + " feito.")

documentos_mod = documentos.copy()
doc_keys = []

for doc in documentos_mod["documento"]:
    doc_keys += list(doc.keys())
    
doc_keys = list(set(doc_keys))

for k in doc_keys:
    documentos_mod[k] = []

for doc in documentos_mod["documento"]:
    for k in doc_keys:
        if k in doc.keys():
            documentos_mod[k].append(doc[k])
        else:
            documentos_mod[k].append(None)

documentos_mod.pop("documento")

documentos_df = pd.DataFrame(documentos_mod)
documentos_df["dodf"] = documentos_df["dodf"].str.replace(".json", "")
documentos_df = documentos_df.drop(columns=["coDemandante", "situacao", "coMateria", "layout", "nuOrdem", "regraSituacao"])

documentos_df["texto"] = documentos_df["texto"].apply(lambda text: parse_html_text(text))
documentos_df["preambulo"] = documentos_df["preambulo"].apply(lambda text: parse_html_text(text))

documentos_df.to_csv("dodf_watcher.csv", index=False)