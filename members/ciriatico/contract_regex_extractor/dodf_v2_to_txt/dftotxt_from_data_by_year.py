import pandas as pd
from dftotxt import DFtoTXT

path_extract = "/home/gabriel/Desktop/unb/knedle/testes-commit/dodfminer_testes/fabricio_dodf_txts"

data = pd.read_parquet("/home/gabriel/Desktop/unb/knedle/contract_extractor/df.parquet.gzip")

dodf_years = []

for year in data["year"].unique():
    dodf_years.append(data[data["year"] == year].copy().reset_index(drop=True))

del data

for dodf_year in dodf_years:
    DFtoTXT.extract_txts(path_extract, dodf_year)