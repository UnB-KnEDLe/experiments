from contract_extractor_v2 import ContractExtractor
import pandas as pd

data = pd.read_parquet("df.parquet.gzip")

dodf_years = []

for year in data["year"].unique():
	dodf_years.append(data[data["year"] == year].copy().reset_index(drop=True))

for dodf_year in dodf_years:
	ContractExtractor.extract_to_file(dodf_year)