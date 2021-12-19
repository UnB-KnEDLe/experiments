import pandas as pd
import re
from pandas import DataFrame


df = pd.read_parquet("df.parquet.gzip", engine='fastparquet')

all_acts = []
for year in range(2000,2021+1):
    for num in range(1, 284):
      data = df[['text', 'file_name']].loc[(df['number'] == num) & (df['year'] == year)].sort_index(ascending=False).reset_index(drop=True)
      lista = df[['text']].loc[(df['number'] == num) & (df['year'] == year)].sort_index(ascending=False)['text']
      if len(data) > 0:
        texto = "\n".join(lista)
        texto = data['file_name'][0] + texto
        all_acts.append(texto)

data = DataFrame(all_acts,columns=['Dodfs_list'])

df_acts.to_csv('dodfs.csv')
