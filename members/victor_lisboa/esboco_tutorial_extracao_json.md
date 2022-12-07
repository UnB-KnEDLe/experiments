# Tutorial de utilização da extração de entidades de um arquivo JSON

This tutorial is meant to help the process of extracting acts from the section 3 of the DODF JSON files. These acts are the of the following type:

- Contrato or Convênio
- Aditamento
- Licitação
- Anulação / Revogação
- Suspensão


### Importação da biblioteca DODFMiner

You might import the DODFMiner library in order to extract the acts from a JSON file. You can do that by doing this import:

```Python
from dodfminer.extract.polished.core import ActsExtractor
```

Each of the 5 types of acts have their own class that manages the whole process of extraction from the JSON file, but it is possible to extract all of them at once. To do that, you have to use the ActsExtractor method.

```Python
ActsExtractor.get_all_obj_sec3(file)
```

- Parameters:
    - **file** (string) - Path to JSON file.

- Returns:
    - Dictionary containing the class objects correspondent to each type of act.

Within each class object in the returned dictionary, there is a dataframe containing all the information about each act of that type found in the JSON.

For each type of act, the DataFrame information follows the pattern:

- Aditamento 

| numero_dodf | titulo | text | PROCESSO | CONTRATANTE | OBJ_ADITIVO | DATA_ESCRITO | CODIGO_SIGGO |
|-------------|--------|------|----------|-------------|-------------|--------------|--------------|

- Licitação

| numero_dodf | titulo | text | MODALIDADE_LICITACAO | OBJ_LICITACAO | DATA_ABERTURA | SISTEMA_COMPRAS | CODIGO_SISTEMA_COMPRAS | PROCESSO | VALOR_ESTIMADO | ORGAO_LICITANTE |
|-------------|--------|------|----------------------|---------------|---------------|-----------------|------------------------|----------|----------------|-----------------|

- Suspensão

| numero_dodf | titulo | text | PROCESSO | OBJ_ADITIVO |
|-------------|--------|------|----------|-------------|

- Anulação e Revogação

| numero_dodf | titulo | text | ORGAO_LICITANTE | MODALIDADE_LICITACAO | NUM_LICITACAO | IDENTIFICACAO_OCORRENCIA |
|-------------|--------|------|-----------------|----------------------|---------------|--------------------------|

- Contrato/Convênio

| numero_dodf | titulo | text | NUM_AJUSTE | PROCESSO | CONTRATADA_ou_CONVENENTE | OBJ_AJUSTE | VALOR | NATUREZA_DESPESA | FONTE_RECURSO | NOTA_EMPENHO | VIGENCIA | DATA_ASSINATURA | CONTRATANTE_ou_CONCEDENTE | PROGRAMA_TRABALHO | CNPJ_CONTRATADA_ou_CONVENENTE | CODIGO_UO | CODIGO_SIGGO | CNPJ_CONTRATANTE_ou_CONCEDENTE | NOME_RESPONSAVEL |
|-------------|--------|------|------------|----------|--------------------------|------------|-------|------------------|---------------|--------------|----------|-----------------|---------------------------|-------------------|-------------------------------|-----------|--------------|--------------------------------|------------------|