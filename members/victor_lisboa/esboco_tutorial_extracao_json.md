# JSON acts extraction tutorial 

This tutorial is meant to help in the process of extracting acts from the section 3 of the DODF JSON files. These acts are the of the following type:

- Contrato / Convênio
- Aditamento
- Licitação
- Anulação / Revogação
- Suspensão

The first step to do is importing the DODFMiner `ActsExtractor` class in order to extract the acts from a JSON file:

```Python
from dodfminer.extract.polished.core import ActsExtractor
```

Each of the 5 types of acts have their own class that manages the whole process of extraction from the JSON file, but it is possible to extract all of them at once. To do that, you have to use the `get_all_obj` method.

```Python
ActsExtractor.get_all_obj(file, backend)
```

- Parameters:
    - **file** (string) - Path to JSON file.
    - **backend** (string) - Backend of act extraction (the default value is `None` and the extraction will be done using NER regardless of what is passed as argument).

- Returns:
    - Dictionary containing the class objects correspondent to each type of act.

### Returned object details

Within each class object in the returned dictionary, there's a pandas dataframe attribute (`df`) containing all the entities from that act type.

For each type of act, the dataframe entities are the following:

#### Aditamento

| numero_dodf | titulo | text | PROCESSO | CONTRATANTE | OBJ_ADITIVO | DATA_ESCRITO | CODIGO_SIGGO |
|-------------|--------|------|----------|-------------|-------------|--------------|--------------|

Example:

- **numero_dodf**: 3
- **titulo**: EXTRATO DO 3º TERMO ADITIVO AO CONTRATO Nº 52/2018
- **text**: [The entire text without segmentation]
- **PROCESSO**: 0080-014041/2016. 
- **CONTRATANTE**: SEEDF X SOLLAR ENGENHARIA LTDA 
- **OBJ_ADITIVO**: registrar o acréscimo de, aproximadamente, 0,87% ao valor inicial do Contrato nº 52/2018, que corresponde a R$ 62.652,35 (sessenta e dois mil, seiscentos e cinquenta e dois reais e trinta e cinco centavos), nos termos do inciso I da alínea b do art. 65 da Lei nº 8.666, de 21 de junho de 1993, e da Justificativa, passando o Contrato a ter o total de R$ 8.081.278,28 (oito milhões, oitenta e um mil, duzentos e setenta e oito reais e vinte e oito centavos).
- **DATA_ESCRITO**: 19/08/2021
- **CODIGO_SIGGO**: 43859

#### Licitação

| numero_dodf | titulo | text | MODALIDADE_LICITACAO | OBJ_LICITACAO | DATA_ABERTURA | SISTEMA_COMPRAS | CODIGO_SISTEMA_COMPRAS | PROCESSO | VALOR_ESTIMADO | ORGAO_LICITANTE |
|-------------|--------|------|----------------------|---------------|---------------|-----------------|------------------------|----------|----------------|-----------------|

Example:

- **numero_dodf**: 41
- **titulo**: AVISO DE LICITAÇÃO
- **text**: [The entire text without segmentation]
- **MODALIDADE_LICITACAO**: PREGÃO ELETRÔNICO
- **OBJ_LICITACAO**: aquisição de gêneros alimentícios para a Residência Oﬁcial do Lago Sul - ROLS, conforme especiﬁcações, quantitativos e condições estabelecidas no Edital e seus anexos.
- **DATA_ABERTURA**: 08/09/2021
- **SISTEMA_COMPRAS**: www.comprasgovernamentais.gov.br
- **CODIGO_SISTEMA_COMPRAS**: 974002
- **PROCESSO**: 00014-00000265/2021-09
- **VALOR_ESTIMADO**: 108.104,17
- **ORGAO_LICITANTE**: Subsecretaria de Compras Governamentais

#### Suspensão

| numero_dodf | titulo | text | PROCESSO | OBJ_ADITIVO |
|-------------|--------|------|----------|-------------|

Example:

- **numero_dodf**: 39
- **titulo**: AVISO DE SUSPENSÃO
- **text**: [The entire text without segmentation]
- **PROCESSO**: 00060-00136812/2018-14
- **OBJ_ADITIVO**: aquisição de Equipamentos Médicos e Hospitalares: ARCO CIRÚRGICO MÓVEL COM DETECTOR DIGITAL PLANO (FLAT PANEL)

#### Anulação e Revogação

| numero_dodf | titulo | text | ORGAO_LICITANTE | MODALIDADE_LICITACAO | NUM_LICITACAO | IDENTIFICACAO_OCORRENCIA |
|-------------|--------|------|-----------------|----------------------|---------------|--------------------------|

Example:

- **numero_dodf**: 25
- **titulo**: AVISO DE REVOGAÇÃO DE LICITAÇÃO
- **text**: [The entire text without segmentation]
- **ORGAO_LICITANTE**: TERRACAP
- **MODALIDADE_LICITACAO**: Concorrência
- **NUM_LICITACAO**: 05/2017
- **IDENTIFICACAO_OCORRENCIA**: anulado

#### Contrato/Convênio

| numero_dodf | titulo | text | NUM_AJUSTE | PROCESSO | CONTRATADA_ou_CONVENENTE | OBJ_AJUSTE | VALOR | NATUREZA_DESPESA | FONTE_RECURSO | NOTA_EMPENHO | VIGENCIA | DATA_ASSINATURA | CONTRATANTE_ou_CONCEDENTE | PROGRAMA_TRABALHO | CNPJ_CONTRATADA_ou_CONVENENTE | CODIGO_UO | CODIGO_SIGGO | CNPJ_CONTRATANTE_ou_CONCEDENTE | NOME_RESPONSAVEL |
|-------------|--------|------|------------|----------|--------------------------|------------|-------|------------------|---------------|--------------|----------|-----------------|---------------------------|-------------------|-------------------------------|-----------|--------------|--------------------------------|------------------|

- **numero_dodf**: 2
- **titulo**: EXTRATO DO CONTRATO PARA AQUISIÇÃO DE BENS Nº 04/2021, REGISTRO SIGGO Nº 44083/2021, NOS TERMOS PADRÃO Nº 08/2002
- **text**: [The entire text without segmentation]
- **NUM_AJUSTE**: 469/2022
- **PROCESSO**: 00146-0000000457/2021-01
- **CONTRATADA_ou_CONVENENTE**: ABV CONSTRUCOES LTDA
- **OBJ_AJUSTE**: aquisição de AREIA - tipo lavada, granulometria fina
- **VALOR**: 6,000.00
- **NATUREZA_DESPESA**: 44.90.52
- **FONTE_RECURSO**: 100000000
- **NOTA_EMPENHO**: 2021NE0/0125
- **VIGENCIA**: 12 (doze) meses
- **DATA_ASSINATURA**: 12/08/2021
- **CONTRATANTE_ou_CONCEDENTE**: ADMINISTRAÇÃO REGIONAL DO LAGO SUL
- **PROGRAMA_TRABALHO**: 154516209850810
- **CNPJ_CONTRATADA_ou_CONVENENTE**: 01.911.452/0001-98
- **CODIGO_UO**: 9118
- **CODIGO_SIGGO**: 43964
- **CNPJ_CONTRATANTE_ou_CONCEDENTE**: 16.615.705/0001-53
- **NOME_RESPONSAVEL**: RENATO BENATTI SANTOS