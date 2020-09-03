# Preprocessamento dos dados anotados de aposentadoria

## Bases de dados NER de aposentadoria
O split de treinamento e teste das bases apresentadas foi realizado de forma cronológica de forma a evitar que o mesmo DODF apareça tanto no conjunto de treinamento quanto no de teste.
### aposentadoria_NER_Dataset.zip (formato CoNLL03)
- train_set.txt: Conjunto de dados de treinamento.
- test_set.txt: Conjunto de dados de teste.
- Split de 70-30 para treinamento e teste.

### aposentadoria_AL_NER_Dataset.zip (formato CoNLL03)
- labeled_set.txt: Conjunto de dados rotulados inicial para testes com aprendizagem ativa (1% do conjunto de treinamento da base acima)
- unlabeled_set.txt: Conjunto de dados não rotulados inicial para testes com aprendizagem ativa (99% do conjunto de treinamento da base acima)
- test_set.txt: Conjunto de dados para teste de desempenho durante o processo de aprendizagem ativa (Mesmo conjunto de teste da base acima)

## Notebooks de preprocessamento
### extract_text_V1.ipynb
 - Realiza a identificação dos trechos de texto do DODF (extraido do DODFminer) que contém as entidades nomeadas anotadas no arquivo .csv (Atos_Aposentadoria_validados.csv)
 - DODFs devem ter sido extraidos como pure-text pelo DODFminer
 
### Preprocess_data.ipynb (ainda em andamento)
 - Realiza a identificação das entidades nomeadas nos trechos de texto identificados pelo notebook extract_text_V1.ipynb
 - É realizada a correção manual para as diferentes classes de entidades nomeadas (nome, matricula, classe, ...)
 - Entidades ainda não totalmente verificadas:
   * FUND_LEGAL: Falta verificação de 99 entidades
   * EMPRESA_ATO: Falta a verificação de 2961 entidades
 
### crf.ipynb
  - Implementa um modelo CRF utilizando os dados extraídos pelo notebook Preprocess_data.ipynb
