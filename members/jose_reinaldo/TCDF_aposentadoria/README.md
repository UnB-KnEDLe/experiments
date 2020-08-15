# Preprocessamento dos dados anotados de aposentadoria

## extract_text_V1.ipynb
 - Realiza a identificação dos trechos de texto do DODF (extraido do DODFminer) que contém as entidades nomeadas anotadas no arquivo .csv (Atos_Aposentadoria_validados.csv)
 - DODFs devem ter sido extraidos como pure-text pelo DODFminer
 
## Preprocess_data.ipynb (ainda em andamento)
 - Realiza a identificação das entidades nomeadas nos trechos de texto identificados pelo notebook extract_text_V1.ipynb
 - É realizada a correção manual para as diferentes classes de entidades nomeadas (nome, matricula, classe, ...)
 
 ## crf.ipynb
  - Implementa um modelo CRF utilizando os dados extraídos pelo notebook Preprocess_data.ipynb
