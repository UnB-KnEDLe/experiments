# Experimentos para segmentação de texto de atos de aposentadoria

## Notebooks
- preprocess_dodf.ipynb
  - Identifica os atos de aposentadoria (anotados em labeled.csv na coluna ['text']) nos blocos de texto extraidos pelo DODFminer.
  - Requisitos:
    - Possuir os arquivos json extraídos pelo DODFminer em formatos de blocos sem header
    
- pretrained_word2vec.ipynb
  - Modelo LSTM-LSTM-CRF em pytorch para segmentação de textos
  - Requisitos:
    - Baixar embeddings word2vec CBOW de 100 dimensões pré-treinados [aqui](http://www.nilc.icmc.usp.br/embeddings)
    
- train_word2vec.ipynb
  - Ainda incompleto (*coming soon*)
