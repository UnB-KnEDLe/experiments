Nesta pasta constam os arquivos para geração de modelos `crf` para detecção de entidades nomeadas a partir de atos extraídos e anotados ao longo do projeto KnEDLe.  

Os arquivos são, respectivamente:

####. `ner_CV.ipynb`

Arquivo principal. Nele são treinados o modelos `crf` e registrados usando o **mlflow**.
Para correto uso do notebook, os passos a serem seguidos são:

```sh
    ./start-server.sh &
    # Note que `Ato_Cessao pode ser alterado para outro conforme requerido.
    # E a hash é também parametrizável; está é apenas a que está no GitHub
    ./start-model-serving.sh 5fdbcc9794c0461baf2518cd13ac3a5d Ato_Cessao &
    jupyter nbconvert --to script  ner_CV.ipynb 
    python ner_CV.py
```

Para customização das configurações utilizadas, basta alterar os arquivos `start-server.sh` e `start-model-serving.sh`

#### \*.sh

Arquivos de *bash script* para iniciar o servidor do **mlflow** bem como o registro de modelos. Além disso, o arquivo `inference_run.sh` pode ser executado após da seguinte forma para demonstração de inferência:

```sh
    ./start-server.sh &
    # Note que `Ato_Cessao pode ser alterado para outro conforme requerido.
    # E a hash é também parametrizável; está é apenas a que está no GitHub
    ./start-model-serving.sh 5fdbcc9794c0461baf2518cd13ac3a5d Ato_Cessao &
    ./inference_run.sh
```

##### NOTA

1. O primeiro script a ser executado é sempre o `start-server`, e este deve ser executado apenas uma vez, a menos que se deseje levantar outro servidor.
2. Caso altere o `backend`, endereço do servidor ou a porta e deseje-se treinar novos modelos, faz-se necessário alterar a seguinte linha em `ner_CV.ipynb`:

```python
mlflow.set_tracking_uri("sqlite:///localhost:5000")
```

