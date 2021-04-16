mlflow models serve \
  --no-conda \
  --model-uri "artefatos/0/$1/artifacts/model-$2/" \
  --port 1234

