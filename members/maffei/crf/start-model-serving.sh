mlflow models serve \
  --no-conda \
  --model-uri "mlruns/0/$1/artifacts/model-$2/" \
  --port 1234

