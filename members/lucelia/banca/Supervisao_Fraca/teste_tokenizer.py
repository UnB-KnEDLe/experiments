from spacy_tokenizer import spacy_tokenizer

s = spacy_tokenizer()

df = "EXTRATO DO CONTRATO  matrícula 243. 612-4 matrícula 243.612-4. No 11/2019  SESI/DF PROCESSO: 00390-00003006/2019-92."

print(s.tokenize(df))
