'''
    AL_aposentadorias

    Active learning with Support Vector Machine
    baseado no trabalho do aluno Hichemm Khalid Medeiros
'''
import re
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import activeLearning as al


def cleanText(text):
    '''Normalização do texto retirando acentuação, caracteres especiais,
       espaços adicionais e caracteres não textuais'''

    text = str(text)
    text = text.lower()
    text = re.sub(r"ú", "u", text)
    text = re.sub(r"á", "a", text)
    text = re.sub(r"é", "e", text)
    text = re.sub(r"í", "i", text)
    text = re.sub(r"ó", "o", text)
    text = re.sub(r"u", "u", text)
    text = re.sub(r"â", "a", text)
    text = re.sub(r"ê", "e", text)
    text = re.sub(r"ô", "o", text)
    text = re.sub(r"à", "a", text)
    text = re.sub(r"ã", "a", text)
    text = re.sub(r"õ", "o", text)
    text = re.sub(r"ç", "c", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"\\s+", " ", text)
    text = text.strip(' ')
    return text




# Quantidade de requisicoes de labelss para o oraculo que serao feitas por vez
NUM_QUESTIONS = 15
ENCODING = 'utf-8'

estimator = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3, class_weight='balanced')
clf = CalibratedClassifierCV(base_estimator=estimator, cv=2)
vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)


data = pd.read_csv("DODF_Aposentadoria.csv")
# data.drop(columns='COD_EMPRESA', inplace=True)

colunas = data.columns

data = data.applymap(lambda com: cleanText(com))

delete = data.groupby('labels').count()[data.groupby('labels').count().REF_ANOMES<=2].index
data = data[data.labels.isin(delete) == False]


data = data[['conteudo', 'labels']]

categories = data.labels.unique()


# data = data.sample(n = 1000, random_state = 1)

"""Divisao do dataset entre informacoes de treinamento e teste:"""

df_test = data.sample(frac = 0.2, random_state = 1)
# df_test = df.sample(n=230, random_state = 1)

df_train = data.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

al.activeLearning(classifier=clf, vectorizer=vectorizer, df_train=df_train, df_test=df_test, column_label='labels', column_conteudo='conteudo', NUM_QUESTIONS=NUM_QUESTIONS)
