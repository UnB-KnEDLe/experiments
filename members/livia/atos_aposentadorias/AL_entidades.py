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


def benchmarkBvSB():
    '''Processamento do conjunto de treinamento e escolha dos exemplos a serem rotulados utilizando o metodo Best vs Second Best'''

    clf.fit(X_train, y_train)

    print('Labeled examples: ', df_train.rotulo.size)
    # Para cada instância, obtém as probabilidades de pertencer a cada classe
    probabilities = clf.predict_proba(X_unlabeled)

    BvSB = []
    for list in probabilities:
        list = list.tolist()
        # Obtém a probabilidade da instância pertencer à classe mais provável
        best = list.pop(list.index(max(list)))
        # Obtém a probabilidade da instância pertencer à segunda classe mais provável
        second_best = list.pop(list.index(max(list)))
        # Calcula a diferença e adiciona à lista
        BvSB.append(best-second_best)

    df = pd.DataFrame(clf.predict(X_unlabeled))
    df = df.assign(conf = BvSB)
    df.columns = ['rotulo', 'conf']
    df.sort_values(by=['conf'], ascending=True, inplace=True)
    question_samples = []

    for category in categories:
        low_confidence_samples = df[df.rotulo == category].conf.index[0:NUM_QUESTIONS]
        question_samples.extend(low_confidence_samples.tolist())
        df.drop(index=df[df.rotulo == category][0:NUM_QUESTIONS].index, inplace=True)

    return question_samples


def clfTest():
    '''Faz as classificacoes e mostra a f1-score resultante'''

    pred = clf.predict(X_test)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    return metrics.f1_score(y_test, pred, average='micro')



estimator = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3, class_weight='balanced')
clf = CalibratedClassifierCV(base_estimator=estimator, cv=2)

# Quantidade de requisicoes de rotulos para o oraculo que serao feitas por vez
NUM_QUESTIONS = 15
ENCODING = 'utf-8'
result_x = []
result_y = []

data = pd.read_excel('DODF_Atos_de_Aposentadoria_validados_201804_201909.xlsx')
data.drop(columns='COD_EMPRESA', inplace=True)

colunas = data.columns

data = data.applymap(lambda com: cleanText(com))


# Cria um novo dataframe em que as entidades estão na coluna 'entidade' e o rótulo corresponde à sua respectiva coluna em 'data'
data_AL = pd.DataFrame({})

for coluna in colunas:
    aux = pd.DataFrame({'entidade': data[coluna].values, 'rotulo': [coluna]*data[coluna].shape[0]})
    data_AL = data_AL.append(aux, ignore_index=True)

data_AL = data_AL[data_AL.entidade == data_AL.entidade]


categories = data_AL.rotulo.unique()

# data_AL = data_AL.sample(n = 300, random_state = 1)

"""Divisao do dataset entre informacoes de treinamento e teste:"""

df_test = data_AL.sample(frac = 0.2, random_state = 1)
# df_test = df.sample(n=230, random_state = 1)

df_train = data_AL.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

MAX_SIZE = df_train.rotulo.size

"""Cria o dataframe com os exemplos rotulados:

*    Seleciona um exemplo para cada rotulo
"""

df_labeled = pd.DataFrame()

for category in categories:
    df_labeled = df_labeled.append( df_train[df_train.rotulo==category][0:3], ignore_index=True )
    df_train.drop(index = df_train[df_train.rotulo==category][0:3].index, inplace=True)

df_unlabeled = df_train

df_train = df_labeled


# Active learning : loop

while True:

    y_train = df_train.rotulo
    y_test = df_test.rotulo

    df_unlabeled = df_unlabeled.reset_index(drop=True)

    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)

    X_train = vectorizer.fit_transform(df_train.entidade)
    X_test = vectorizer.transform(df_test.entidade)
    X_unlabeled = vectorizer.transform(df_unlabeled.entidade)

    df_unified = df_train.append(df_unlabeled)
    X_unified  = vectorizer.transform(df_unified.entidade)

    question_samples = benchmarkBvSB()
    result_x.append(clfTest())
    result_y.append(df_train.rotulo.size)

    print('Labeled examples: ', df_train.rotulo.size)

    if (df_train.rotulo.size < MAX_SIZE - (NUM_QUESTIONS + 1)) and ((len(result_x) < 2) or ( (result_x[-1] - result_x[-2] > -1) or (result_x[-1] < result_x[-2]) )):
        insert = {'rotulo':[], 'entidade':[]}
        cont = 0
        for i in question_samples:
            try:
                insert["rotulo"].insert(cont, df_unlabeled.rotulo[i])
                insert["entidade"].insert(cont, df_unlabeled.entidade[i])
                cont += 1
                df_unlabeled = df_unlabeled.drop(i)
            except Exception as e:
                print("Error:", e)

        df_insert = pd.DataFrame.from_dict(insert)
        df_train = df_train.append(df_insert, ignore_index=True, sort=False)

    else:
        result_y_active = result_y
        result_x_active = result_x
        plt.plot(result_y_active, result_x_active, label='Active learning')
        #plt.plot(result_y_spv, result_x_spv,label = 'Convencional')
        plt.axis([0, MAX_SIZE, 0.3, 1.0])
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.grid(True)
        plt.xlabel('Training set size')
        plt.ylabel('f1-score')
        plt.title('Documents set')
        plt.show()

        result = pd.DataFrame(result_y)
        result = result.assign(y=result_x)
        np.savetxt('results.txt', result, fmt='%f')

        break

    # end