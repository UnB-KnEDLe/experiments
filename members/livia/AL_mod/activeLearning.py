import pandas as  pd
import matplotlib.pyplot as plt
from sklearn import metrics

def benchmarkBvSB(X_train, y_train, X_unlabeled, clf, categories, NUM_QUESTIONS):
    '''Processamento do conjunto de treinamento e escolha dos exemplos a serem rotulados utilizando o metodo Best vs Second Best'''

    clf.fit(X_train, y_train)

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
    df.columns = ['labels', 'conf']
    df.sort_values(by=['conf'], ascending=True, inplace=True)
    question_samples = []

    for category in categories:
        low_confidence_samples = df[df.labels == category].conf.index[0:NUM_QUESTIONS]
        question_samples.extend(low_confidence_samples.tolist())
        df.drop(index=df[df.labels == category][0:NUM_QUESTIONS].index, inplace=True)

    return question_samples


def clfTest(X_test, y_test, clf):
    '''Faz as classificacoes e mostra a f1-score resultante'''

    pred = clf.predict(X_test)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    return metrics.f1_score(y_test, pred, average='micro')



def activeLearning(classifier, vectorizer, df_train, df_test, column_label='labels', column_conteudo='conteudo', NUM_QUESTIONS=5):

    MAX_SIZE = df_train[column_label].size
    categories = df_train[column_label].unique()
    result_x = []
    result_y = []

    """Cria o dataframe com os exemplos rotulados:

    *    Seleciona um exemplo para cada label
    """

    df_labeled = pd.DataFrame()

    for category in categories:
        df_labeled = df_labeled.append( df_train[df_train[column_label]==category][0:3], ignore_index=True )
        df_train.drop(index = df_train[df_train[column_label]==category][0:3].index, inplace=True)

    df_unlabeled = df_train

    df_train = df_labeled


    # Active learning : loop

    while True:

        y_train = df_train[column_label]
        y_test = df_test[column_label]

        df_unlabeled = df_unlabeled.reset_index(drop=True)

        X_train = vectorizer.fit_transform(df_train[column_conteudo])
        X_test = vectorizer.transform(df_test[column_conteudo])
        X_unlabeled = vectorizer.transform(df_unlabeled[column_conteudo])

        df_unified = df_train.append(df_unlabeled)
        X_unified  = vectorizer.transform(df_unified[column_conteudo])

        question_samples = benchmarkBvSB(X_train, y_train, X_unlabeled, classifier, categories, NUM_QUESTIONS)
        result_x.append(clfTest(X_test, y_test, classifier))
        result_y.append(df_train[column_label].size)

        print('Labeled examples: ', df_train[column_label].size)

        if (df_train[column_label].size < MAX_SIZE - (NUM_QUESTIONS + 1)) and ((len(result_x) < 2) or ( (result_x[-1] - result_x[-2] > 0) or (result_x[-1] < result_x[-2]) )):
            insert = {'labels':[], 'conteudo':[]}
            cont = 0
            for i in question_samples:
                try:
                    insert["labels"].insert(cont, df_unlabeled[column_label][i])
                    insert["conteudo"].insert(cont, df_unlabeled[column_conteudo][i])
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
