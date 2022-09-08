import imp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF, metrics
from nltk.tokenize import word_tokenize
import os

class CRF_Flow():
    
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.ato = self.df.ato[0].lower()
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model = self.crf = CRF(
            algorithm = 'lbfgs',
            c1=0.17,
            c2=0.17,
            max_iterations=50,
            all_possible_transitions=True
        )
        self.result = ''
        self.run(file_path)
        
    def run(self, file_path):
        self.load()
        self.train()
        self.validation()
        self.save(file_path)


    def load(self):
        """ Carrega os arquivos IOB
        divide em treino e teste (80% treino e 20% teste)
        """
        x = []
        y = []
        
        for row in range(len(self.df)):
            texto = self.df.treated_text[row]
            iob = self.df.IOB[row].split()
            x.append(word_tokenize(texto))
            y.append(iob)

        for i in range(len(x)):
            x[i] = self.get_features(x[i])
        
        self.x_train, self.x_test,\
        self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    def train(self):
        """
        Utilizar o Modelo CRF
        """
        self.crf.fit(self.x_train, self.y_train)


    def validation(self):
        """"
        Aplicar o modelo treinado nos dados de teste
        """
        classes = self.crf.classes_
        classes.remove('O')
        y_pred = self.crf.predict(self.x_test)
        f1 = metrics.flat_f1_score(self.y_test, y_pred, average='weighted', labels=classes)
        self.result = metrics.flat_classification_report(self.y_test, y_pred, labels=classes, digits=3)
        

    def save(self, file_path):
        """
        Salvar os resultados em arquivo dos valores de F-Score por entidades.
        """
        if not os.path.exists('resultados/'):
            os.mkdir('resultados/')

        with open('resultados/' + self.ato + '.txt', 'w') as f:
            f.write(self.result)


    def get_features(self, sentence):
        """Create features for each word in act.
        Create a list of dict of words features to be used in the predictor module.
        Args:
            act (list): List of words in an act.
        Returns:
            A list with a dictionary of features for each of the words.
        """
        sent_features = []
        for i in range(len(sentence)):
            word_feat = {
                # Palavra atual
                'word': sentence[i].lower(),
                'capital_letter': sentence[i][0].isupper(),
                'all_capital': sentence[i].isupper(),
                'isdigit': sentence[i].isdigit(),
                # Uma palavra antes
                'word_before': '' if i == 0 else sentence[i-1].lower(),
                'word_before_isdigit': '' if i == 0 else sentence[i-1].isdigit(),
                'word_before_isupper': '' if i == 0 else sentence[i-1].isupper(),
                'word_before_istitle': '' if i == 0 else sentence[i-1].istitle(),

                # Uma palavra depois
                'word_after': '' if i+1 >= len(sentence) else sentence[i+1].lower(),
                'word_after_isdigit': '' if i+1 >= len(sentence) else sentence[i+1].isdigit(),
                'word_after_isupper': '' if i+1 >= len(sentence) else sentence[i+1].isupper(),
                'word_after_istitle': '' if i+1 >= len(sentence) else sentence[i+1].istitle(),

                'BOS': i == 0,
                'EOS': i == len(sentence)-1
            }
            sent_features.append(word_feat)
        return sent_features