import os
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from dodfminer.extract.polished.acts.base import Atos

class Aditamento(Atos):
    '''
    Classe para atos de aditamento
    '''

    def __init__(self, file, backend):
        super().__init__(file, backend)
        self.file = file
        self.backend = backend  # sempre vai ser NER
        self.data_frame = pd.DataFrame()
        self.model = None
        self.flow()


    def flow(self):
        self._load_model()
        #self.preprocessing()
        self.ner_extraction()
        #self.pos_processing()

    
    def _load_model(self):
        f_path = os.path.dirname(__file__)
        f_path += '/models/aditamento.pkl'
        self.model = joblib.load(f_path)
        return


    def preprocessing(self):
        pass
    

    def ner_extraction(self, act):
        tokenized_act = word_tokenize(act)
        features = self.get_features(tokenized_act)
        prediction = self.crf.predict([features])
        # carregar resultados em dataframe
        return


    def pos_processing(self):
        pass


    def _act_name(self):
        return "Extrato de Aditamento Contratual"
    

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