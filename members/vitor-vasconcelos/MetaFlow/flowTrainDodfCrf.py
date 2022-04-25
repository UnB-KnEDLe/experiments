from metaflow import get_metadata, metadata, Parameter
import dodfCRF
from sklearn_crfsuite import metrics
import joblib
from metaflow import FlowSpec, step


def preprocess(ner_set):
    sentences = []
    tags = []

    temp_sentence = []
    temp_tag = []
    for line in ner_set:
        try:
            word, _, _, tag = line.split()
            temp_sentence.append(word)
            temp_tag.append(tag)
        except:
            sentences.append(temp_sentence)
            tags.append(temp_tag)
            temp_sentence = []
            temp_tag = []

    if temp_sentence:
        sentences.append(temp_sentence)
        tags.append(temp_tag)
    return sentences, tags


class TrainDodfCRFFlow(FlowSpec):
    path_train_data = Parameter('path_train_data',
                                help='Caminho para arquivo .txt',
                                default='/home/vitor_oliveira/projects/CorrigindoLabels/content/geral_v2_train.txt')

    @step
    def start(self):
        self.next(self.load_data_and_preprocess)

    @step
    def load_data_and_preprocess(self):
        # dependendo da sua base de dados, esse passo pode variar
        # self.train_x: lista de listas contendo cada palavra de cada ato de contrato (treinamento)
        # self.train_y: lista de listas contendo o Iob para cada palavra de cada ato de contrato (treinamento)
        # self.dados_train_x: lista de textos(strings) correspondentes aos atos dos contratos (tratamento de train_x)
        self.train_x, self.train_y = preprocess(
            open(self.path_train_data, 'r'))

        self.dados_train_x = []

        for i in self.train_x:
            self.dados_train_x.append(' '.join(i))

        self.next(self.init_crf_lbfgs)

    @step
    def init_crf_lbfgs(self):
        self.crf = dodfCRF.CRFContratos()
        self.crf.init_model_lbfgs()
        self.next(self.train_model)

    @step
    def train_model(self):
        txt = (self.crf._get_features(
            self.dados_train_x[i].split())for i in range(len(self.dados_train_x)))
        lbl = self.train_y

        try:
            self.crf.model.fit(txt, lbl)
        except AttributeError:
            pass

        self.next(self.save_model)

    @step
    def save_model(self):
        joblib.dump(self.crf.model, "CrfNew.pkl")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TrainDodfCRFFlow()
