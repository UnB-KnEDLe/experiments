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


class TestDodfCRFFlow(FlowSpec):
    path_test_data = Parameter('path_test_data',
                               help='Caminho para arquivo .txt',
                               default='/home/vitor_oliveira/projects/CorrigindoLabels/content/geral_v2_testb.txt')

    path_model = Parameter('path_model',
                           help='Caminho para arquivo .pkl',
                           default='/home/vitor_oliveira/projects/CorrigindoLabels/CrfNew.pkl')

    @step
    def start(self):
        self.next(self.load_data_and_preprocess)

    @step
    def load_data_and_preprocess(self):
        # dependendo da sua base de dados, esse passo pode variar
        # self.test_x: lista de listas contendo cada palavra de cada ato de contrato (teste)
        # self.test_y: lista de listas contendo o Iob para cada palavra de cada ato de contrato (teste)
        # self.dados_test_x: lista de textos(strings) correspondentes aos atos dos contratos (tratamento de test_x)
        self.test_x, self.test_y = preprocess(open(self.path_test_data, 'r'))

        self.dados_test_x = []

        for i in self.test_x:
            self.dados_test_x.append(' '.join(i))

        self.next(self.load_model)

    @step
    def load_model(self):
        self.crf = dodfCRF.CRFContratos()
        self.crf.init_model_from_path(self.path_model)
        self.next(self.test_model)

    @step
    def test_model(self):
        txt = (self.crf._get_features(
            self.dados_test_x[i].split())for i in range(len(self.dados_test_x)))

        crfIOB = self.crf.model.predict(txt)

        labels = list(self.crf.model.classes_)
        labels.remove('O')

        f1 = metrics.flat_f1_score(
            self.test_y, crfIOB, average='weighted', labels=labels)

        print("Model Score:", f1,
              "\n     =========//=========//=========//=========     ")

        print(metrics.flat_classification_report(
            self.test_y, crfIOB, labels=labels, digits=3
        ))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TestDodfCRFFlow()
