import dodfSkweak
from metaflow import FlowSpec, step, Parameter


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


class DodfSkweakFlow(FlowSpec):
    path_train_data = Parameter('path_train_data',
                                help='Caminho para arquivo .txt',
                                default='/home/vitor_oliveira/projects/CorrigindoLabels/content/geral_v2_train.txt')

    @step
    def start(self):
        self.next(self.load_data_and_preprocess)

    @step
    def load_data_and_preprocess(self):
        # dependendo da sua base de dados, esse passo pode variar (para aplicacao da supervisão fraca eh necessario apenas self.dados)
        # self.dados: lista de textos(strings) correspondentes aos atos dos contratos
        train_x, train_y = preprocess(open(self.path_train_data, 'r'))

        self.dados = []

        for i in train_x:
            self.dados.append(' '.join(i))

        self.next(self.init_skweak)

    @step
    def init_skweak(self):
        self.skw = dodfSkweak.SkweakContratos(self.dados)
        self.next(self.apply_label_functions)

    @step
    def apply_label_functions(self):
        self.skw.apply_label_functions()
        self.next(self.train_hmm_dodf)

    @step
    def train_hmm_dodf(self):
        self.skw.train_HMM_Dodf()
        self.next(self.get_hmm_dataframe)

    @step
    def get_hmm_dataframe(self):
        self.df = self.skw.get_hmm_dataframe()
        self.next(self.end)

    @step
    def end(self):
        print(self.df)
        pass


if __name__ == '__main__':
    DodfSkweakFlow()
