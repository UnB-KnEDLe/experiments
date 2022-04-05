import unicodedata
import numpy as np
import torch
from metaflow import FlowSpec, step
from gensim.models import KeyedVectors
from Data_loader import segmentation_dataset
from torch.utils.data import DataLoader
from LSTM_CRF_arquiteture import LSTM_CRF

tipos_atos = ['aditamento_contratual',
'extrato_de_contrato_ou_convenio',
'aviso_de_suspensao_de_licitacao',
'revogacao_anulacao',
'aviso_de_licitacao']

class SegmentationFlow(FlowSpec):
    @step
    def start(self):
        self.ato = tipos_atos[0]
        self.data_splits = ['train', 'validation', 'test']
        self.next(self.prepare_embedings)

    @step
    def prepare_embedings(self):
        emb = KeyedVectors.load_word2vec_format("../cbow_s50_2.txt")
        dic = {}
        for j in emb.index_to_key:
            num = emb.key_to_index[j]
            word = unicodedata.normalize('NFKD', j).encode('ascii', 'ignore').decode('utf8')
            dic[word] = emb.key_to_index[j]

        self.dic_tag = {'B': 0, 'I': 1, 'O': 2}
        self.idx2tag = {0: 'B', 1:'I', 2:'O'}
        self.emb = emb
        self.dic = dic
        self.next(self.load_data, foreach='data_splits')

    @step
    def load_data(self):
        print(self.ato, self.input)
        path = "../Processed_datasets/dodf_atos_segmentacao_"+self.ato+".iob"
        data = segmentation_dataset(tag2idx=self.dic_tag, 
                                    word2idx=self.dic,
                                    set_type=self.input,
                                    tipo_ato=self.ato,
                                    path=path
        )
        self.data_loader = DataLoader(dataset = data, 
                                    batch_size=32, 
                                    shuffle=False,
                                    num_workers=2,
                                    pin_memory=True
        )
        self.set_type = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        self.data_dicts = {branch.set_type : branch.data_loader for branch in inputs}
        self.merge_artifacts(inputs, exclude=['set_type', 'data_loader'])     
        print(self.ato)
        self.next(self.fit)

    @step
    def fit(self):
        path = "seg_model_256_" +self.ato
        model = LSTM_CRF(
            embedding_dim=50,
            num_tags=3,
            hidden_dim=256,
            pretrained_emb=self.emb,
            idx2tag=self.idx2tag,
            tipo_ato=self.ato,
            path=path
        )
        model = model.to(model.device)
        self.valid_loss = model.fit(
            epoch=50, 
            train_loader = self.data_dicts['train'], 
            eval_loader = self.data_dicts['validation'], 
            lr=0.001, 
            weight_decay=1e-6
        )
        self.model = model
        self.next(self.predict)

    @step
    def predict(self):
        path = "seg_model_256_"+self.ato
        self.model.load_state_dict(torch.load(path))
        results = self.model.evaluate(self.data_dicts['test'], opt='f1')
        print(results)
        np.save(f"metrics_{self.ato}", results)
        self.next(self.end)

    @step
    def end(self):
        path = "seg_model_256_"+self.ato
        torch.save(self.model.state_dict(), path)

if __name__ == '__main__':
  SegmentationFlow()