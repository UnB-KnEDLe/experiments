"""

This module implements a metaflow flow for training and evaluating a
segmentation model using a LSTM+CRF.

To run the flow, you must pass as arguments the type of act that the model will
be trained for, the path to the trained embedding that will be used by the
model, the path to the pre-processed dataset that will be used for training,
validation, and testing, and the path to save the model. For example:

python3 segmentation_flow.py run
--act aviso_de_licitacao
--embedding cbow_s50_2.txt
--dataset Processed_dataset/segmentacao_aviso_de_licitacao.iob
--output Models/

The dataset must have been preprocessed by the PreProcessFlow flow.

"""

import unicodedata
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from data_loader import SegmentationDataset
from model_arquiteture import LstmCrf
from metaflow import FlowSpec, step, Parameter


class SegmentationFlow(FlowSpec):
    """
    A flow to train and evaluate a LSTM+CRF model for segmentation.
    """

    act = Parameter(
        "act",
        help="Type of act to model (example: 'aviso de licitação'",
        required=True,
        type=str,
    )

    embedding = Parameter(
        "embedding", help="Path to the trained embeding", required=True, type=str
    )

    dataset = Parameter("dataset", help="Path to the dataset", required=True, type=str)

    output = Parameter(
        "output", help="Path to store the output model", required=True, type=str
    )

    @step
    def start(self):
        """
        Start setp, define the data splits and calls the next step.
        """
        self.data_splits = ["train", "validation", "test"]
        self.next(self.prepare_embedings)

    @step
    def prepare_embedings(self):
        """
        Prepare the embeddings and tags dictionaries to transform the dataset in the next step
        """
        emb = KeyedVectors.load_word2vec_format(f"{self.embedding}")
        dic = {}
        for j in emb.index_to_key:
            word = (
                unicodedata.normalize("NFKD", j)
                .encode("ascii", "ignore")
                .decode("utf8")
            )
            dic[word] = emb.key_to_index[j]

        self.dic_tag = {"B": 0, "I": 1, "O": 2}
        self.idx2tag = {0: "B", 1: "I", 2: "O"}
        self.emb = emb
        self.dic = dic
        print(self.act)
        self.next(self.load_data, foreach="data_splits")

    @step
    def load_data(self):
        """
        Creates an object of the SegmentationDataset class to load and
        perform the necessary transformations on the dataset,
        such as spliting train, test and validation sets,
        truncating sentences and long blocks and padding them.
        """
        data = SegmentationDataset(
            tag2idx=self.dic_tag,
            word2idx=self.dic,
            set_type=self.input,
            tipo_ato=self.act,
            path=self.dataset,
        )
        print(self.input, np.unique(data.y, return_counts=True))
        self.data_loader = DataLoader(
            dataset=data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
        )
        self.set_type = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Merge the branches created in the previous step and store the data loaders in the dictionary
        'self.data_dicts'.
        """
        self.data_dicts = {branch.set_type: branch.data_loader for branch in inputs}
        self.merge_artifacts(inputs, exclude=["set_type", "data_loader"])
        self.next(self.fit)

    @step
    def fit(self):
        """
        Train the LSTM+CRF model.
        """
        path = self.output + "seg_model_256_" + self.act
        model = LstmCrf(
            embedding_dim=50,
            num_tags=3,
            hidden_dim=256,
            pretrained_emb=self.emb,
            idx2tag=self.idx2tag,
            tipo_ato=self.act,
            path=path,
        )
        model = model.to(model.device)
        self.valid_loss = model.fit(
            epoch=50,
            train_loader=self.data_dicts["train"],
            eval_loader=self.data_dicts["validation"],
            lr=0.001,
            weight_decay=1e-6,
        )
        self.model = model
        self.next(self.predict)

    @step
    def predict(self):
        """
        Evaluates the LSTM+CRF model in the test set
        """
        path = self.output + "seg_model_256_" + self.act
        self.model.load_state_dict(torch.load(path))
        results = self.model.evaluate(self.data_dicts["test"], opt="f1")
        print(results)
        np.save(f"{self.output}/metrics_{self.act}", results)
        self.next(self.end)

    @step
    def end(self):
        """
        Final step, save the trained model.
        """
        path = self.output + "seg_model_256_" + self.act
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    SegmentationFlow()
