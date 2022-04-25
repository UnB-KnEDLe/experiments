import os
import torch
import joblib
from metaflow import FlowSpec, step, Parameter
from sklearn.model_selection import train_test_split
from biLSTM_CRF_process_data import preprocess, word_dict, tag_dict, numericalize, add_special_tokens
from biLSTM_CRF_architecture import biLSTM_CRF

class BiLSTM_CRF_Flow(FlowSpec):
    dataset_path = Parameter('dataset_path',
                                help='Dataset path')
    save_path = Parameter('save_path',
                            help='Path to store data')
    epoch = Parameter('epoch',
                        help='Number of epochs to fit the model',
                        default=30)
    test_size = Parameter('test_size',
                            help='Test size',
                            default=0.2)
    lrate = Parameter('lrate',
                        help='Learning rate for NER mdoel training',
                        default=0.0015)
    momentum = Parameter('momentum',
                            help='Momentum for the SGD optimization process',
                            default=0.9)
    project_name = Parameter('project_name',
    						 help='Name of the project')

    @step
    def start(self):
        print("flow started")
        self.data_splits = ["train", "test", "valid"]
        self.next(self.split_data)

    @step
    def split_data(self):
        dataset = open(self.dataset_path, "r").readlines()
        dataset = "".join(dataset).split("\n\n")

        train_set, test_set = train_test_split(dataset, test_size=self.test_size, random_state=42)
        test_set, valid_set = train_test_split(test_set, test_size=0.5, random_state=42)
        self.sets = {
            "train": "\n\n".join(train_set).splitlines(True),
            "test": "\n\n".join(test_set).splitlines(True),
            "valid": "\n\n".join(valid_set).splitlines(True)
        }
        self.next(self.prepare_embeddings)

    @step
    def prepare_embeddings(self):
        X_train, Y_train = preprocess(self.sets["train"])

        self.word2idx = word_dict(X_train)
        self.tag2idx  = tag_dict(Y_train)
        self.next(self.load_data, foreach='data_splits')

    @step
    def load_data(self):
        X_data, Y_data = preprocess(self.sets[self.input])
        X_data, Y_data = numericalize(X_data, self.word2idx, Y_data, self.tag2idx)
        self.X_data, self.Y_data = add_special_tokens(X_data, self.word2idx, Y_data, self.tag2idx)
        self.set_type = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        data_dicts_X = {"X_{}".format(branch.set_type) : branch.X_data for branch in inputs}
        data_dicts_Y = {"Y_{}".format(branch.set_type) : branch.Y_data for branch in inputs}
        self.data_dicts = {**data_dicts_X, **data_dicts_Y}
        self.merge_artifacts(inputs, exclude=['set_type', 'X_data', 'Y_data'])
        self.next(self.fit)

    @step
    def fit(self):
        model = biLSTM_CRF(word2idx=self.word2idx, tag2idx=self.tag2idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.info = model.fit(
            self.device,
            self.data_dicts["X_train"],
            self.data_dicts["Y_train"],
            self.data_dicts["X_test"],
            self.data_dicts["Y_test"],
            self.data_dicts["X_valid"],
            self.data_dicts["Y_valid"],
            self.word2idx,
            self.tag2idx,
            self.epoch,
            self.lrate,
            self.momentum
        )
        self.model = model
        self.next(self.end)

    @step
    def end(self):
        joblib.dump(self.info, os.path.join(self.save_path, "{}_info".format(self.project_name)))
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "{}_model".format(self.project_name)))
        print("flow finished")

if __name__ == '__main__':
    BiLSTM_CRF_Flow()