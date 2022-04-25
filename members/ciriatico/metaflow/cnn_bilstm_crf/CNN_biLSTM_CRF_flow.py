import os
import numpy as np
from math import sqrt
from metaflow import FlowSpec, step, Parameter
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import joblib
from CNN_biLSTM_CRF_process_data import dataset, create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, augment_pretrained_embedding
from CNN_biLSTM_CRF_architecture import CNN_biLSTM_CRF

class CNN_biLSTM_CRF_Flow(FlowSpec):
	embedding_path = Parameter('embedding_path',
								help='Embedding file path')
	dataset_path = Parameter('dataset_path',
								help='Dataset path')
	save_path = Parameter('save_path',
							help='Path to store data')
	project_name = Parameter('project_name',
								help='Name of the project',
								default='NER-CNN_CNN_LSTM')
	epoch = Parameter('epoch',
						help='Number of epochs to fit the model',
						default=30)
	test_size = Parameter('test_size',
							help='Test size',
							default=0.2)
	batch_size = Parameter('batch_size',
							help='Batch size',
							default=16)
	dataset_format = Parameter('dataset_format',
								help='Format of the dataset (e.g. iob1, iob2, iobes)',
								default='iob1')
	lrate = Parameter('lrate',
						help='Learning rate for NER mdoel training',
						default=0.0015)
	momentum = Parameter('momentum',
							help='Momentum for the SGD optimization process',
							default=0.9)
	char_embedding_dim = Parameter('char_embedding_dim',
									help='Embedding dimension for each character',
									default=30)
	char_out_channels = Parameter('char_out_channels',
									help='# of channels to be used in 1-d convolutions to form character level word embeddings',
									default=50)
	word_out_channels = Parameter('word_out_channels',
									help='# of channels to be used in 1-d convolutions to encode word-level features',
									default=800)
	word_conv_layers = Parameter('word_conv_layers',
									help='# of convolution blocks to be used to encode word-level features',
									default=2)
	decoder_layers = Parameter('decoder_layers',
								help='# of layers of the LSTM greedy decoder',
								default=1)
	decoder_hidden_size = Parameter('decoder_hidden_size',
									help='Size of the LSTM greedy decoder layer',
									default=256)
	grad_clip = Parameter('grad_clip',
							help='Value at which to clip the model gradient throughout training',
							default=5)

	@step
	def start(self):
		print("flow started")
		self.data_splits = ['train', 'test']
		self.next(self.split_data)

	@step
	def split_data(self):
		dataset = open(self.dataset_path, "r").readlines()
		dataset = "".join(dataset).split("\n\n")

		train_set, test_set = train_test_split(dataset, test_size=self.test_size, random_state=42)
		self.sets = {
			"train": "\n\n".join(train_set).splitlines(True),
			"test": "\n\n".join(test_set).splitlines(True)
		}
		self.next(self.prepare_embeddings)

	@step
	def prepare_embeddings(self):
		emb = KeyedVectors.load(self.embedding_path)
		bias = sqrt(3/emb.vector_size)

		if '<START>' not in emb:
		    emb.add_vector('<START>', np.random.uniform(-bias, bias, emb.vector_size))
		if '<END>' not in emb:
		    emb.add_vector('<END>', np.random.uniform(-bias, bias, emb.vector_size))
		if '<UNK>' not in emb:
		    emb.add_vector('<UNK>', np.random.uniform(-bias, bias, emb.vector_size))
		if '<PAD>' not in emb:
		    emb.add_vector('<PAD>', np.zeros(100))

		self.word2idx = create_word2idx_dict(emb, self.sets["train"])
		self.char2idx = create_char2idx_dict(train_set=self.sets["train"])
		self.tag2idx  = create_tag2idx_dict(train_set=self.sets["train"])
		self.collate_object = new_custom_collate_fn(pad_idx=emb.key_to_index['<PAD>'], unk_idx=emb.key_to_index['<UNK>'])
		self.emb = emb
		self.next(self.load_data, foreach='data_splits')

	@step
	def load_data(self):
		data = dataset(d_set=self.sets[self.input],
							word2idx_dic=self.word2idx,
							char2idx_dic=self.char2idx,
							tag2idx_dic=self.tag2idx,
							data_format=self.dataset_format
		)
		self.data_loader = DataLoader(data,
								batch_size=self.batch_size,
								pin_memory=True,
								collate_fn = self.collate_object,
								shuffle=False
		)
		self.set_type = self.input
		self.next(self.join)

	@step
	def join(self, inputs):
		self.data_dicts = {branch.set_type : branch.data_loader for branch in inputs}
		self.merge_artifacts(inputs, exclude=['set_type', 'data_loader'])
		self.next(self.fit)

	@step
	def fit(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = CNN_biLSTM_CRF(
			char_vocab_size=len(self.char2idx),
			char_embedding_dim=self.char_embedding_dim,
			char_out_channels=self.char_out_channels,
			pretrained_word_emb=self.emb,
			lstm_hidden_size=self.decoder_hidden_size,
			num_classes=len(self.tag2idx),
			device=device,
		)
		model = model.to(device)
		self.info = model.fit(
			epoch=self.epoch,
			train_loader=self.data_dicts["train"],
			test_loader=self.data_dicts["test"],
			tag2idx=self.tag2idx,
			lr=self.lrate,
			momentum=self.momentum,
			clipping_value=self.grad_clip
		)
		self.model = model
		self.next(self.end)

	@step
	def end(self):
		joblib.dump(self.info, os.path.join(self.save_path, "{}_info".format(self.project_name)))
		torch.save(self.model.state_dict(), os.path.join(self.save_path, "{}_model".format(self.project_name)))
		print("flow finished")

if __name__ == '__main__':
	CNN_biLSTM_CRF_Flow()