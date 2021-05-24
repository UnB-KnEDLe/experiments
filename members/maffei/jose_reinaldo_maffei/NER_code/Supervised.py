# -*- coding: utf-8 -*-
"""Copy of cnn_cnn_lstm_WORKING.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13oqSLZ88VvrECimE5E3a3AhCgmjzsYXt
"""

from google.colab import drive
# from pathlib import Path
drive.mount('/content/drive')

!pip install --upgrade gensim

!pip install seqeval

# Basic packages
import itertools
import joblib
import re
import math
import operator
import argparse
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
# NER open packages
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
# my NER packages
# from data3 import active_dataset
from utils import create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, find_iobes_entities, find_iobes_entities2
import metrics
import data3
import utils
from CNN_biLSTM_CRF import cnn_bilstm_crf
from CNN_CNN_LSTM2 import CNN_CNN_LSTM

def devicefy(lis, device):
  return [i.to(device) for i in lis]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH=Path('drive/MyDrive/knedle_data/')
SAVE_PATH=Path('drive/MyDrive/knedle_data/NER_results')

def build_parser(**kwargs):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_path', 
      action='store', dest='save_path', 
      default=kwargs.get('save_path', SAVE_PATH), type=str,
      # default='experiments/Supervised/', type=str
      )
  parser.add_argument(
      '--epochs', 
      action='store', 
      dest='epochs', default=kwargs.get('epochs', 20), type=int)
  parser.add_argument(
      '--dataset', 
      action='store', 
      dest='dataset', default=kwargs.get('dataset', 'ato_nomeacao_efetivo'), type=str)
      # dest='dataset', default='aposentadoria', type=str)
  parser.add_argument(
      '--model', 
      action='store', 
      dest='model', default = kwargs.get('model', 'CNN-CNN-LSTM'), type=str)
  parser.add_argument(
      '--lstm_hidden_size', 
      action='store', 
      dest='lstm_hidden_size', default=kwargs.get('lstm_hidden_size', 128),
      type=int)
  parser.add_argument(
      '--batch_size', 
      action='store', 
      dest='batch_size', default=kwargs.get('batch_size', 16), type=int)
  parser.add_argument(
      '--use_dev_set', 
      action = 'store', 
      dest='use_dev_set', default=kwargs.get('use_dev_set', False), type=bool)
  # parser_opt = parser.parse_args()
  global parser_opt
  parser_opt, unknown = parser.parse_known_args()
  print(f'Experiment:')

from importlib import reload as rl
import utils
rl(utils)
rl(data3)


def pre_model_builder():
  global parser_opt, DATA_PATH
  global emb, train_path, test_path, data_format

  emb, train_path, test_path, data_format = utils.load_embedding(parser_opt, DATA_PATH)
  print(train_path, test_path)

  global collate_object
  collate_object = utils.new_custom_collate_fn(
      pad_idx=emb.key_to_index['<PAD>'], 
      unk_idx=emb.key_to_index['<UNK>'],
  )

  print('\nGenerating text2idx dictionaries (word, char, tag)')
  global word2idx, char2idx, tag2idx

  word2idx = utils.create_word2idx_dict(emb, train_path)
  char2idx = utils.create_char2idx_dict(train_path=train_path)
  tag2idx, tag_not_added  = utils.create_tag2idx_dict(train_path=train_path)

# model_opt = parser_opt.model
                     
# CNN-CNN-LSTM (greedy decoding) model

def model_builder():

  global parser_opt
  global char2idx, emb, word2idx, tag2idx, DEVICE
  global lrate, momentum, clipping_value, flag_adjust_lrate
  global model

  if parser_opt.model == 'CNN-CNN-LSTM':
      model = CNN_CNN_LSTM(char_vocab_size=len(char2idx),
                                  char_embedding_dim=25,
                                  char_out_channels=50,
                                  pretrained_word_emb=emb,
                                  word2idx = word2idx,
                                  word_out_channels=400,
                                  word_conv_layers = 1,
                                  num_classes=len(tag2idx),
                                  decoder_layers = 1,
                                  decoder_hidden_size = 128,
                                  device=DEVICE)
      lrate = 0.010
      momentum = 0.9
      clipping_value = 5.0
      flag_adjust_lrate = False
      
  elif model_opt == 'CNN-biLSTM-CRF':
      model = cnn_bilstm_crf(char_vocab_size=len(char2idx), 
                    char_embedding_dim=30, 
                    char_out_channels=30, 
                    pretrained_word_emb=emb, 
                    num_classes=len(tag2idx), 
                    device=DEVICE, 
                    lstm_hidden_size=parser_opt.lstm_hidden_size)
      lrate = 0.0025
      momentum = 0.9
      clipping_value = 5.0
      flag_adjust_lrate = False
  model.to(DEVICE)

def datasets_builder():
  print('\nCreating training dataset')
  global train_path, word2idx, char2idx, tag2idx, data_format
  global train_set

  train_set = data3.active_dataset(
      data=train_path, 
      word2idx_dic=word2idx, 
      char2idx_dic=char2idx, 
      tag2idx_dic=tag2idx, 
      data_format=data_format)
  # Putting all sentences into the labeled set for training
  train_set.flag_labeled = False
  train_set.label_data( [*range(len(train_set))] )
  train_set.flag_labeled = True

  print('\nCreating test dataset')
  global test_set
  test_set  = data3.active_dataset(
      data=test_path, 
      word2idx_dic=word2idx, 
      char2idx_dic=char2idx, 
      tag2idx_dic=tag2idx, 
      data_format=data_format)
  # Putting all sentences into the labeled set for testing
  test_set.flag_labeled = False
  test_set.label_data([*range(len(test_set))])
  test_set.flag_labeled = True
  global test_dataloader
  test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_object)


def post_model_builder():
  global parser_opt, model, lrate, momentum
  supervised_epochs = parser_opt.epochs
  global optim
  optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum)

  train_set.flag_labeled = True
  batch_size = parser_opt.batch_size
  # batch_size = 64
  global train_dataloader
  train_dataloader = DataLoader(
      train_set, 
      batch_size=parser_opt.batch_size, 
      pin_memory=True, collate_fn = collate_object, shuffle=False)

ACTS = [
  # 'aposentadoria',
  # 'ato_nomeacao_efetivo',
  'aposentadoria',
  'ato_exoneracao_efetivo',
  'ato_cessao',
  'ato_exoneracao_comissionado',
  'ato_nomeacao_comissionado',
  'ato_retificacao_comissionado',
  'ato_retificacao_efetivo',
  'ato_reversao',
  'ato_substituicao',
  'ato_tornado_sem_efeito_apo',
  'ato_tornado_sem_efeito_exo_nom',
  'ato_abono_permanencia',
]

for act in ACTS:
  build_parser()
  if act == 'aposentadoria':
    parser_opt.batch_size = 64
  
  parser_opt.dataset = act
  pre_model_builder()
  model_builder()
  datasets_builder()
  post_model_builder()

  f1_history = []
  print('ACT:', act)
  print(f'\nInitiating supervised training\n\n')
  for epoch in range(20):
# for epoch in range(supervised_epochs):
    print(f'\tEpoch: {epoch}')
    model.train()        
    # for sent, tag, word, mask in dataloader:
    for tup in train_dataloader:
      sent, tag, word, mask = devicefy(tup, DEVICE)
      optim.zero_grad()
      loss = model(sent, word, tag, mask)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
      optim.step()
    
    # Verify performance on test set after supervised training
    model.eval()
    with torch.no_grad():
      predictions, targets = metrics.preprocess_pred_targ(model, test_dataloader, DEVICE)
      predictions = metrics.IOBES_tags(predictions, tag2idx)
      targets = metrics.IOBES_tags(targets, tag2idx)
      micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
      f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
      print(f'\tmicro f1-score: {micro_f1}\n')

hyperparams = {'model': str(model), 'LR': lrate, 'momentum': momentum, 'clipping': clipping_value}
dic = {'f1_hist': f1_history, 'hyperparams': hyperparams}
path = parser_opt.save_path.as_posix()

from glob import glob
cnt = 1
for name in glob(path+'/*.pkl'):
    if model_opt in name:
        cnt += 1
f_name = model_opt + '_' + str(cnt) + '.pkl'

joblib.dump(dic, path + f_name)
print(f'Training saved in: {path + f_name}')