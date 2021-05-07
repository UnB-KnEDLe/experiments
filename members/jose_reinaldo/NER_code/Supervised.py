# Basic packages
import argparse
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import torch
from torch.utils.data import DataLoader
import itertools
import joblib
import re
import matplotlib.pyplot as plt
import operator
import math
from random import sample
# NER open packages
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
# my NER packages
from data3 import active_dataset
from utils import create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, find_iobes_entities, find_iobes_entities2
from metrics import exact_f1_score, preprocess_pred_targ, IOBES_tags
from CNN_biLSTM_CRF import cnn_bilstm_crf
from CNN_CNN_LSTM2 import CNN_CNN_LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', action='store', dest='save_path', default='experiments/Supervised/', type=str)
parser.add_argument('--epochs', action='store', dest='epochs', default=50, type=int)
parser.add_argument('--dataset', action='store', dest='dataset', default='aposentadoria', type=str)
parser.add_argument('--model', action='store', dest='model', default = 'CNN-CNN-LSTM', type=str)
parser.add_argument('--lstm_hidden_size', action='store', dest='lstm_hidden_size', default=128, type=int)
parser.add_argument('--batch_size', action='store', dest='batch_size', default=16, type=int)
parser.add_argument('--use_dev_set', action = 'store', dest='use_dev_set', default=False, type=bool)
parser_opt = parser.parse_args()
print(f'Experiment:')


# ==============================================================================================
# ==============================================================================================
# =============================     Load embeddings     ========================================
# ==============================================================================================
# ==============================================================================================
assert parser_opt.dataset in ['ontonotes', 'conll', 'aposentadoria'], 'Dataset not recognized. Options are: conll, ontonotes and aposentadoria'
if parser_opt.dataset == 'conll':
    train_path = 'conll03/eng_train.txt'
    if parser_opt.use_dev_set:
        test_path = 'conll03/eng_testa.txt'
    else:
        test_path  = 'conll03/eng_testb.txt'
    data_format = 'iob1'
    embedding_path = ''
    emb = KeyedVectors.load(embedding_path)

elif parser_opt.dataset == 'ontonotes':
    train_path = 'ontonotes/good_splits/onto.train.ner'
    if parser_opt.use_dev_set:
        test_path = 'ontonotes/good_splits/onto.development.ner'
    else:
        test_path  = 'ontonotes/good_splits/onto.test.ner'
    data_format = 'iob2'
    embedding_path = ''
    emb = KeyedVectors.load(embedding_path)

elif parser_opt.dataset == 'aposentadoria':
    train_path = 'dataset/aposentadoria_train.txt'
    if parser_opt.use_dev_set:
        test_path = 'dataset/aposentadoria_testa.txt'
    else:
        test_path  = 'dataset/aposentadoria_testb.txt'
    data_format = 'iob2'

    embedding_path = 'pt_embedding/skip_word2vec_100d.kv'
    emb = KeyedVectors.load(embedding_path)
    vocab = {}
    f = open(train_path)
    for line in f:
        try:
            word = line.split()[0]
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        except:
            pass
    found = {}
    not_found = {}
    for word in vocab:
        if word not in emb and word.lower() not in emb:
            not_found[word] = vocab[word]
        else:
            found[word] = vocab[word] 
    sorted_x = sorted(not_found.items(), key=operator.itemgetter(1))[::-1]

    # Augment pretrained embeddings with most frequent out-of-vocabulary words from the train set
    if '<START>' not in emb:
        print(f'Special token <START> not found, it\'s being added now')
        emb.add('<START>', np.random.uniform(0.1,1,100))
    if '<END>' not in emb:
        print(f'Special token <END> not found, it\'s being added now')
        emb.add('<END>', np.random.uniform(0.1,1,100))
    if '<UNK>' not in emb:
        print(f'Special token <UNK> not found, it\'s being added now')
        emb.add('<UNK>', np.random.uniform(0.1,1,100))
    if '<PAD>' not in emb:
        print(f'Special token <PAD> not found, it\'s being added now')
        emb.add('<PAD>', np.zeros(100))
    for (token, freq) in sorted_x[:37]:
        # print(f'Token {token} from training set not found, it\'s being added now')
        emb.add(token, np.random.uniform(0.1, 1, 100))

# ==============================================================================================
# ==============================================================================================
# ============================ Create train and test sets ======================================
# ==============================================================================================
# ==============================================================================================

collate_object = new_custom_collate_fn(pad_idx=emb.vocab['<PAD>'].index, unk_idx=emb.vocab['<UNK>'].index)

print('\nGenerating text2idx dictionaries (word, char, tag)')
word2idx = create_word2idx_dict(emb, train_path)
char2idx = create_char2idx_dict(train_path=train_path)
tag2idx  = create_tag2idx_dict(train_path=train_path)

print('\nCreating training dataset')
train_set = active_dataset(path=train_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=data_format)
# Putting all sentences into the labeled set for training
train_set.flag_labeled = False
train_set.label_data([i for i in range(len(train_set))])
train_set.flag_labeled = True

print('\nCreating test dataset')
test_set  = active_dataset(path=test_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=data_format)
# Putting all sentences into the labeled set for testing
test_set.flag_labeled = False
test_set.label_data([i for i in range(len(test_set))])
test_set.flag_labeled = True
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_object)

# ==============================================================================================
# ==============================================================================================
# ============================= Instantiate neural model =======================================
# ==============================================================================================
# ==============================================================================================

# Instantiating the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_opt = parser_opt.model
                     
# CNN-CNN-LSTM (greedy decoding) model
if model_opt == 'CNN-CNN-LSTM':
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
                                device=device)
    lrate = 0.010
    new_lrate = 0.010
    momentum = 0.9
    clipping_value = 5.0
    flag_adjust_lrate = False
    
elif model_opt == 'CNN-biLSTM-CRF':
    model = cnn_bilstm_crf(char_vocab_size=len(char2idx), 
                   char_embedding_dim=30, 
                   char_out_channels=30, 
                   pretrained_word_emb=emb, 
                   num_classes=len(tag2idx), 
                   device=device, 
                   lstm_hidden_size=parser_opt.lstm_hidden_size)
    lrate = 0.0025
    clipping_value = 5.0
    momentum = 0.9
    flag_adjust_lrate = False

model.to(device)

# ==============================================================================================
# ==============================================================================================
# =============================== Define training hyperparams ==================================
# ==============================================================================================
# ==============================================================================================

# Defining supervised training hyperparameters
supervised_epochs = parser_opt.epochs
optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum)


# ==============================================================================================
# ==============================================================================================
# ============================= Supervised learning algorithm ==================================
# ==============================================================================================
# ==============================================================================================
print(f'\nInitiating supervised training\n\n')
f1_history = []

train_set.flag_labeled = True
batch_size = parser_opt.batch_size
dataloader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, collate_fn = collate_object, shuffle=False)

# Supervised training (traditional)    
for epoch in range(supervised_epochs):
    print(f'Epoch: {epoch}')
    model.train()        
    for sent, tag, word, mask in dataloader:
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        mask = mask.to(device)
        optim.zero_grad()
        loss = model(sent, word, tag, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optim.step()
    
    # Verify performance on test set after supervised training
    model.eval()
    with torch.no_grad():
        predictions, targets = preprocess_pred_targ(model, test_dataloader, device)
        predictions = IOBES_tags(predictions, tag2idx)
        targets = IOBES_tags(targets, tag2idx)
        micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
        f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
        print(f'micro f1-score: {micro_f1}\n')

# ==============================================================================================
# ==============================================================================================
# ================================= Save training history ======================================
# ==============================================================================================
# ==============================================================================================

hyperparams = {'model': str(model), 'LR': lrate, 'momentum': momentum, 'clipping': clipping_value}
dic = {'f1_hist': f1_history, 'hyperparams': hyperparams}
path = parser_opt.save_path

from glob import glob
cnt = 1
for name in glob(path+'/*.pkl'):
    if model_opt in name:
        cnt += 1
f_name = model_opt + '_' + str(cnt) + '.pkl'

joblib.dump(dic, path + f_name)
print(f'Training saved in: {path + f_name}')
   

