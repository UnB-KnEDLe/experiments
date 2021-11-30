# Basic packages
import copy
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
# NER my packages
from data import active_dataset
from utils import create_char2idx_dict, create_tag2idx_dict, create_word2idx_dict, new_custom_collate_fn, budget_limit, get_confidence
from metrics import preprocess_pred_targ, IOBES_tags
from query_functions import least_confidence, random_sampling, normalized_least_confidence
from CNN_CNN_LSTM import CNN_CNN_LSTM
from CNN_biLSTM_CRF import cnn_bilstm_crf
from early_stopping import DevSetF1_ES, DUTE_ES, BatchDisparity_ES, DevSetLoss_ES, naive_var_computation
from self_label import self_label

parser = argparse.ArgumentParser(description='Active-self learning training procedure for NER models!')
parser.add_argument('--save_training_path', dest='save_training_path', type=str, default=None, help='Path to save training history and hyperaparms used')
parser.add_argument('--save_model_path', dest='save_model_path', type=str, default=None, help='Path to save trained model')
# dataset parameters
parser.add_argument('--train_path', dest='train_path', action='store', type=str, default=None, help='Path to load training set from')
parser.add_argument('--valid_path', dest='valid_path', action='store', type=str, default=None, help='Path to load validation set from')
parser.add_argument('--test_path', dest='test_path', action='store', type=str, default=None, help='Path to load testing set from')
parser.add_argument('--dataset_format', dest='dataset_format', action='store', type=str, default='iob1', help='Format of the dataset (e.g. iob1, iob2, iobes)')
# Embedding parameters
parser.add_argument('--embedding_path', dest='embedding_path', type=str, default=None, help='Path to load pretrained embeddings from')
# General model parameters
parser.add_argument('--model', dest='model', action='store', type=str, default='CNN-CNN-LSTM', help='Neural NER model architecture')
parser.add_argument('--char_embedding_dim', dest='char_embedding_dim', action='store', type=int, default=30, help='Embedding dimension for each character')
parser.add_argument('--char_out_channels', dest='char_out_channels', action='store', type=int, default=50, help='# of channels to be used in 1-d convolutions to form character level word embeddings')
# CNN-CNN-LSTM specific parameters
parser.add_argument('--word_out_channels', dest='word_out_channels', action='store', type=int, default=800, help='# of channels to be used in 1-d convolutions to encode word-level features')
parser.add_argument('--word_conv_layers', dest='word_conv_layers', action='store', type=int, default=2, help='# of convolution blocks to be used to encode word-level features')
parser.add_argument('--decoder_layers', dest='decoder_layers', action='store', type=int, default=1, help='# of layers of the LSTM greedy decoder')
parser.add_argument('--decoder_hidden_size', dest='decoder_hidden_size', action='store', type=int, default=256, help='Size of the LSTM greedy decoder layer')
# CNN-biLSTM-CRF specific parameters
parser.add_argument('--lstm_hidden_size', dest='lstm_hidden_size', action='store', type=int, default=200, help='Size of the lstm for word-level feature encoder')
# Trainign parameters
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='Number of supervised training epochs')
parser.add_argument('--lr', dest='lr', action='store', type=float, default=0.015, help='Learning rate for NER model training')
parser.add_argument('--grad_clip', dest='grad_clip', action='store', type=float, default=5.0, help='Value at which to clip the model gradient throughout training')
parser.add_argument('--momentum', dest='momentum', action='store', type=float, default=0.9, help='Momentum for the SGD optimization process')
parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=16, help='Batch size for training')
parser.add_argument('--early_stop_method', dest='early_stop_method', action='store', type=str, default='DevSetF1', help='Early stopping technique to be used')
parser.add_argument('--patience', dest='patience', action='store', type=int, default=5, help='Patience of the early stopping technique')
# Active learning parameters
parser.add_argument('--initial_set_seed', dest='initial_set_seed', action='store', type=int, default=0, help='Seed for reproducible experiments')
parser.add_argument('--labeled_percent_stop', dest='labeled_percent_stop', action='store', type=float, default=0.5, help='Percentage of the training set that must go to the labeled set before stoping the active learning algorithm')
parser.add_argument('--query_fn', dest='query_fn', action='store', type=str, default='normalized_least_confidence', help='Querying/sampling function to be used')
parser.add_argument('--query_budget', dest='query_budget', action='store', type=int, default=4000, help='Number of words that can be queried for annotation in a single active learning iteration')
# Self labeling options
parser.add_argument('--TokenSelfLabel_flag', dest='TokenSelfLabel_flag', action='store', type=int, default=0, help='Flag to define whether to use or not the token level self-label with refinement of predictions (0 to not use, 1 to use)')
parser.add_argument('--min_confidence', dest='min_confidence', action='store', type=float, default=0.99, help='Minimum confidence of the trained model to self label a token')
parser_opt = parser.parse_args()

assert parser_opt.TokenSelfLabel_flag in [0, 1], f'Value of the TokenSelfLabel_flag argument must be 0 (for not using self-label) or 1 (for using self-label)'
assert not(parser_opt.model == 'CNN-biLSTM-CRF' and parser_opt.TokenSelfLabel_flag == 1), f'Model CNN-biLSTM-CRF is not compatible with token level self-label, disable self-labeling or change the neural model'
print(f'\n****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'****************************************************************************************************************')
print(f'Experiment: Active learning')
print(f'Token level self-labeling: {parser_opt.TokenSelfLabel_flag}')
print(f'Training set file: {parser_opt.train_path}')
print(f'Model: {parser_opt.model}')
print(f'Initial seed: {parser_opt.initial_set_seed}')
print(f'batch size: {parser_opt.batch_size}')

# ==============================================================================================
# ==============================================================================================
# =============================     Load embeddings     ========================================
# ==============================================================================================
# ==============================================================================================
emb = KeyedVectors.load(parser_opt.embedding_path)

# bias = math.sqrt(3/emb.vector_size)
# if '<START>' not in emb:
#     emb.add('<START>', np.random.uniform(-bias, bias, emb.vector_size))
# if '<END>' not in emb:
#     emb.add('<END>', np.random.uniform(-bias, bias, emb.vector_size))
# if '<UNK>' not in emb:
#     emb.add('<UNK>', np.random.uniform(-bias, bias, emb.vector_size))
# if '<PAD>' not in emb:
#     emb.add('<PAD>', np.zeros(100))
if '<START>' not in emb:
    emb.add('<START>', np.random.uniform(0.1,1,100))
if '<END>' not in emb:
    emb.add('<END>', np.random.uniform(0.1,1,100))
if '<UNK>' not in emb:
    emb.add('<UNK>', np.random.uniform(0.1,1,100))
if '<PAD>' not in emb:
    emb.add('<PAD>', np.zeros(100))

# ==============================================================================================
# ==============================================================================================
# ============================ Create train and test sets ======================================
# ==============================================================================================
# ==============================================================================================

collate_object = new_custom_collate_fn(pad_idx=emb.key_to_index['<PAD>'], unk_idx=emb.key_to_index['<UNK>'])

print('\nGenerating text2idx dictionaries (word, char, tag)')
word2idx = create_word2idx_dict(emb, parser_opt.train_path)
char2idx = create_char2idx_dict(train_path=parser_opt.train_path)
tag2idx  = create_tag2idx_dict(train_path=parser_opt.train_path)

print('\nCreating training dataset')
train_set = active_dataset(path=parser_opt.train_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)
total_word_count = train_set.get_word_count()
print(f'Total word count on the unlabeled set: {total_word_count}')
train_set.flag_labeled = False

print('\nCreating validation dataset')
valid_set  = active_dataset(path=parser_opt.valid_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)
valid_set.flag_labeled = False
valid_set.label_data([i for i in range(len(valid_set))])
valid_set.flag_labeled = True
valid_dataloader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_object)
print(f'Validation set: ')

print('\nCreating test dataset')
test_set  = active_dataset(path=parser_opt.test_path, word2idx_dic=word2idx, char2idx_dic=char2idx, tag2idx_dic=tag2idx, data_format=parser_opt.dataset_format)
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

if parser_opt.model == "CNN-CNN-LSTM":
    model = CNN_CNN_LSTM(char_vocab_size=len(char2idx),
                            char_embedding_dim=parser_opt.char_embedding_dim,
                            char_out_channels=parser_opt.char_out_channels,
                            pretrained_word_emb=emb,
                            word2idx = word2idx,
                            word_out_channels=parser_opt.word_out_channels,
                            word_conv_layers = parser_opt.word_conv_layers,
                            num_classes=len(tag2idx),
                            decoder_layers = parser_opt.decoder_layers,
                            decoder_hidden_size = parser_opt.decoder_hidden_size,
                            device=device)

elif parser_opt.model == "CNN-biLSTM-CRF":
    model = cnn_bilstm_crf(char_vocab_size=len(char2idx), 
                            char_embedding_dim=parser_opt.char_embedding_dim, 
                            char_out_channels=parser_opt.char_out_channels,
                            pretrained_word_emb=emb,
                            num_classes=len(tag2idx),
                            device=device,
                            lstm_hidden_size=parser_opt.lstm_hidden_size)

# Model training parameters (learning rate, momentum, gradient_clipping value)
lrate = parser_opt.lr
momentum = 0.9
clipping_value = 5.0

model = model.to(device)

# ==============================================================================================
# ==============================================================================================
# =============================== Define training hyperparams ==================================
# ==============================================================================================
# ==============================================================================================

# Defining supervised training hyperparameters
supervised_epochs = parser_opt.epochs
optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=momentum)
std_batch_size = parser_opt.batch_size

# Defining active learning (hyper)parameters
labeled_set_percent_stop = parser_opt.labeled_percent_stop
query_opt = {'random_sampling': 0, 'least_confidence': 1, 'normalized_least_confidence': 2}[parser_opt.query_fn]
query_fn = {0: random_sampling, 1: least_confidence, 2: normalized_least_confidence}[query_opt]
query_budget = parser_opt.query_budget


# ==============================================================================================
# ==============================================================================================
# =============================== Create initial labeled set ===================================
# ==============================================================================================
# ==============================================================================================

# Init first batch of labeled samples randomly (1% of the entire train set)
train_set.flag_labeled = False
total_word_count = train_set.get_word_count()
total_sent_count = len(train_set)
init_budget = math.ceil(total_word_count*0.01)

initial_set_seed = [10, 43, 94, 114, 157, 182][parser_opt.initial_set_seed]
initial_labeled_set_idx, _ = random_sampling(dataloader=DataLoader(train_set, batch_size=80, pin_memory=True, collate_fn = collate_object, shuffle=False), budget=init_budget, seed = initial_set_seed)
train_set.label_data(initial_labeled_set_idx)
train_set.flag_labeled = True
init_word_count = train_set.get_word_count()
print(f'initial labeled set size: {init_word_count/total_word_count}')

# ==============================================================================================
# ==============================================================================================
# =============================== Active learning algorithm ====================================
# ==============================================================================================
# ==============================================================================================
print(f'\nInitiating active learning algorithm\n\n')
f1_history = []
labeled_percentage = []
labeled_words_percentage = []
epochs_history = []
active_tokens_hist = []
self_tokens_hist = []
mislabeled_tokens_hist = []

# Active learning start
active_epochs = 10000
for active_epoch in range(active_epochs):

    train_set.flag_labeled = True
    batch_size = std_batch_size if len(train_set)/10 > std_batch_size else math.floor(len(train_set)/10)
    dataloader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, collate_fn = collate_object, shuffle=False)

    # Supervised training (traditional)
    model.train()
    validation_performance = []
    early_stop_flag = False
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(supervised_epochs):
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

        # Early stopping methods
        if parser_opt.early_stop_method == 'DevSetF1':
            early_stop_flag, validation_performance = DevSetF1_ES(model, valid_dataloader, validation_performance, parser_opt.patience, tag2idx)
            # Save model parameters if it has the best performance so far
            if validation_performance[-1] == max(validation_performance):
                    best_model = copy.deepcopy(model.state_dict())
            # If patience's been reached, load best performing model and break training at current epoch
            if early_stop_flag == True:
                print(f'Supervised training stopped at epoch {epoch} due to early stopping using f1-score on validation set')
                model.load_state_dict(best_model)
                epochs_history.append(epoch)
                break
            elif epoch == supervised_epochs-1:
                epochs_history.append(epoch)

        elif parser_opt.early_stop_method == 'DevSetLoss':
            early_stop_flag, validation_performance = DevSetLoss_ES(model, valid_dataloader, validation_performance, parser_opt.patience, tag2idx)
            # Save model parameters if it has the best performance so far
            if validation_performance[-1] == min(validation_performance):
                    best_model = copy.deepcopy(model.state_dict())
            # If patience's been reached, load best performing model and break training at current epoch
            if early_stop_flag == True:
                print(f'Supervised training stopped at epoch {epoch} due to early stopping using mean loss on validation set')
                model.load_state_dict(best_model)
                epochs_history.append(epoch)
                break
            elif epoch == supervised_epochs-1:
                epochs_history.append(epoch)

        elif parser_opt.early_stop_method == 'DUTE':
            if epoch == supervised_epochs - 1:
                epochs_history.append(supervised_epochs)
                print(f'Supervised training stopped at epoch {supervised_epochs} due to DUTE strategy')
                dataloader = DataLoader(train_set, batch_size=128, pin_memory=True, collate_fn = collate_object, shuffle=False)
                confidence = get_confidence(model, dataloader)
                supervised_epochs = DUTE_ES(supervised_epochs, confidence)
                print(f'Unsupervised confidence: {confidence}')

        elif parser_opt.early_stop_method == 'full_epochs':
            if epoch == supervised_epochs-1:
                print(f'Supervised training stopped at epoch {epoch} due to reaching chosen number of training epochs')
                epochs_history.append(epoch)
        
    
    # Verify performance on test set after supervised training
    model.eval()
    with torch.no_grad():
        predictions, targets = preprocess_pred_targ(model, test_dataloader, device)
        predictions = IOBES_tags(predictions, tag2idx)
        targets = IOBES_tags(targets, tag2idx)
        micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
        f1_history.append(0 if np.isnan(micro_f1) else micro_f1)
    
    labeled_percentage.append(len(train_set.labeled_sentences)/(len(train_set.labeled_sentences)+len(train_set.unlabeled_sentences)))
    labeled_word_count = train_set.get_word_count()
    labeled_words_percentage.append(labeled_word_count/total_word_count)
    print(f"Labeled data so far (word level): {labeled_word_count/total_word_count}")
    print(f'micro f1-score: {micro_f1}')

    # If enough data has been labeled, break active training loop
    if labeled_word_count/total_word_count >= labeled_set_percent_stop:
        break
    
    print(f'\n\nActive epoch {active_epoch}')
    print('Begin query generation!')
    # Active learning querying
    train_set.flag_labeled = False
    dataloader = DataLoader(train_set, batch_size=128, pin_memory=True, collate_fn = collate_object, shuffle=False)
    query_idx, residual_budget = query_fn(model=model, dataloader=dataloader, budget=query_budget, device=device)
    print('Begin annotating the queried samples!')
    query_idx = torch.LongTensor(query_idx).to('cpu')
    if parser_opt.TokenSelfLabel_flag != 0:
        active_tokens, self_tokens, mislabeled_tokens = self_label(model, dataloader, query_idx, parser_opt.min_confidence, collate_object, 1, word2idx, device)
        active_tokens_hist.append(active_tokens)
        self_tokens_hist.append(self_tokens)
        mislabeled_tokens_hist.append(mislabeled_tokens)
        print(f'Human labeled tokens: {active_tokens}\nSelf-labeled tokens: {self_tokens}\nMislabeled tokens: {mislabeled_tokens}')
    train_set.label_data(query_idx)

# ==============================================================================================
# ==============================================================================================
# ================================= Save training history ======================================
# ==============================================================================================
# ==============================================================================================

if parser_opt.save_training_path:
    hyperparams = {'model': str(model), 'LR': lrate, 'momentum': momentum, 'clipping': clipping_value}
    dic = { 'f1_hist': f1_history,
            'labeled_percent': labeled_percentage,
            'labeled_words_percent': labeled_words_percentage,
            'hyperparams': hyperparams,
            'initial_set_seed': initial_set_seed,
            'epochs_hist': epochs_history,
            'active_tokens_hist': active_tokens_hist,
            'self_tokens_hist': self_tokens_hist,
            'mislabeled_tokens_hist': mislabeled_tokens_hist}
    joblib.dump(dic, parser_opt.save_training_path)

if parser_opt.save_model_path:
    torch.save(model.state_dict(), parser_opt.save_model_path)

