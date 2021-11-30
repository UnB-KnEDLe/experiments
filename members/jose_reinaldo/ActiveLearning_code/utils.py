import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import unicodedata
import numpy as np
from bisect import bisect

def get_confidence(model, dataloader):
    """
    Returns the harmonic mean confidence of the model over its normalized probability for the unlabeled samples

    Input:
        model (pytorch): The neural model 
        dataloader (pytorch): The pytorch dataloader structure

    Output:
        mean_confidence (float value): A value that indicates the harmonic mean confidence of the model on the dataloader
    """
    prev_flag = dataloader.dataset.flag_labeled
    if isinstance(dataloader.dataset.flag_labeled, str):
        dataloader.dataset.flag_labeled = 'unlabeled'
    else:
        dataloader.dataset.flag_labeled = False
    with torch.no_grad():
        full_prob = torch.Tensor([]).to(next(model.parameters()).device)
        for sent, tag, word, mask in dataloader:
            sent = sent.to(next(model.parameters()).device)
            tag = tag.to(next(model.parameters()).device)
            word = word.to(next(model.parameters()).device)
            mask = mask.to(next(model.parameters()).device)
            pred, prob = model.decode(sent, word, mask)
            norm_prob = (prob.log() / mask.sum(dim=1)).exp()
            full_prob = torch.cat((full_prob, prob))
    mean_confidence = full_prob.shape[0]/((1/full_prob).sum())
    dataloader.dataset.flag_labeled = prev_flag
    return mean_confidence

def create_word2idx_dict(emb, train_path):
    """
    Creates dictionary that maps words from the pretrained embedding into unique integer values

    Input:
        emb (gensim embedding): pretrained (gensim) embedding
        train_path (string): path to the training set file (unused in current version)

    Output:
        dic (python dictionary): maps words to unique integer values
    """
    # dic = {}
    # for word in emb.index2word:
    #   dic[word] = emb.vocab[word].index
    #   dic[word] = emb.key_to_index[word]
    return emb.key_to_index
    # return dic

def create_char2idx_dict(train_path):
    """
    Creates dictionary that maps characters from the training file into unique integer values

    Input:
        train_path (string): path to the training set file (must be conll format)

    Output:
        dic (python dictionary): maps characters to unique integer values
    """
    f = open(train_path, 'r').readlines()
    dic = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for line in f:
        if line == '\n':
            continue
        word = line.split()[0]
        for char in word:
            if char not in dic:
                dic[char] = len(dic)
    return dic

def create_tag2idx_dict(train_path):
    """
    Creates dictionary that maps NER tags from the training file into unique integer values

    Input:
        train_path (string): path to the training set file (must be conll format)

    Output:
        iob2_dic (python dictionary): maps classes to unique integer values
    """
    f = open(train_path, 'r').readlines()
    dic = {}
    for line in f:
        if line != '\n':
            tag = line.split()[3]
            if tag not in dic and tag[0]=='I':
                dic[tag] = len(dic)
    iob2_dic = {'<PAD>': 0, 'O': 1}
    for tag in dic:
        iob2_dic['B'+tag[1:]] = len(iob2_dic)
        iob2_dic[tag] = len(iob2_dic)
        iob2_dic['S'+tag[1:]] = len(iob2_dic)
        iob2_dic['E'+tag[1:]] = len(iob2_dic)

    # iob2_dic['<GO>'] = len(iob2_dic)
    return iob2_dic

class new_custom_collate_fn():
    def __init__(self, pad_idx, unk_idx):
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
    def __call__(self, batch):
        words = [torch.LongTensor(batch[i][0]) for i in range(len(batch))]
        tags  = [torch.LongTensor(batch[i][1]) for i in range(len(batch))]
        chars = [batch[i][2].copy() for i in range(len(batch))]

        # Pad word/tag level
        words = pad_sequence(words, batch_first = True, padding_value=self.pad_idx)
        tags  = pad_sequence(tags, batch_first = True, padding_value = 0)

        # Pad character level
        max_word_len = -1
        for sentence in chars:
            for word in sentence:
                max_word_len = max(max_word_len, len(word))
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = [0 if k >= len(chars[i][j]) else chars[i][j][k] for k in range(max_word_len)]
        for i in range(len(chars)):
            chars[i] = [[0 for _ in range(max_word_len)] if j>= len(chars[i]) else chars[i][j] for j in range(words.shape[1])]
        chars = torch.LongTensor(chars)

        mask = words != self.pad_idx

        return words, tags, chars, mask

def budget_limit(list_idx, budget, dataloader):
    """
    Function that restricts the number of queried samples by the available query budget

    Input:
        list_idx (python list): list of indices for the unlabeled samples (ordered from the most uncertain to least uncertain - output of the sampling function chosen)
        budget (integer): integer value that represents the number of words (budget) that can be queried for annotation in a given active learning round
        dataloader (pytorch): The pytorch dataloader structure

    Output:
        budget_list (python list): List of the indices for the unlabeled samples that are to be queried for annotation
        budget (integer): Unused budget, number of words that were not queried
    """
    budget_list = []
    for i in range(len(list_idx)):
        sent_len = len(dataloader.dataset.sentences[dataloader.dataset.unlabeled_sentences[list_idx[i]]]) - 2
        if sent_len <= budget:
            budget_list.append(list_idx[i])
            budget -= sent_len
    return budget_list, budget

