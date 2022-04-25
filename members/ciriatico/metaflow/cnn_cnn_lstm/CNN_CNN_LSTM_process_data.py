import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import unicodedata
import numpy as np
from bisect import bisect
from math import sqrt
from torch.utils.data import Dataset, DataLoader
import itertools
import matplotlib.pyplot as plt
import re

def create_word2idx_dict(emb, train_set):
    # dic = {}
    # for word in emb.index2word:
    #   dic[word] = emb.vocab[word].index
    # return dic
    return emb.key_to_index

def create_char2idx_dict(train_set):
    dic = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for line in train_set:
        if line == '\n':
            continue
        word = line.split()[0]
        for char in word:
            if char not in dic:
                dic[char] = len(dic)
    return dic

def create_tag2idx_dict(train_set):
    dic = {}
    for line in train_set:
        if line != '\n':
            tag = line.split()[3]
            if tag not in dic and tag[0]=='B':
                dic[tag] = len(dic)
    iob2_dic = {'<PAD>': 0, 'O': 1}
    for tag in dic:
        iob2_dic[tag] = len(iob2_dic)
        iob2_dic['I'+tag[1:]] = len(iob2_dic)
        iob2_dic['S'+tag[1:]] = len(iob2_dic)
        iob2_dic['E'+tag[1:]] = len(iob2_dic)

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
    budget_list = []
    for i in range(len(list_idx)):
        sent_len = len(dataloader.dataset.sentences[dataloader.dataset.unlabeled_sentences[list_idx[i]]]) - 2
        if sent_len <= budget:
            budget_list.append(list_idx[i])
            budget -= sent_len
    return budget_list, budget

def budget_limit2(list_idx, budget, dataloader):
    """
    Changes done to adapt to active_self_dataset
    """
    budget_list = []
    for i in range(len(list_idx)):
        sent_len = len(dataloader.dataset.sentences[dataloader.dataset.unlabeled_set[list_idx[i]]]) - 2
        if sent_len <= budget:
            budget_list.append(list_idx[i])
            budget -= sent_len
    return budget_list, budget

def augment_pretrained_embedding(embedding, train_set):
    """
    Augment pretrained embeddings with tokens from the training set
    """
    vocab = {}
    for line in train_set:
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
        if word not in embedding and word.lower() not in embedding:
            not_found[word] = vocab[word]
        else:
            found[word] = vocab[word]

    bias = sqrt(3/embedding.vector_size)
    for word in not_found:
        embedding.add(word, np.random.uniform(-bias, bias, embedding.vector_size))

class CustomDropout(torch.nn.Module):
    """
    Custom dropout layer based on inverted dropout to allow for frozen dropout masks
    """
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        assert p > 0 and p < 1, 'Dropout probability out of range (0 < p < 1)'
        self.p = p
        self.drop_mask = None
        self.repeat_mask_flag = False

    def forward(self, x):
        if self.training:
            if not self.repeat_mask_flag:
                self.drop_mask = torch.distributions.binomial.Binomial(probs=1-self.p).sample(x.size()).to(x.device)
                self.drop_mask *= (1.0/(1-self.p))
            return x * self.drop_mask
        return x


class dataset(Dataset):
    def __init__(self, word2idx_dic, char2idx_dic, tag2idx_dic, d_set, data_format='iob1'):
        super(dataset, self).__init__()
        self.data_format = data_format
        self.word2idx_dic = word2idx_dic
        self.char2idx_dic = char2idx_dic
        self.tag2idx_dic  = tag2idx_dic
        # Initialize full dataset
        self.sentences = []
        self.tags = []
        self.words = []

        self.sentences, self.tags, self.words = self.load_data(d_set)

        self.char2idx()
        self.word2idx()
        self.tag2idx()

    def __getitem__(self, index):
        return self.sentences[index], self.tags[index], self.words[index]

    def __len__(self):
        return len(self.sentences)
        # if self.flag_labeled:
        #     return len(self.labeled_sentences)
        # else:
        #     return len(self.unlabeled_sentences)

    def load_data(self, d_set):
        temp_sentences = []
        temp_tags = []
        temp_words = []

        sentences = []
        tags = []
        words = []

        for line in d_set:
            if line == '\n' or not line:
                if temp_sentences:
                    temp_sentences = [word for word in itertools.chain(['<START>'], temp_sentences, ['<END>'])]
                    temp_tags = [tag for tag in itertools.chain(['O'], temp_tags, ['O'])]
                    temp_words = [[item for item in itertools.chain(['<START>'], [char for char in word], ['<END>'])] for word in temp_sentences]
                    sentences.append(temp_sentences)
                    words.append(temp_words)
                    if self.data_format == 'iob1':
                        tags.append(self.convert_IOB2_2_IOBES(self.convert_IOB1_2_IOB2(temp_tags)))
                    elif self.data_format == 'iob2':
                        tags.append(self.convert_IOB2_2_IOBES(temp_tags))
                    else:
                        tags.append(temp_tags)

                    temp_sentences = []
                    temp_tags = []
                    temp_words = []

            else:
                temp_sentences.append(line.split()[0])
                temp_tags.append(line.split()[3])
                
        if temp_sentences:
            temp_sentences = [word for word in itertools.chain(['<START>'], temp_sentences, ['<END>'])]
            temp_tags = [tag for tag in itertools.chain(['O'], temp_tags, ['O'])]
            temp_words = [[item for item in itertools.chain(['<START>'], [char for char in word], ['<END>'])] for word in temp_sentences]
            sentences.append(temp_sentences)
            words.append(temp_words)

            if self.data_format == 'iob1':
                tags.append(self.convert_IOB2_2_IOBES(self.convert_IOB1_2_IOB2(temp_tags)))
            elif self.data_format == 'iob2':
                tags.append(self.convert_IOB2_2_IOBES(temp_tags))
            else:
                tags.append(temp_tags)

        return sentences, tags, words
    
    def convert_IOB1_2_IOB2(self, sentence):
        prev_tag = 'O'
        for i in range(len(sentence)):
            if sentence[i][0] == 'I' and sentence[i] != prev_tag and prev_tag != 'B'+sentence[i][1:]:
                sentence[i] = 'B' + sentence[i][1:]
            prev_tag = sentence[i]
        return sentence

    def convert_IOB2_2_IOBES(self, sentence):
      for i in range(len(sentence)):
        if sentence[i][0] == 'I' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
          sentence[i] = 'E' + sentence[i][1:]
        elif sentence[i][0] == 'B' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
          sentence[i] = 'S' + sentence[i][1:]
      return sentence

    def clean_numbers(self, x):

        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def word2idx(self):
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                word = self.clean_numbers(self.sentences[i][j])
                if word in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word]
                elif word.lower() in self.word2idx_dic:
                    self.sentences[i][j] = self.word2idx_dic[word.lower()]
                else:
                    self.sentences[i][j] = self.word2idx_dic['<UNK>']

    def tag2idx(self):
        for i in range(len(self.tags)):
            for j in range(len(self.tags[i])):
                self.tags[i][j] = self.tag2idx_dic[self.tags[i][j]]

    def char2idx(self):
        self.words = [[[self.char2idx_dic['<UNK>'] if char not in self.char2idx_dic else self.char2idx_dic[char] for char in word] for word in sentence] for sentence in self.words]

    def sort_set(self, unordered_sentences, unordered_words, unordered_tags):
        # Change here
        ordered_idx = np.argsort([len(self.sentences[unordered_sentences[i]]) for i in range(len(unordered_sentences))])
        ordered_sentences = [unordered_sentences[i] for i in ordered_idx]
        ordered_words = [unordered_words[i] for i in ordered_idx]
        ordered_tags = [unordered_tags[i] for i in ordered_idx]
        return ordered_sentences, ordered_words, ordered_tags

    def get_word_count(self):
        # Change here
        if self.flag_labeled:
            sentences_idx = self.labeled_sentences
        else:
            sentences_idx = self.unlabeled_sentences
        word_count = 0
        for idx in sentences_idx:
            sentence = self.sentences[idx]
            # word_count += len(sentence) - 2
            for word in sentence:
                if word != self.word2idx_dic['<PAD>'] and word != self.word2idx_dic['<START>'] and word != self.word2idx_dic['<END>']:
                    word_count += 1
        return word_count