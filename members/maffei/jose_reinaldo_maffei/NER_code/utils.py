import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import unicodedata
import numpy as np
from bisect import bisect

from gensim.models import KeyedVectors


def find_entities(tag):
    entities = []
    prev_tag = 1
    begin_entity = -1

    for i in range(len(tag)):
        # Check if current tag is new entity by checking if it's 'B-' of any class
        if tag[i]%2==0 and tag[i]>=2:
            if prev_tag >=2:
                entities.append((begin_entity, i-1, prev_tag-1))
            begin_entity = i
            prev_tag = tag[i]+1
        # Check if current tag is new entity (by comparing to previous tag)
        elif tag[i] != prev_tag:
            if prev_tag >= 2:
                entities.append((begin_entity, i-1, prev_tag-1))
            begin_entity = i
            prev_tag = tag[i]
    # Check if entity continues to the end of tensor tag
    if prev_tag >=2:
        entities.append((begin_entity, len(tag)-1, prev_tag-1))
    return entities

def create_word2idx_dict(emb, train_path):
    dic = {}
    for word in emb.index2word:
      dic[word] = emb.vocab[word].index
    return dic

def create_char2idx_dict(train_path):
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

def find_iobes_entities(sentence, tag2idx):
  t = [2+i for i in range(0, len(tag2idx), 4)]
  entities = []
  tag_flag = False
  entity_start = -1
  curr_tag_class = 0
  for i in range(len(sentence)):
    tag = sentence[i]
    # 'O' or '<PAD>' or '<GO>' classes
    if tag == 0 or tag == 1:# or tag == tag2idx['<GO>']:
      curr_tag_class = 0
      tag_flag == False
      continue
    tag_class = t[bisect(t, tag)-1]
    tag_mark  = tag - tag_class
    # B- class
    if tag_mark == 0:
      tag_flag = True
      entity_start = i
      curr_tag_class = tag_class
    # I- class
    elif tag_mark == 1 and curr_tag_class != tag_class:
      tag_flag = False
    # S- class
    elif tag_mark == 2:
      entities.append((i, i, tag_class))
      tag_flag = False
    # E- class
    elif tag_mark == 3:
      if tag_flag and (curr_tag_class == tag_class):
        entities.append((entity_start, i, tag_class))
      tag_flag = False
  return entities

def find_iobes_entities2(sentence, tag2idx):
  t = [2+i for i in range(0, len(tag2idx), 4)]
  entities = set({})
  tag_flag = False
  entity_start = -1
  curr_tag_class = 0
  for i in range(len(sentence)):
    tag = sentence[i]
    # 'O' or '<PAD>' or '<GO>' classes
    if tag == 0 or tag == 1:# or tag == tag2idx['<GO>']:
      curr_tag_class = 0
      tag_flag == False
      continue
    tag_class = t[bisect(t, tag)-1]
    tag_mark  = tag - tag_class
    # B- class
    if tag_mark == 0:
      tag_flag = True
      entity_start = i
      curr_tag_class = tag_class
    # I- class
    elif tag_mark == 1 and curr_tag_class != tag_class:
      tag_flag = False
    # S- class
    elif tag_mark == 2:
      entities.add((i, i, tag_class))
      tag_flag = False
    # E- class
    elif tag_mark == 3:
      if tag_flag and (curr_tag_class == tag_class):
        entities.add((entity_start, i, tag_class))
      tag_flag = False
  return entities

from pathlib import Path


def load_embedding(parser_opt, base_path: Path):
    if parser_opt.dataset == 'conll':
        train_path = base_path / 'conll03/eng_train.txt'
        if parser_opt.use_dev_set:
            test_path = base_path / 'conll03/eng_testa.txt'
        else:
            test_path  = base_path / 'conll03/eng_testb.txt'
        data_format = 'iob1'
        embedding_path = ''
        emb = KeyedVectors.load(embedding_path.as_posix())

    elif parser_opt.dataset == 'ontonotes':
        train_path = 'ontonotes/good_splits/onto.train.ner'
        if parser_opt.use_dev_set:
            test_path = 'ontonotes/good_splits/onto.development.ner'
        else:
            test_path  = 'ontonotes/good_splits/onto.test.ner'
        data_format = 'iob2'
        embedding_path = ''
        emb = KeyedVectors.load(embedding_path.as_posix())

    elif parser_opt.dataset == 'aposentadoria':
        train_path = 'dataset/aposentadoria_train.txt'
        if parser_opt.use_dev_set:
            test_path = 'dataset/aposentadoria_testa.txt'
        else:
            test_path  = 'dataset/aposentadoria_testb.txt'
        data_format = 'iob2'

        embedding_path = base_path /  'pt_embedding/skip_word2vec_100d.kv'
        emb = KeyedVectors.load(embedding_path.as_posix())
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
        # found = {}
        not_found = {}
        for word in vocab:
            if word not in emb and word.lower() not in emb:
                not_found[word] = vocab[word]
            # else:
            #     found[word] = vocab[word] 
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
    return emb, train_path.as_posix(), test_path.as_posix()


# def custom_collate_fn(batch):
#     words = [torch.LongTensor(batch[i][0]) for i in range(len(batch))]
#     tags  = [torch.LongTensor(batch[i][1]) for i in range(len(batch))]
#     chars = [batch[i][2].copy() for i in range(len(batch))]

#     # Pad word/tag level
#     # words = pad_sequence(words, batch_first = True, padding_value=314815) # glove
#     words = pad_sequence(words, batch_first = True, padding_value=3000000) # googlenews
#     tags  = pad_sequence(tags, batch_first = True, padding_value = 0)

#     # Pad character level
#     max_word_len = -1
#     for sentence in chars:
#         for word in sentence:
#             max_word_len = max(max_word_len, len(word))
#     for i in range(len(chars)):
#         for j in range(len(chars[i])):
#             chars[i][j] = [0 if k >= len(chars[i][j]) else chars[i][j][k] for k in range(max_word_len)]
#     for i in range(len(chars)):
#         chars[i] = [[0 for _ in range(max_word_len)] if j>= len(chars[i]) else chars[i][j] for j in range(words.shape[1])]
#     chars = torch.LongTensor(chars)

#     # mask = words != 314815 # glove
#     mask = words != 3000000 # googlenews

#     return words, tags, chars, mask
