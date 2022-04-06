from collections import OrderedDict
import itertools
import torch
import pickle

def preprocess(ner_set):
    sentences = []
    tags = []

    temp_sentence = []
    temp_tag = []
    for line in ner_set:
        try:
            word, _, _, tag = line.split()
            temp_sentence.append(word)
            temp_tag.append(tag)
        except:
            sentences.append(temp_sentence)
            tags.append(temp_tag)
            temp_sentence = []
            temp_tag = []

    if temp_sentence:
        sentences.append(temp_sentence)
        tags.append(temp_tag)
    return sentences, tags

def word_dict(sentences):
    word2idx = OrderedDict({'<UNK>': 0, '<PAD>': 1, '<BOS>': 2, '<EOS>': 3})
    for sentence in sentences:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

def tag_dict(tag_sentences):
    tag2idx = OrderedDict({'<PAD>': 0})
    for tags in tag_sentences:
        for tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return tag2idx

def numericalize(sentences, word2idx, tag_sentences, tag2idx):
    numericalized_sentences = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in sentences]
    numericalized_tags = [[tag2idx[tag] for tag in tags] for tags in tag_sentences]
    return numericalized_sentences, numericalized_tags

def add_special_tokens(sentences, word2idx, tag_sentences, tag2idx):
    formatted_sentences = [torch.LongTensor([word for word in itertools.chain([word2idx['<BOS>']], sentence, [word2idx['<EOS>']])]) for sentence in sentences]
    formatted_tags = [torch.LongTensor([tag for tag in itertools.chain([tag2idx['O']], tags, [tag2idx['O']])]) for tags in tag_sentences]
    return formatted_sentences, formatted_tags

def read_pkl(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
    
def write_pkl(file, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def IOBify(tags_sequence, idx2tag):
    if isinstance(tags_sequence[0], list):
        iob_y = [[idx2tag[tag] for tag in tags] for tags in tags_sequence]
    else:
        iob_y = [[idx2tag[tag.item()] for tag in tags] for tags in tags_sequence]
    return iob_y