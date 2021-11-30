import torch
# Libs to create Dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import re

class active_dataset(Dataset):
    """
    Dataset to be used in the active learning algorithm (separates unlabeled and labeled samples)

    -> Defines __init__(), __getitem__() and __len__() methods, which are required for the pytorch Dataset class.

    -> Defines the load_data(), convert_IOB1_2_IOB2(), convert_IOB2_2_IOBES(), clean_numbers(), word2idx(), tag2idx(), and char2idx()
    methods in order to load the datasets and format accordingly 

    -> Defines the label_data(), sort_set(), and get_word_count() to be during the active learning process

    Args:
        word2idx_dic (python dictionary): Dictionary that maps words (string) to indices (integer values)
        char2idx_dic (python dictionary): Dictionary that maps characters (char) to indices (integer values)
        tag2idx_dic (python dictionary): Dictionary that maps classes (string) to indices (integer values)
        path (string): Path to the data file (one of the three train-dev-set files), file must be in conll format
        data_format (string): Format of the data file (supports: 'iob1' and 'iob2')

    """
    def __init__(self, word2idx_dic, char2idx_dic, tag2idx_dic, path, data_format='iob1'):
        super(active_dataset, self).__init__()
        self.data_format = data_format
        self.word2idx_dic = word2idx_dic
        self.char2idx_dic = char2idx_dic
        self.tag2idx_dic  = tag2idx_dic
        # Initialize labeled set
        self.labeled_sentences = []
        self.labeled_tags  = []
        self.labeled_words = []
        # Initialize unlabeled set
        self.unlabeled_sentences = []
        self.unlabeled_tags  = []
        self.unlabeled_words = []
        # Initialize full dataset
        self.sentences = []
        self.tags = []
        self.words = []

        # Flag to control labeled/unlabeled data to be retrieved by __getitem__()
        self.flag_labeled = False

        self.sentences, self.tags, self.words = self.load_data(path)

        self.char2idx()
        self.word2idx()
        self.tag2idx()

        idx = [i for i in range(len(self.sentences))]
        self.unlabeled_sentences = idx
        self.unlabeled_tags = idx
        self.unlabeled_words = idx

    def __getitem__(self, index):
        if self.flag_labeled:
            return self.sentences[self.labeled_sentences[index]], self.tags[self.labeled_tags[index]], self.words[self.labeled_words[index]]
        else:
            return self.sentences[self.unlabeled_sentences[index]], self.tags[self.unlabeled_tags[index]], self.words[self.unlabeled_words[index]]

    def __len__(self):
        if self.flag_labeled:
            return len(self.labeled_sentences)
        else:
            return len(self.unlabeled_sentences)

    def load_data(self, path):
        """
        Method that loads the dataset from the file specified by the path, adds special tokens (<START> and <END>) to limit the 
        boundaries of the sentence, and transforms to IOBES notation

        Input:
            path (string): path to the file containing the dataset to be labeled (either train, dev, test sets)

        Output:
            sentences (python list): List containing the formated sentences (each sentence presented as a list of words)
            tags (python list):  List containing the classes loaded
            words (python list): List containing the formated words (each word presented as a list of characters)
        """

        f = open(path, 'r').readlines()

        temp_sentences = []
        temp_tags = []
        temp_words = []

        sentences = []
        tags = []
        words = []

        for line in f:
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
        """
        Converts the tags in IOB1 notation to the IOB2 notation

        Input:
            sentence (python list): List of tags in IOB1 format

        Output:
            sentence (python list): List of tags in IOB2 format
        """

        prev_tag = 'O'
        for i in range(len(sentence)):
            if sentence[i][0] == 'I' and sentence[i] != prev_tag and prev_tag != 'B'+sentence[i][1:]:
                sentence[i] = 'B' + sentence[i][1:]
            prev_tag = sentence[i]
        return sentence

    def convert_IOB2_2_IOBES(self, sentence):
        """
        Converts the tags in IOB2 notation to the IOBES notation

        Input:
            sentence (python list): List of tags in IOB2 format

        Output:
            sentence (python list): List of tags in IOBES format
        """
        for i in range(len(sentence)):
            if sentence[i][0] == 'I' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
                sentence[i] = 'E' + sentence[i][1:]
            elif sentence[i][0] == 'B' and (i+1==len(sentence) or sentence[i+1][0] != 'I'):
                sentence[i] = 'S' + sentence[i][1:]
        return sentence

    def clean_numbers(self, x):
        """
        Replaces digits for the # symbol (to make the input compatible with the pretrained embedding)
        """
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def word2idx(self):
        """
        Replaces words (string) by their respective indices (integer value) from the word2idx_dic dictionary
        """
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
        """
        Replaces tags (string) by their respective indices (integer value) from the tag2idx_dic dictionary
        """
        for i in range(len(self.tags)):
            for j in range(len(self.tags[i])):
                self.tags[i][j] = self.tag2idx_dic[self.tags[i][j]]

    def char2idx(self):
        """
        Replaces characters (string) by their respective indices (integer value) from the char2idx_dic dictionary
        """
        self.words = [[[self.char2idx_dic['<UNK>'] if char not in self.char2idx_dic else self.char2idx_dic[char] for char in word] for word in sentence] for sentence in self.words]

    def label_data(self, idx_list):
        """
        Switches samples from the unlabeled set to the labeled set

        Input:
            idx_list (python list): List containing the indices of the unlabeled samples to be sent to the labeled set
        """
        
        new_unlabeled_sentences = [self.unlabeled_sentences[i] for i in range(len(self.unlabeled_sentences)) if i not in idx_list]
        new_unlabeled_tags = [self.unlabeled_tags[i] for i in range(len(self.unlabeled_tags)) if i not in idx_list]
        new_unlabeled_words = [self.unlabeled_words[i] for i in range(len(self.unlabeled_words)) if i not in idx_list]
        
        new_labeled_sentences = [self.unlabeled_sentences[i] for i in idx_list]
        new_labeled_tags = [self.unlabeled_tags[i] for i in idx_list]
        new_labeled_words = [self.unlabeled_words[i] for i in idx_list]
        
        self.unlabeled_sentences = new_unlabeled_sentences
        self.unlabeled_tags = new_unlabeled_tags
        self.unlabeled_words = new_unlabeled_words
        self.labeled_sentences += new_labeled_sentences
        self.labeled_tags += new_labeled_tags
        self.labeled_words += new_labeled_words
        
        # Sort sentences by sentence length
        self.unlabeled_sentences, self.unlabeled_words, self.unlabeled_tags = self.sort_set(self.unlabeled_sentences, self.unlabeled_words, self.unlabeled_tags)
        self.labeled_sentences, self.labeled_words, self.labeled_tags = self.sort_set(self.labeled_sentences, self.labeled_words, self.labeled_tags)

    def sort_set(self, unordered_sentences, unordered_words, unordered_tags):
        """
        Sorts the labeled set be the length of the sentences, in order to group sentences of the same length in the same batch, 
        as proposed by (Shen et al - https://arxiv.org/abs/1707.05928)

        Input:
            unordered_sentences (python list): unordered sentences
            unordered_words (python list): unordered words
            unordered_tags (python list): unordered tags
            
        Output:
            ordered_sentences (python list): sentences ordered by sentence length
            ordered_words (python list): words ordered by the sentence length
            ordered_tags (python list): tags ordered by the sentence length
        """
        ordered_idx = np.argsort([len(self.sentences[unordered_sentences[i]]) for i in range(len(unordered_sentences))])
        ordered_sentences = [unordered_sentences[i] for i in ordered_idx]
        ordered_words = [unordered_words[i] for i in ordered_idx]
        ordered_tags = [unordered_tags[i] for i in ordered_idx]
        return ordered_sentences, ordered_words, ordered_tags

    def get_word_count(self):
        """
        Counts number of words in the labeled or unlabeled set (depending on the self.flag_labeled flag)

        Outputs:
            word_count (integer value): Number of words in the labeled/unlabeled set
        """
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

