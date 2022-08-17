"""
This module implements a class for processing the segmentation dataset.
It is used within the SegmentationFlow flow.
"""

from math import floor
import torch
from torch.utils.data import Dataset
import copy


class SegmentationDataset(Dataset):
    def __init__(
        self,
        tag2idx,
        word2idx,
        set_type,
        path,
        min_sequence_length=3,
        max_sequence_lenght=20,
        max_sentence_length=30,
    ):
        """
        Load the segmentation dataset, (self.__load__())
        Truncate long sentences/split long sequences/ignore very short sequences (self.__trunc__())
        get indices of words and tags (self.__get_idx__())
        PAD the sentences and sequences/create MASKS representing padded elements (self.__pad__())
        """
        self.set_type = set_type
        self.path = path

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_lenght
        self.max_sentence_length = max_sentence_length

        self.__load__()
        self.__trunc__()
        self.__split__()
        self.__get_idx__(tag2idx, word2idx)
        self.__pad__()

        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)
        self.mask = torch.ByteTensor(self.mask)

    def __getitem__(self, index):
        """
        Get one item from the dataset
        """
        return self.x[index], self.mask[index], self.y[index]

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.x)

    def __load__(self):
        """
        Load raw dataset
        """
        data = open(self.path, "r").read()
        self.x, self.y = [], []
        for sequence in data.split("\n\n"):
            x_sequence, y_sequence = [], []
            lines = sequence.split("\n")
            for line in lines:
                line = line.split()
                if line != []:
                    x_sequence.append(line[1:])
                    y_sequence.append(line[0])
            self.x.append(x_sequence)
            self.y.append(y_sequence)

    def __trunc__(self):
        """
        Truncate sentences with length > max_sentence_length
        Separate sequences with length > max_sequence_length
        Delete sequences with length < min_sequence_length
        """
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                if len(self.x[i][j]) > self.max_sentence_length:
                    self.x[i][j] = self.x[i][j][: self.max_sentence_length]

        sep_x = []
        sep_y = []
        for i in range(len(self.x)):
            if len(self.x[i]) > self.max_sequence_length:
                for j in range((len(self.x[i]) // self.max_sequence_length) + 1):
                    if self.x[i][
                        j
                        * self.max_sequence_length : (j + 1)
                        * self.max_sequence_length
                    ]:
                        sep_x.append(
                            self.x[i][
                                j
                                * self.max_sequence_length : (j + 1)
                                * self.max_sequence_length
                            ]
                        )
                        sep_y.append(
                            self.y[i][
                                j
                                * self.max_sequence_length : (j + 1)
                                * self.max_sequence_length
                            ]
                        )
            elif len(self.x[i]) < self.min_sequence_length:
                continue
            else:
                sep_x.append(self.x[i])
                sep_y.append(self.y[i])

        self.truncated = copy.deepcopy(sep_x)
        self.x = sep_x
        self.y = sep_y

    def __split__(self):
        train_split = floor(0.6 * len(self.x))
        valid_split = floor(0.1 * len(self.x)) + train_split
        if self.set_type == "whole":
            pass
        elif self.set_type == "train":
            self.x = self.x[:train_split]
            self.y = self.y[:train_split]
        elif self.set_type == "valid":
            self.x = self.x[train_split:valid_split]
            self.y = self.y[train_split:valid_split]
        elif self.set_type == "test":
            self.x = self.x[valid_split:]
            self.y = self.y[valid_split:]

    def __get_idx__(self, tag2idx, word2idx):
        """
        Convert words (x) into indices for the embedding layer
        PADs sentences to self.max_sentence_length
        Convert tags (y) into indices
        """
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                for k in range(len(self.x[i][j])):
                    w = (
                        self.x[i][j][k]
                        .replace(",", "")
                        .replace(";", "")
                        .replace(":", "")
                        .lower()
                    )
                    if w in word2idx:
                        self.x[i][j][k] = word2idx[w]
                    else:
                        self.x[i][j][k] = 0
                while len(self.x[i][j]) < self.max_sentence_length:
                    self.x[i][j].append(0)

        for i in range(len(self.y)):
            for j in range(len(self.y[i])):
                self.y[i][j] = tag2idx[self.y[i][j]]

    def __pad__(self):
        """
        Pad sequences of sentences to self.max_sequence_length
        Creates MASKS to indicate padded sentences in each sequence
        """
        self.mask = []
        sent_padder = [0 for i in range(self.max_sentence_length)]
        for i in range(len(self.x)):
            temp_mask = [0 for i in range(self.max_sequence_length)]
            for j in range(len(self.x[i])):
                temp_mask[j] = 1
            self.mask.append(temp_mask)
            while len(self.x[i]) < self.max_sequence_length:
                self.x[i].append(sent_padder)
                self.y[i].append(-1)