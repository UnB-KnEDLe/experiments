import torch
from torch.utils.data import Dataset
from math import floor

class segmentation_dataset(Dataset):
    def __init__(self, tag2idx, word2idx, set_type, tipo_ato, path):
        """
        Load the segmentation dataset, (self.__load__())
        Truncate long sentences/split long blocks/ignore very short blocks (self.__trunc__())
        get indices of words and tags (self.__get_idx__())
        PAD the sentences and blocks/create MASKS representing padded elements (self.__pad__())
        """
        self.set_type = set_type
        self.min_block_length = 3
        self.max_sentence_length = 30
        self.max_block_length = 20
        self.ato = tipo_ato
        self.path = path
        
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
        data = open(self.path, 'r').read()
        self.x, self.y = [], []
        for block in data.split("\n\n"):
            x_block, y_block = [], []
            lines = block.split("\n")
            for line in lines:
                line = line.split()
                if line != []:
                    x_block.append(line[1:])
                    y_block.append(line[0])
            self.x.append(x_block)
            self.y.append(y_block)

    def __trunc__(self):
        """
        Truncate sentences with length > max_sentence_length
        Separate blocks with length > max_block_length
        Delete blocks with length < min_block_length
        """
        # For each block in x
        for i in range(len(self.x)):
            # For each sentence in x[i]
            for j in range(len(self.x[i])):
                if len(self.x[i][j]) > self.max_sentence_length:
                    self.x[i][j] = self.x[i][j][:self.max_sentence_length]

        sep_x = []
        sep_y = []
        for i in range(len(self.x)):
            if len(self.x[i]) > self.max_block_length:
                for j in range((len(self.x[i])//self.max_block_length)+1):
                    if self.x[i][j*self.max_block_length:(j+1)*self.max_block_length]:# and 'B' in self.y[i]:
                        sep_x.append(self.x[i][j*self.max_block_length:(j+1)*self.max_block_length])
                        sep_y.append(self.y[i][j*self.max_block_length:(j+1)*self.max_block_length])
            elif len(self.x[i]) < self.min_block_length:
                continue
#             elif 'B' in self.y[i]:
            else:
                sep_x.append(self.x[i])
                sep_y.append(self.y[i])
        self.x = sep_x
        self.y = sep_y
        
    def __split__(self):
        train_split = floor(0.6*len(self.x))
        valid_split = floor(0.1*len(self.x)) + train_split 
        if self.set_type == 'train':
            self.x = self.x[:train_split]
            self.y = self.y[:train_split]
        elif self.set_type == 'valid':
            self.x = self.x[train_split:valid_split]
            self.y = self.y[train_split:valid_split]
        else:
            self.x = self.x[valid_split:]
            self.y = self.y[valid_split:]
        
    def __get_idx__(self, tag2idx, word2idx):
        """
        Convert words (x) into indices for the embedding layer
        PADs sentences to self.max_sentence_length
        Convert tags (y) into indices 
        """
        # get_idx of x
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                for k in range(len(self.x[i][j])):
                    w = self.x[i][j][k].replace(',', '').replace(';', '').replace(':', '').lower()
                    if w in word2idx:
                        self.x[i][j][k] = word2idx[w]
                    else:
                        self.x[i][j][k] = 0
                while(len(self.x[i][j]) < self.max_sentence_length):
                    self.x[i][j].append(0)
                
        # get_idx of y      
        for i in range(len(self.y)):
            for j in range(len(self.y[i])):
                self.y[i][j] = tag2idx[self.y[i][j]]
                
    def __pad__(self):
        """
        Pad blocks of sentences to self.max_block_length
        Creates MASKS to indicate padded sentences in each block
        """
        self.mask = []
        sent_padder = [0 for i in range(self.max_sentence_length)]
        for i in range(len(self.x)):
            temp_mask = [0 for i in range(self.max_block_length)]
            for j in range(len(self.x[i])):
                temp_mask[j] = 1
            self.mask.append(temp_mask)
            while len(self.x[i]) < self.max_block_length:
                self.x[i].append(sent_padder)
                self.y[i].append(-1)
                