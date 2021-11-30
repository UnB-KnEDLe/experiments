import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm
from math import sqrt

class char_cnn(nn.Module):
    """
    Convolutional neural network to generate character-level word embeddings as proposed by Ma and Hovy (https://arxiv.org/abs/1603.01354)

    Args:
        embedding_size (integer value): Number (quantity) of character embeddings (i.e. number of unique characters)
        embedding_dim (integer value): size of the character embeddings vector (transform characters into dense vectors)
        out_channels (integer value): Number of channels to be used in the convolutional layer (it is also the size of the output embedding)
    """
    def __init__(self, embedding_size, embedding_dim, out_channels):
        super(char_cnn, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.init_weight()

    def init_weight(self):
        bias = sqrt(3.0/self.embedding.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bias, bias)
        # nn.init.kaiming_uniform_(self.conv.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        shape = x.shape
        x = self.conv(x.reshape([shape[0]*shape[1], shape[2], shape[3]]).permute(0, 2, 1))
        x = torch.nn.functional.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
        x = self.relu(x)
        return x.reshape([shape[0], shape[1], -1])

class word_cnn(nn.Module):
    """
    Word level convolutional neural network module. Implementation following the work from Shen et al (https://arxiv.org/abs/1707.05928)

    Args:
        pretrained_word_emb (): The weights for the pretrained word embeddings
        word2idx (python dictionary): Dictionary that maps words (strings) to indices (integer values)
        full_embedding_size (integer value): Full embedding size (character-level embedding size + pretrained word embedding size)
        conv_layers (integer value): Number of convolutional blocks (blocks defined same as Shen et al - https://arxiv.org/abs/1707.05928)
        out_channels (integer value): Number of channels to be used for the convolutional layers
    """

    def __init__(self, pretrained_word_emb, word2idx, full_embedding_size, conv_layers, out_channels):
        super(word_cnn, self).__init__()
        self.word2idx = word2idx
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_word_emb.vectors), freeze=False)
        # self.conv1 = nn.Conv1d(in_channels=full_embedding_size, out_channels=out_channels, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.5)
        convnet = []
        for i in range(conv_layers):
            if i == 0:
                convnet.append(nn.Conv1d(in_channels=full_embedding_size, out_channels=out_channels, kernel_size = 5, padding = 2))
                convnet.append(nn.ReLU())
                convnet.append(nn.Dropout(p=0.5))
            else:
                convnet.append(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size = 5, padding = 2))
                convnet.append(nn.ReLU())
                convnet.append(nn.Dropout(p=0.5))
        self.convnet = nn.Sequential(*convnet)
        self.init_weight()
        # Flag to control wordDrop inside the model
        self.flag_WordDrop = True

    def init_weight(self):
        pass
        # nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x, char_embeddings):
        # 50% chance word dropout to improve generalization
        if self.training and self.flag_WordDrop:
            mask = torch.distributions.Bernoulli(probs=(1-0.5)).sample(x.size())
            x[~mask.bool()] = self.word2idx['<UNK>']
        # word2vec embedding
        x = self.embedding(x)
        # concat word and char embedding
        x = torch.cat((x, char_embeddings), dim=2)
        # x = self.dropout(x)
        w = x.clone()
        x = x.permute(0, 2, 1)
        x = self.convnet(x)
        x = x.permute(0, 2, 1)
        # Return Concat w_full (word representation - combination of word and char embeddings) and h_enc (output of conv layers)
        return torch.cat((x, w), dim=2)

class decoder(nn.Module):
    """
    LSTM decoder with greedy decoding, as proposed by Shen et al (https://arxiv.org/abs/1707.05928)

    Args:
        feature_size (integer value): Size of the input (input tensor) that is fed to this decoder module
        num_classes (integer value): Number of classes that can be predicted
        hidden_size (integer value): size of the LSTM cell
        decoder_layers (integer value): Number of consecutive LSTM layers 
        device (string): 'cpu' or 'cuda' indicating the device in which the module is currently in
    """
    def __init__(self, feature_size, num_classes, hidden_size, decoder_layers, device):
        super(decoder, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.lstm = torch.nn.LSTM(input_size=feature_size+num_classes, hidden_size=hidden_size, num_layers=decoder_layers)
        self.linear = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0, reduction='sum')
        self.init_weight()

    def init_weight(self):
        # Initialize linear layer
        bias = sqrt(6.0/(self.linear.weight.shape[0]+self.linear.weight.shape[1]))
        nn.init.uniform_(self.linear.weight, -bias, bias)
        nn.init.constant_(self.linear.bias, 0.0)
        # Initialize LSTM layer
        for name, params in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(params, 0.0)
                nn.init.constant_(params[self.lstm.hidden_size:2*self.lstm.hidden_size], 1.0)
            else:
                bias = sqrt(6.0/(params.shape[0]+params.shape[1]))
                nn.init.uniform_(params, -bias, bias)

    def forward_step(self, x, prev_tag, prev_lstm_state):
        prev_tag_onehot = torch.nn.functional.one_hot(prev_tag, num_classes=self.num_classes).to(self.device)
        lstm_input = torch.cat((x, prev_tag_onehot), dim=1).unsqueeze(dim=0)
        lstm_output, lstm_state = self.lstm(lstm_input, prev_lstm_state)
        lstm_output = self.dropout(lstm_output)
        linear_output = self.linear(lstm_output.squeeze(dim=0))
        prediction = torch.argmax(linear_output, dim=1)
        return linear_output, prediction, lstm_state
    
    def forward(self, x, tag):
        batch_size, seq_len, _ = x.size()
        x = x.permute(1, 0, 2)
        tag = tag.permute(1, 0)
        pred = torch.LongTensor([0 for i in range(batch_size)])
        lstm_state = None
        loss = 0.0
        for i in range(seq_len):
            output, pred, lstm_state = self.forward_step(x=x[i], prev_tag=pred, prev_lstm_state=lstm_state)
            loss += self.loss_fn(output, tag[i])
        return loss

    def decode(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.permute(1, 0, 2)
        pred = torch.LongTensor([0 for i in range(batch_size)])
        lstm_state = None
        full_predictions = torch.LongTensor().to(self.device)
        full_output = torch.Tensor().to(self.device)
        for i in range(seq_len):
            output, pred, lstm_state = self.forward_step(x=x[i], prev_tag=pred, prev_lstm_state=lstm_state)
            full_predictions = torch.cat((full_predictions, pred.unsqueeze(dim=0)), dim=0)
            full_output = torch.cat((full_output, output.unsqueeze(dim=0)), dim=0)
        return full_predictions.permute(1, 0), full_output.permute(1, 0, 2)

    def refined_decode(self, x, tag, refined_mask):
        batch_size, seq_len, _ = x.size()
        x = x.permute(1, 0, 2)
        tag = tag.permute(1, 0)
        mask = refined_mask.permute(1, 0)
        pred = torch.LongTensor([0 for i in range(batch_size)]).to(self.device)
        lstm_state = None
        full_predictions = torch.LongTensor().to(self.device)
        full_output = torch.Tensor().to(self.device)
        for i in range(seq_len):
            pred = torch.where(mask[i]==False, tag[i], pred)
            output, pred, lstm_state = self.forward_step(x=x[i], prev_tag=pred, prev_lstm_state=lstm_state)
            full_predictions = torch.cat((full_predictions, pred.unsqueeze(dim=0)), dim=0)
            full_output = torch.cat((full_output, output.unsqueeze(dim=0)), dim=0)
        return full_predictions.permute(1, 0), full_output.permute(1, 0, 2)



class CNN_CNN_LSTM(nn.Module):
    """
    CNN-CNN-LSTM model with greedy decoding for prediction (as proposed in shen et al - https://arxiv.org/abs/1707.05928)

    We added the refinement of predictions, which considers that some of the tokens have reliable (hand annotated) labels during the 
    greedy decoding process
    """
    def __init__(self, char_vocab_size, char_embedding_dim, char_out_channels, word2idx, pretrained_word_emb, word_conv_layers, word_out_channels, num_classes, decoder_layers, decoder_hidden_size, device):
        super(CNN_CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.char_encoder = char_cnn(embedding_size=char_vocab_size, embedding_dim=char_embedding_dim, out_channels=char_out_channels)
        self.word_encoder = word_cnn(pretrained_word_emb=pretrained_word_emb, word2idx=word2idx, conv_layers = word_conv_layers, full_embedding_size=pretrained_word_emb.vector_size+char_out_channels, out_channels=word_out_channels)
        self.decoder      = decoder(num_classes=num_classes, feature_size=pretrained_word_emb.vector_size+char_out_channels+word_out_channels, hidden_size=decoder_hidden_size, decoder_layers=decoder_layers, device=device)
        
    def forward(self, sentence, word, tag, mask):
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        x = self.decoder(x, tag)
        return x
    
    def encode(self, sentence, word, mask):
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        return x

    def decode(self, sentence, word, mask, return_token_probs=False):
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        predictions, output = self.decoder.decode(x)
        
        if return_token_probs == True:
            return predictions, torch.nn.functional.softmax(output, dim=2).max(dim=2).values
        else:
            probability = torch.nn.functional.softmax(output, dim=2).max(dim=2).values
            probability[mask==False] = 1.0
            probability = probability.prod(dim = 1)
            return predictions, probability

    def refined_decode(self, sentence, word, tags, refined_mask):
        """
        Decoding algorithm for prediction refinement, when certain tags are known (annotated by the oracle)
        """
        x = self.char_encoder(word)
        x = self.word_encoder(sentence, x)
        predictions, output = self.decoder.refined_decode(x, tags, refined_mask)
        return predictions
