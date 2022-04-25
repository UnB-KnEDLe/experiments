"""
This module implements a LSTM + CRF to be used in text segmentation tasks.
"""

import torch
from torch import nn
from torchcrf import CRF
from tqdm import tqdm
from tabulate import tabulate
import numpy as np


class LstmCrf(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_tags,
        hidden_dim,
        pretrained_emb,
        idx2tag,
        tipo_ato,
        path,
    ):
        super(LstmCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.idx2tag_dict = idx2tag
        self.ato = tipo_ato
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_emb.vectors)
        )
        self.embed_layer.weight[0] = 0
        self.word_lstm_layer = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1, batch_first=True
        )
        self.sent_lstm_layer = nn.LSTM(
            hidden_dim, hidden_dim // 4, num_layers=1, batch_first=True
        )
        self.linear_layer = nn.Linear(hidden_dim // 4, num_tags)
        self.crf_layer = CRF(num_tags, batch_first=True)

    def forward(self, batch_input, batch_tags, mask_pad):
        """
        Method to compute the forward pass of the LstmCrf model
        Input: (x) shape:        (batch_size) x (sequence_length) x (sentence_length)
               (y) shape:        (batch_size) x (sequence_length)
               (mask_pad) shape: (batch_size) x (sequence_length)
        output: log_likelihood  of the probability of the expected sequence of tags
        """
        batch_size = batch_input.shape[0]
        sequence_pad_size = batch_input.shape[1]
        sentence_pad_size = batch_input.shape[2]

        emb_out = self.embed_layer(batch_input)
        word_lstm_out, (h_n, _) = self.word_lstm_layer(
            emb_out.view(
                batch_size * sequence_pad_size, sentence_pad_size, self.embedding_dim
            )
        )
        sent_lstm_out, (h_n, _) = self.sent_lstm_layer(
            h_n.view(batch_size, sequence_pad_size, self.hidden_dim)
        )
        lin_out = self.linear_layer(
            sent_lstm_out.reshape(batch_size * sequence_pad_size, self.hidden_dim // 4)
        )

        return self.crf_layer(
            lin_out.view(batch_size, sequence_pad_size, self.num_tags), batch_tags
        )

    def fit(self, epoch, train_loader, eval_loader, **kwargs):
        """
        Method to train the LstmCrf model
        Input: (x) shape: (number of batches) x (batch_size) x (sequence_length) x (sentence_length)
               (y) shape: (number of batches) x (batch_size) x (sequence_length)
        output: validation_loss (validation loss over all epochs)
        """
        if "lr" in kwargs:
            learning_rate = kwargs.get("lr")
        else:
            learning_rate = 0.01

        if "weight_decay" in kwargs:
            w_d = kwargs.get("weight_decay")
        else:
            w_d = 1e-4

        validation_loss = []
        min_loss = float("inf")
        #         optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=w_d)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=w_d
        )
        for it in range(epoch):
            print("Epoch:", it)
            for batch in tqdm(train_loader):
                x, mask, y = batch
                x = x.to(self.device)
                mask = mask.to(self.device)
                y = y.to(self.device)

                self.zero_grad()
                loss = -self.forward(x, y, mask)
                loss.backward()
                optimizer.step()

            new_loss = self.evaluate(eval_loader, opt="loss")
            if new_loss < min_loss:
                min_loss = new_loss
                torch.save(self.state_dict(), self.path)
            validation_loss.append(new_loss)
        return validation_loss

    def evaluate(self, valid_loader, opt):
        """
        Method to evaluate trained model on unseen data
        """
        if opt == "loss":
            with torch.no_grad():
                loss = 0
                for batch in valid_loader:
                    x, mask, y = batch
                    x = x.to(self.device)
                    mask = mask.to(self.device)
                    y = y.to(self.device)
                    loss -= self.forward(x, y, mask)
                return loss

        elif opt == "f1":
            true_p = np.array([0, 0, 0])
            false_p = np.array([0, 0, 0])
            false_n = np.array([0, 0, 0])
            with torch.no_grad():
                for batch in tqdm(valid_loader):
                    x, mask, y = batch
                    x = x.to(self.device)
                    mask = mask.to(self.device)
                    y = y.to(self.device)
                    pred = self.predict(x, mask)
                    for i, _ in enumerate(pred):
                        for j, _ in enumerate(pred[i]):
                            if y[i][j] == -1:
                                break
                            if pred[i][j] == y[i][j]:
                                true_p[pred[i][j]] += 1
                            else:
                                false_p[pred[i][j]] += 1
                                false_n[y[i][j]] += 1
            recall = true_p / (true_p + false_n)
            precision = true_p / (true_p + false_p)
            f1_score = 2 * (recall * precision) / (recall + precision)
            print(
                tabulate(
                    [
                        ["Recall", recall[0], recall[1], recall[2]],
                        ["Precision", precision[0], precision[1], precision[2]],
                        ["F1 score", f1_score[0], f1_score[1], f1_score[2]],
                    ],
                    headers=[
                        "",
                        self.idx2tag_dict[0],
                        self.idx2tag_dict[1],
                        self.idx2tag_dict[2],
                    ],
                )
            )
            return recall, precision, f1_score

        else:
            print("Chosen opt doesn't exist")
            return

    def predict(self, batch_input, mask):
        """
        Method to predict segmentation tags
        Input: sequence - shape:(batch_size)x(sequence_size)x(sentence_size)
        output: Predicted tags - shape: (batch_size)x(sequence_size)
        """
        with torch.no_grad():
            batch_size = batch_input.shape[0]
            sequence_pad_size = batch_input.shape[1]
            sentence_pad_size = batch_input.shape[2]

            emb_out = self.embed_layer(batch_input)
            word_lstm_out, (h_n, _) = self.word_lstm_layer(
                emb_out.view(
                    batch_size * sequence_pad_size,
                    sentence_pad_size,
                    self.embedding_dim,
                )
            )
            sent_lstm_out, (h_n, _) = self.sent_lstm_layer(
                h_n.view(batch_size, sequence_pad_size, self.hidden_dim)
            )
            lin_out = self.linear_layer(
                sent_lstm_out.reshape(
                    batch_size * sequence_pad_size, self.hidden_dim // 4
                )
            )
            return self.crf_layer.decode(
                lin_out.view(batch_size, sequence_pad_size, self.num_tags), mask=mask
            )
