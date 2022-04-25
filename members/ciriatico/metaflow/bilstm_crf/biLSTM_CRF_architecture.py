import torch
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from seqeval.metrics import f1_score
from seqeval.scheme import IOB1
from nltk.tokenize import word_tokenize

class biLSTM_CRF(torch.nn.Module):
    def __init__(self, word2idx, tag2idx):
        super(biLSTM_CRF, self).__init__()
        # Camada de embeddings
        self.embedding = torch.nn.Embedding(num_embeddings=len(word2idx), embedding_dim = 50, padding_idx = word2idx['<PAD>'])
        # Camada biLSTM
        self.bilstm = torch.nn.LSTM(input_size=50, hidden_size = 200, num_layers = 2, batch_first = True, bidirectional = True, dropout = 0.25)
        # Camada linear
        self.linear = torch.nn.Linear(400, len(tag2idx))
        # Camada CRF
        self.crf = CRF(num_tags = len(tag2idx), batch_first = True)

    def forward(self, x, y, mask):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = self.linear(x)
        loss = self.crf(x, y, mask=mask)
        return loss

    def decode(self, x, mask):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = self.linear(x)
        prediction = self.crf.decode(x, mask=mask)
        return prediction

    def fit(self, device, X_train, Y_train, X_test, Y_test, X_valid, Y_valid, word2idx, tag2idx, num_epochs, lrate, momentum):
        X_train, Y_train, mask = create_batches(X_train, Y_train, batch_size=32, pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])
        X_valid, _, valid_mask = create_batches(X_valid, Y_valid, batch_size=len(X_valid), pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])
        X_test, _, test_mask   = create_batches(X_test,  Y_test,  batch_size=len(X_test),  pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])
        
        loss = self(X_train[0], Y_train[0], mask[0])
        prediction = self.decode(X_valid[0], valid_mask[0])
        
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        
        f1_history = []
        mean_loss_history = []

        optim = torch.optim.SGD(self.parameters(), lr=lrate, momentum=momentum)

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')

            mean_loss = 0.0
            self.train()
            # Iniciando uma epoch de treinamento supervisionado
            for batch in range(len(X_train)):
                x = X_train[batch].to(device)
                y = Y_train[batch].to(device)
                m = mask[batch].to(device)
                optim.zero_grad()
                loss = -self(x, y, m)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optim.step()
                mean_loss += loss
            mean_loss /= len(X_train)
            mean_loss_history.append(mean_loss)

            self.eval()
            # Calculo do desempenho do modelo treinado nesta epoch
            y_pred = self.decode(X_valid[0].to(device), valid_mask[0].to(device))
            y_pred = IOBify(y_pred, idx2tag)
            y_true = IOBify(Y_valid, idx2tag)
            f1 = f1_score(y_true, y_pred)
            f1_history.append(f1)
            print(f'f1-score: {f1}\n')
            
        dic = {'f1_hist': f1_history, 'mean_loss_history': mean_loss_history}

        return dic

def create_batches(x, y, batch_size, pad_token, pad_class):
    batch_x = []
    batch_y = []
    mask = []

    # Separando os batches pelo tamanho de batch_size
    i = 0
    while i < len(x):
        batch_x.append(x[i:min(len(x), i+batch_size)])
        batch_y.append(y[i:min(len(y), i+batch_size)])
        i += batch_size
    
    # Realizando padding dos batches e criando mask para identificar padding durante o treinamento
    for i in range(len(batch_x)):
        batch_x[i] = pad_sequence(batch_x[i], batch_first = True, padding_value = pad_token)
        batch_y[i] = pad_sequence(batch_y[i], batch_first = True, padding_value = pad_class)
        mask.append(batch_x[i] != pad_token)

    return batch_x, batch_y, mask

def IOBify(tags_sequence, idx2tag):
    if isinstance(tags_sequence[0], list):
        iob_y = [[idx2tag[tag] for tag in tags] for tags in tags_sequence]
    else:
        iob_y = [[idx2tag[tag.item()] for tag in tags] for tags in tags_sequence]
    return iob_y