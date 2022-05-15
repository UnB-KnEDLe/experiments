print('aqui')
train_path = 'avi_lic_train.txt'
f = open(train_path, mode='r', encoding='utf-8').read().splitlines()


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


train_x, train_y = preprocess(open('avi_lic_train.txt', mode='r', encoding='utf-8'))
valid_x, valid_y = preprocess(open('avi_lic_testa.txt', mode='r', encoding='utf-8'))
test_x, test_y = preprocess(open('avi_lic_testb.txt', mode='r', encoding='utf-8'))


from collections import OrderedDict


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


word2idx = word_dict(train_x)
tag2idx  = tag_dict(train_y)


def numericalize(sentences, word2idx, tag_sentences, tag2idx):
    numericalized_sentences = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in sentences]
    numericalized_tags = [[tag2idx[tag] for tag in tags] for tags in tag_sentences]
    return numericalized_sentences, numericalized_tags


train_x, train_y = numericalize(train_x, word2idx, train_y, tag2idx)
valid_x, valid_y = numericalize(valid_x, word2idx, valid_y, tag2idx)
test_x, test_y = numericalize(test_x, word2idx, test_y, tag2idx)


import itertools
import torch
def add_special_tokens(sentences, word2idx, tag_sentences, tag2idx):
    formatted_sentences = [torch.LongTensor([word for word in itertools.chain([word2idx['<BOS>']], sentence, [word2idx['<EOS>']])]) for sentence in sentences]
    formatted_tags = [torch.LongTensor([tag for tag in itertools.chain([tag2idx['O']], tags, [tag2idx['O']])]) for tags in tag_sentences]
    return formatted_sentences, formatted_tags


train_x, train_y = add_special_tokens(train_x, word2idx, train_y, tag2idx)
valid_x, valid_y = add_special_tokens(valid_x, word2idx, valid_y, tag2idx)
test_x, test_y   = add_special_tokens(test_x, word2idx, test_y, tag2idx)


import numpy as np

# Ordenando sentencas por tamanho (antes de criar os batches)
ordered_idx = np.argsort([len(train_x[i]) for i in range(len(train_x))])
train_x = [train_x[idx] for idx in ordered_idx]
train_y = [train_y[idx] for idx in ordered_idx]


from torch.nn.utils.rnn import pad_sequence

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

train_x, train_y, mask = create_batches(train_x, train_y, batch_size=32, pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])
valid_x, _, valid_mask = create_batches(valid_x, valid_y, batch_size=len(valid_x), pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])
test_x, _, test_mask   = create_batches(test_x,  test_y,  batch_size=len(test_x),  pad_token=word2idx['<PAD>'], pad_class=tag2idx['<PAD>'])


from torchcrf import CRF


class bilstm_crf(torch.nn.Module):
    def __init__(self, word2idx, tag2idx):
        super(bilstm_crf, self).__init__()
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


model = bilstm_crf(word2idx = word2idx, tag2idx = tag2idx)
loss = model(train_x[0], train_y[0], mask[0])
prediction = model.decode(valid_x[0], valid_mask[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


from seqeval.metrics import f1_score
from seqeval.scheme import IOB1


idx2tag = {idx: tag for tag, idx in tag2idx.items()}

def IOBify(tags_sequence, idx2tag):
    if isinstance(tags_sequence[0], list):
        iob_y = [[idx2tag[tag] for tag in tags] for tags in tags_sequence]
    else:
        iob_y = [[idx2tag[tag.item()] for tag in tags] for tags in tags_sequence]
    return iob_y

y_true = IOBify(valid_y, idx2tag)
y_pred = IOBify(prediction, idx2tag)


lrate = 0.015
optim = torch.optim.SGD(model.parameters(), lr=lrate, momentum=0.9)


f1_history = []
mean_loss_history = []


# Alterar numero de epocas de treinamento (~30-50 epocas para modelo bem treinado)
for epoch in range(3):
    mean_loss = 0.0
    model.train()
    # Iniciando uma epoch de treinamento supervisionado
    for batch in range(len(train_x)):
        x = train_x[batch].to(device)
        y = train_y[batch].to(device)
        m = mask[batch].to(device)
        optim.zero_grad()
        loss = -model(x, y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        mean_loss += loss
    mean_loss /= len(train_x)
    mean_loss_history.append(mean_loss)

    model.eval()
    # Calculo do desempenho do modelo treinado nesta epoch
    y_pred = model.decode(valid_x[0].to(device), valid_mask[0].to(device))
    y_pred = IOBify(y_pred, idx2tag)
    y_true = IOBify(valid_y, idx2tag)
    f1 = f1_score(y_true, y_pred)
    f1_history.append(f1)

    print(f'Epoch: {epoch} | Loss media: {mean_loss} | f1-score: {f1}')

mean_loss_history_ = [mean_loss.item() for mean_loss in mean_loss_history]

with open('model_eval.txt', mode='w', encoding='utf-8') as arquivo:
        arquivo.write(f"f1_history: {f1_history}\n\nmean_loss_history_: {mean_loss_history_}\n\n")


torch.save(model.state_dict(), 'inference_ner_model.pt')
torch.save(model, 'entire_ner_model.pt')
