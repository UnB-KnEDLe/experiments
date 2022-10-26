from itertools import chain

import os, nltk, re
import sklearn
import scipy.stats
from nltk.corpus import stopwords
#from unidecode import unidecode
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

nltk.download('averaged_perceptron_tagger')

nltk.download('conll2002')
nltk.corpus.conll2002.fileids()

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

def word2features(sent, i):
    word = sent[i][0]
    #print (word)
    postag = sent[i][1]
    #print (postag)

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)


#sentences = ["La Coruña is good place to live", "Madrid is good too" ]


with open('/home/82068895153/POS/skweak//data/conll2003_dataset/train_out_1.txt', 'r') as file: 
   #sentences = [list(file.readlines())]
    sentences = (file.readlines())

tokenized_raw_data = 'n'.join(nltk.line_tokenize(sentences))

def function1():
    tokens_sentences = sent_tokenize(tokenized_raw_data.lower())
    unfiltered_tokens = [[word for word in word_tokenize(word)] for word in tokens_sentences]
    tagged_tokens = nltk.pos_tag(unfiltered_tokens)
    nouns = [word.encode('utf-8') for word,pos in tagged_tokens
    return nouns

word_list= for i in range(len(unfiltered_tokens)):
    word_list.append() 
for i in range(len(unfiltered_tokens)):
    for word in unfiltered_tokens[i]:
        if word[1:].isalpha():
            word_list[i].append(word[1:])

tagged_tokens= for token in word_list:
    tagged_tokens.append(nltk.pos_tag(token))

#clean_sentences = [word for word in sentences if word]
#tagged_sentences = nltk.pos_tag(clean_sentences)


#sentences = ["Peter  Blackburn      BRUSSELS  1996-08-22      The  European  Commission  said  on  Thursday  it  disagreed  with  German  advice  to  consumers  to  shun  British  lamb  until  scientists  determine  whether  mad  cow  disease  can  be  transmitted  to  sheep "]
#tokeniza a sentenca
#sentences_tokenized = [ s.split() for s in tagged_sentences]
#lista de palavras em features
X_test =  [sent2features(s) for s in sentences_tokenized]
labels = crf.predict(X_test)

print([list(zip(t[0],t[1])) for t in zip(sentences_tokenized, labels)])