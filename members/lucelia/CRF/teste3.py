from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from nltk.corpus.reader import ConllCorpusReader

nltk.download('averaged_perceptron_tagger')

nltk.download('conll2002')
nltk.corpus.conll2002.fileids()

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))



train = ConllCorpusReader('CoNLL-2003', 'eng.train', ['words', 'pos', 'ignore', 'chunk'])
test = ConllCorpusReader('CoNLL-2003', 'eng.testa', ['words', 'pos', 'ignore', 'chunk'])


""" with open('/home/82068895153/POS/skweak/data/conll2003_dataset/train.txt', 'r') as file:
  train_sents = list(file.readlines())
  train_sents = ''.join(train_sents)
  train_sents = train_sents.split("\n\n")
  train_sents = [s.splitlines() for s in train_sents]

train_sents_t= [tuple(s.split()) for s in train_sents]

print (train_sents_t) """

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

with open('/home/82068895153/POS/skweak/data/conll2003_dataset/train_out_2.txt', 'r') as file:
  sentences = list(file.readlines())
  #sentences = (file.readlines())

clean_sentences = [word.split() for word in sentences if word]

#tagged_sentences = nltk.pos_tag(clean_sentences)
tagged_sentences = [nltk.pos_tag(sent) for sent in clean_sentences]



#sentences = ["Peter  Blackburn      BRUSSELS  1996-08-22      The  European  Commission  said  on  Thursday  it  disagreed  with  German  advice  to  consumers  to  shun  British  lamb  until  scientists  determine  whether  mad  cow  disease  can  be  transmitted  to  sheep "]
#tokeniza a sentenca
#sentences_tokenized = [s.split() for s in tagged_sentences]
#lista de palavras em features
X_test =  [sent2features(s) for s in tagged_sentences]
labels = crf.predict(X_test)
""" 
print = [list(zip(t[0],t[1])) for t in zip(tagged_sentences, labels)]

ner = [list(zip(t[0],t[1])) for t in zip(tagged_sentences, labels)]

Str_ner = "".join(map(str, ner))


# Write the file out again
with open('/home/82068895153/POS/skweak/data/conll2003_dataset/train_ner.txt', 'wt') as fileout:
  fileout.write(Str_ner) """