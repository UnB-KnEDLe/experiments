import os
import re
from itertools import zip_longest
import nltk
import spacy

_reg = re.compile(r'(?!\d\s)([.])\s+(?=[A-Z])')

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

nlp = spacy.load('pt_core_news_sm')
sentencizer = nlp.create_pipe('sentencizer')
nlp.add_pipe(sentencizer, before='parser')

def sentencize_dodf(s, backend='regex'):
    if backend == 'regex':    
        sents = re.split(_reg, s)
        # return [i[0] + (i[1] if not i[0].endswith('.') else '.') for i in
        return [i[0] + i[1] for i in
            zip_longest(sents[0::2], sents[1::2], fillvalue='.')]
    elif backend == 'nltk':
        return sent_tokenizer.tokenize(s)
    elif backend == 'spacy':
        return list(nlp(s).sents)
    else:
        raise ValueError(f"`backend` must be one of {{'regex', 'nltk', 'spacy'}}")

def spaced_letters_fix(s):
    mts = re.split(r'((?:[A-ZÀ-Ž]{1,2}\s){3,})', s)
    offset = 0
    lis = [ s[:offset] ]
    for text, spaced in zip_longest(mts[0::2], mts[1::2], fillvalue=''):
        lis.append(text)
        lis.append(spaced.replace(' ', ''))
        lis.append(' ')
    lis.pop()    # last space is extra
    return ''.join(lis)



def drop_parenthesis(s):
    lis = re.split(r'([()])', s)
    ext_lis = lis + ['(']
    acc = 0
    new = []
    for tex, par in zip(ext_lis[0::2], ext_lis[1::2]):
        if tex:
            if not acc:
                new.append(tex)
        acc += (1 if par == '(' else -1)
    return ''.join(new)


def preprocess(s):
    s = drop_parenthesis(s)
    s = spaced_letters_fix(s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s


