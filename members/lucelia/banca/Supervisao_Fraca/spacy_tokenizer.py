from mimetypes import init
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokens import DocBin

class spacy_tokenizer():
    def __init__(self):
        #self.nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.nlp = spacy.load('pt_core_news_sm', disable = ['parser','ner'])
        

    def tokenize(self, texto):
                
        #nlp.tokenizer = self.custom_tokenizer(nlp)
        doc = self.nlp(texto)
        return[t.text.strip() for t in doc if t.text.strip()]
        #return[t.text for t in doc]   
        
        #return[t.text.strip() for t in doc if t.text.strip() !=""]
        #return[t.text.strip() for t in doc if t.text.strip() not in ["\n", " "]]