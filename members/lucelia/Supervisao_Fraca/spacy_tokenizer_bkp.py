def spacy_tokenizer( teste):
        if teste is not None:
            nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
            doc = nlp(teste)
            return[t.text for t in doc]   

import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

def spacy_tokenizer(nlp):

    infix_re = compile_prefix_regex(nlp.Defaults.infixes)
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
nlp.tokenizer = spacy_tokenizer(nlp)