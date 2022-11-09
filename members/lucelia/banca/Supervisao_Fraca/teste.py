# CODIGO DAS LABEL FUNCIONS E SUPERVISÃO FRACA

import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin
import os


class LabelFunctionsSuspensao:
    '''
    Classe que armazena as Label Functions para a aplicacao da supervisao fraca em contratos

    Atributos: 
        self.docs = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.vet = vetor auxiliar para detecção de entidades nas Label Functions
    '''

    def __init__(self, dados):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(dados))

    def numero_licitacao_(self, doc):
        
       '''
        label function para extracao do número de licitação usando regex
        
        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
       expression = r"((?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)|(?:CONCORR[EÊ]NCIA|[Cc]oncorr[eê]ncia)|(?:TOMADA|[Tt]omada)\s[Dd][Ee]\s(?:PRE[CÇ]OS?|[Pp]re[cç]os?)|(?:CARTA\s|[Cc]arta\s)?(?:CONVITE|[Cc]onvite)|(?:CONCURSO|[Cc]oncurso)|(?:LEIL[AÃ]O|[Ll]eil[aã]o))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)" 
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                   
                    if grupo.span(1)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]  
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(1)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break          
        
       expression = r"((?:CONCORR[E|Ê]NCIA|[Cc]oncorr[eê]ncia)\s(?:P[ÚU]BLICA|[Pp][úu]blica))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"    
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break
        
       expression = r"((?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)\s(?:[Ss][Rr][Pp]))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"                
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                       
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break        
        #RDC Nº 02/2021 
       expression = r"([Rr][Dd][Cc])[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"                
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break 
                    
       expression = r"((?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)\s(?:[Pp][Oo][Rr]|[Dd][Ee]|\s)\s(?:[Ss][Rr][Pp]))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"                
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break                    
        
       expression = r"((?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)\s(?:[Ii][Nn][Tt][Ee][Rr][Nn][Aa][Cc][Ii][Oo][Nn][Aa][Ll]))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"                
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break
        
       expression = r"(?:(?:DISPENSA\sDE\s|[Dd]ispensa\s[Dd]e\s)|)(?:LICITA[CÇ][AÃ]O|[Ll]icita[cç][aã]o)[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"                     
       match = re.finditer(expression, str(doc))
       if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(1)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:  
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(1)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "numero_licitacao"
                        break
                    
    def modalidade_licitacao_(self, doc):   
        '''
        label function para extracao de modalidade de licitação usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''                 
        modalidade = r"(?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)|(?:CONCORR[EÊ]NCIA|[Cc]oncorr[eê]ncia)|(?:TOMADA|[Tt]omada)\s[Dd][Ee]\s(?:PRE[CÇ]OS?|[Pp]re[cç]os?)|(?:CARTA\s|[Cc]arta\s)?(?:CONVITE|[Cc]onvite)|(?:CONCURSO|[Cc]oncurso)|(?:LEIL[AÃ]O|[Ll]eil[aã]o)"
        match = re.finditer(modalidade, str(doc))
        if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(0)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                        flag = 1
                    if token.idx >= grupo.span(0)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':']):
                            end = doc[token.i-1]
                        else:
                            end = token
                        yield start.i, end.i, "modalidade_licitacao"
                        break 
                      
    def processo_(self, doc):
        '''
        label function para extracao de processo usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[P|p][R|r][O|o][C|c][E|e][S|s][S|s][O|o][\s\S].*?[s|:]?.+?(?=[0-9]).*?(\d*[^;|,|a-zA-Z]*)"    
        match = re.search(expression, str(doc))
        if match:
                    flag = 0
                    for token in doc:
                        if match.span(1)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                            if(doc[token.i].text == ':'):
                                start = doc[token.i+1]
                            else:
                                start = token
                            flag = 1
                        if token.idx >= match.span(1)[1] and flag == 1 and token.i > start.i:
                            if(doc[token.i-1].text in ['.', '-', '—', ':']):
                                end = doc[token.i-1]
                            else:
                                end = token
                            yield start.i, end.i, "processo_gdf"
                            break 
class SkweakSuspensao(LabelFunctionsSuspensao):
    '''
    Classe que aplica as Label Functions e realiza o processo de supervisao fraca
    Para seu funcionamento idel, eh necessario inicializa-da com um dataset de contratos

    Atributos: 
        dados = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.df = dataframe inicial vazio para o armazenamento de entidades, texto do contrato e labels IOB
    '''

    def __init__(self, dados):
        ''' Inicializa o docs e o dataframe '''
        super().__init__(dados)
        self.df = pd.DataFrame(columns=["processo_gdf",  "modalidade_licitacao", "numero_licitacao", "text", "labels"])

    def apply_label_functions(self):
        '''
        Aplica as label functions na base de contratos e extrai as entidades
        '''
        doc = self.docs

        detec_processo = skweak.heuristics.FunctionAnnotator(
            "detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))
        
        detec_modalidade = skweak.heuristics.FunctionAnnotator(
            "detec_modalidade", self.modalidade_licitacao_)
        doc = list(detec_modalidade.pipe(doc))

        detec_numero = skweak.heuristics.FunctionAnnotator(
            "detec_numero", self.numero_licitacao_)
        doc = list(detec_numero.pipe(doc))
   
        self.docs = doc

        

    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''

        
# Data de Assinatura: 12/09/2022 b_data_assinatura

# Data de Assinatura: 12/09/2022 O
# Data de Assinatura: 12/09/2022 b_processo


# Data de Assinatura: 12/09/2022 b_data_assinatura

# 12/09/2022 b_data_assinatura = Sim = b_assinatura 



        model = skweak.aggregation.HMM("hmm", ["processo_gdf",  "modalidade_licitacao", "numero_licitacao"], sequence_labelling=True)

        self.docs = model.fit_and_aggregate(self.docs)

        for doc in self.docs:
            if "hmm" in doc.spans:
                doc.ents = doc.spans["hmm"]
            else:
                doc.ents = []

        ''' Salvando modelo HMM em uma pasta data '''
        if os.path.isdir("./data"):
            skweak.utils.docbin_writer(self.docs, "./data/suspensao.spacy")
        else:
            os.mkdir("./data")
            skweak.utils.docbin_writer(self.docs, "./data/suspensao.spacy")

    def get_IOB(self):
        '''
        retorna os resultados das entidades extraidas em IOB
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/suspensao.spacy")
        examples = []

        for doc in doc_bin.get_docs(nlp.vocab):
            lista_iob = []
            for i in range(0, len(doc)):
                label_iob = ""
                _txt_ = doc[i].text
                _label_ = doc[i].ent_iob_
                _ent_ = doc[i].ent_type_
                if _txt_ not in ["", " ", "  ", "   "]:
                    if(_label_ != "O"):
                        label_iob += f'{_txt_} {_label_}-{_ent_}'
                    else:
                        label_iob += f'{_txt_} {_label_}'
                    lista_iob.append(label_iob)
            examples.append(lista_iob)

        return examples

    def list_spans_specific(self, x):
        '''
        Mostra os spans HMM (entidades finais resultantes da aplicacao do HMM ) de um doc da base de dados especifico

        parametros:
            x: inteiro representano a posicao do doc no vetor de contratos da base de dados
        '''
        print(self.docs[x].spans["hmm"])

    def list_spans_all(self):
        '''
        Mostra os spans HMM (entidades finais resultantes da aplicacao do HMM ) de toda a base de dados
        '''
        for doc in self.docs:
            print(doc.spans["hmm"])

    # Retorna o dataframe com todas as entidades, textos e IOB para cada documento
    def get_hmm_dataframe(self):
        '''
        Retorna o dataframe final com todas as entidades, textos e labels-IOB para cada documento da base de dados
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/suspensao.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"processo_gdf": "", "modalidade_licitacao": "", "numero_licitacao": "", "text": "", "labels": ""}

            for entity in doc.ents:
                aux[entity[0].ent_type_] = entity.text

            for token in doc:
                aux["text"] += token.text + ' '

            for i in range(0, len(doc)):
                _txt_ = doc[i].text
                _label_ = doc[i].ent_iob_
                _ent_ = doc[i].ent_type_
                if _txt_ not in ["", " ", "  ", "   "]:
                    if(_label_ != "O"):
                        aux["labels"] += f'{_label_}-{_ent_} '
                    else:
                        aux["labels"] += f'{_label_} '

            self.df = self.df.append(aux, ignore_index=True)

        return self.df

    def save_dataframe_csv(self, name):
        '''
        Salva o dataframe em um .csv

        parametros:
            name: string representado o nome com o qual deseja salvar o dataframe
        '''
        nome = str(name)+".csv"
        self.df.to_csv(nome)