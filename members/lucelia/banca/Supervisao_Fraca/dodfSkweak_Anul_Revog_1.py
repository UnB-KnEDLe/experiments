# CODIGO DAS LABEL FUNCIONS E SUPERVISÃO FRACA

import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin
import os
import pickle



class LabelFunctionsAnulRevog:
    '''
    Classe que armazena as Label Functions para a aplicacao da supervisao fraca em contratos

    Atributos: 
        self.docs = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.vet = vetor auxiliar para detecção de entidades nas Label Functions
    '''

    def __init__(self, dados):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(dados))

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

    def modalidade_(self, doc):
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
                                       
    def numero_licitacao_(self, doc):
        '''
        label function para extracao de modalidade de licitação usando regex parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''

        expression = r"((?:PREG[AÃ]O|[Pp]reg[aã]o)\s(?:ELETR[OÔ]NICO|[Ee]letr[oô]nico)|(?:CONCORR[EÊ]NCIA|[Cc]oncorr[eê]ncia)|(?:TOMADA|[Tt]omada)\s[Dd][Ee]\s(?:PRE[CÇ]OS?|[Pp]re[cç]os?)|(?:CARTA\s|[Cc]arta\s)?(?:CONVITE|[Cc]onvite)|(?:CONCURSO|[Cc]oncurso)|(?:LEIL[AÃ]O|[Ll]eil[aã]o))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"     
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
        
        expression = r"((?:CONCORR[E|Ê]NCIA|[Cc]oncorr[eê]ncia)\s(?:P[ÚU]BLICA|[Pp][úu]blica))[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"   
        match = re.finditer(expression, str(doc))
        if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        #print(token, "1")
                        
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                            #print(token, "2")
                        else:
                            start = token
                            #print(token, "3")
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
    
    def data_assinatura_(self, doc):
            '''
            label function para extracao de data de assinatura usando regex

            parametros:
                doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
            '''
            #expression = r"([A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a].*?[\s\S])(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}\/\d{2}\/\d{4})"
            #expression = r"([A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a].*?[\s\S].*?[s|:]?)(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}\/\d{2}\/\d{4})"
            
            #expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a][\s\S].+?(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
            expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a].*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
            
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
                            yield start.i, end.i, "data_escrito"
                            break  
            
            #expression = r"[B|b][R|r][A|a][S|s][I|i|Í|í][L|l][I|i][A|a][\/][D|d][F|f][\s\S].+?(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
            expression = r"[B|b][R|r][A|a][S|s][I|i|Í|í][L|l][I|i][A|a][\/][D|d][F|f].*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
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
                            yield start.i, end.i, "data_escrito"
                            break                
            
            #expression = r"[D|d][A|a][T|t][A|a][\s\S].+?(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
            expression = r"[D|d][A|a][T|t][A|a][\s\S].*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
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
                            yield start.i, end.i, "data_escrito"
                            break                 
    def processo_detector_fun(self, doc):
        '''
        label function para extracao de processos com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.text in ['Processo', 'PROCESSO', 'Processo:', 'PROCESSO:']:
                if token.i+2 < len(doc):
                    for x in range(1, len(doc)-token.i):
                        if doc[token.i+x].text[0].isdigit() and doc[token.i+x].i < doc[token.i+x].i+1:
                            k = 0
                            if token.i+x+1 < len(doc):
                                if(doc[token.i+x+1].text[0].isdigit()):
                                    k = 1
                            yield doc[token.i+x].i, doc[token.i+x].i+1+k, "processo_gdf"
                            break
                                                
    def numero_licitacao_ml_fun(self, doc):
            #Extrai as features do CRF
            def _get_features(sentence):
                """Create features for each word in act.
                Create a list of dict of words features to be used in the predictor module.
                Args:
                    act (list): List of words in an act.
                Returns:
                    A list with a dictionary of features for each of the words.
                """
                sent_features = []
                
                for i in range(len(sentence)):
                    word_feat = {
                        'word': sentence[i].lower(),
                        'capital_letter': sentence[i][0].isupper(),
                        'all_capital': sentence[i].isupper(),
                        'isdigit': sentence[i].isdigit(),
                        'word_before': sentence[i].lower() if i == 0 else sentence[i-1].lower(),
                        'word_after:': sentence[i].lower() if i+1 >= len(sentence) else sentence[i+1].lower(),
                        'BOS': i == 0,
                        'EOS': i == len(sentence)-1
                    }
                    sent_features.append(word_feat)
                return sent_features
        
            # Carregar modelo
            with open('./crf_modelo_aviso_licitacao.pkl', 'rb') as f:
                model = pickle.load(f)
            
            #print([token.text for token in doc])  # ['Hello', 'world', '!']
            #print(doc.text)  # 'Hello world!'
            #validacao=[token.text for token in doc]
            validacao = []
            
            for token in doc:
                #print("token",token)
                #print("len(doc)",len(doc))
                #print("doc[token].text",token.text)
                validacao = _get_features(token.text)
                #print(validacao)
                #for i in range(len(doc)):
                
                #    validacao[i] = _get_features(doc[token].text)
            #for i in range(len(doc)):
            #    validacao[i] = _get_features(doc[i])
                
            processo_lb = model.predict(validacao)
           # print('processo_lb', processo_lb)
            
            processo_lb_list = []
            for i in processo_lb:
                if not isinstance(i, list):
                    processo_lb_list.append(i)
                else:
                    for j in i:
                        processo_lb_list.append(j)
            
            resultados = []
            comeco_tag = 0
            final_tag = 0
            
            
            #pega a posicao do token de inicio e fim
            for token_list, tag_list in zip(doc, processo_lb_list):
                #print('ML token_list', token_list, tag_list)
                #for (idx_token, token),(idx_tag, tag) in zip(enumerate(token_list), enumerate(tag_list)):
                for (idx_tag, tag) in (enumerate(tag_list)):
                    #print('ML numero de licitacao',token, idx_tag, tag )
                    if tag == 'B-numero_licitacao':
                        #print('ML tag',tag)
                        acumulador = 0
                        comeco_tag = idx_tag
                        for tag_seguinte in tag_list[idx_tag+1:]:
                            if tag_seguinte == 'I-numero_licitacao':
                                acumulador+=1        
                            else:
                                break
                        #print(acumulador)
                        final_tag = comeco_tag + acumulador
                        #print('label ML',comeco_tag, final_tag, 'numero_licitacao')
                        yield comeco_tag, final_tag, 'numero_licitacao'
                        break                      
    
    def valor_(self, doc):
        '''
        label function para extracao de valor usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[V|v][a|A][l|L][o|O][r|R].*?[\s\S][D|d][O|o].*?[\s\S][C|c][O|o][N|n][V|v][E|e][N|n][I|i][O|o].*?[\s\S].*?([\d\.]*,\d{2})"
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
                        yield start.i, end.i, "valor_contrato"
                        break
class SkweakAnulRevog(LabelFunctionsAnulRevog):
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
        self.df = pd.DataFrame(columns=["processo_gdf", "modalidade_licitacao", "numero_licitacao", "data_escrito", "text", "labels"])

    def apply_label_functions(self):
        '''
        Aplica as label functions na base de contratos e extrai as entidades
        '''
        doc = self.docs
        
        ''' 
        Aplica a label function para extracao do processo do ato de anulação e revogação de licitação 
        '''                    
        detec_processo = skweak.heuristics.FunctionAnnotator("detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))

        '''
            Aplica a label function para extracao de modalidade de licitação do ato de anulação e revogação de licitação 
        '''                     
        detec_modalidade = skweak.heuristics.FunctionAnnotator("detec_modalidade", self.modalidade_)
        doc = list(detec_modalidade.pipe(doc))

        processo_detector = skweak.heuristics.FunctionAnnotator("processo_detector", self.processo_detector_fun)
        doc = list(processo_detector.pipe(doc))
        
        detec_valor = skweak.heuristics.FunctionAnnotator("detec_valor", self.valor_)
        doc = list(detec_valor.pipe(doc))
        
        '''
            Aplica a label function para extracao da data de assintura do ato de anulação e revogação de licitação 
        '''  
        detec_data = skweak.heuristics.FunctionAnnotator("detec_data", self.data_assinatura_)
        doc = list(detec_data.pipe(doc))
        self.docs = doc

        
        '''
            Aplica a label function para extracao do número de licitação do ato de anulação e revogação de licitação  
        '''                     
        detec_numero_licitacao = skweak.heuristics.FunctionAnnotator("detec_numero_licitacao", self.numero_licitacao_)
        doc = list(detec_numero_licitacao.pipe(doc))   
               
        ##ML
        detec_numero_licitacao_ml = skweak.heuristics.FunctionAnnotator(
            "detec_numero_licitacao_ml", self.numero_licitacao_ml_fun)
        doc = list(detec_numero_licitacao_ml.pipe(doc))
     

        

    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''
        model = skweak.aggregation.HMM("hmm", ["processo_gdf", "modalidade_licitacao", "numero_licitacao", "data_escrito", "valor_contrato"], sequence_labelling=True)

        self.docs = model.fit_and_aggregate(self.docs)
        #print(self.docs)
        for doc in self.docs:
            #print(f"\n\ndoc.spans \n{doc.spans}" )
            if "hmm" in doc.spans:
                doc.ents = doc.spans["hmm"]
               # print(f"\n\ndoc.ents \n{doc.ents}" )
                
            else:
                doc.ents = []

        ''' Salvando modelo HMM em uma pasta data '''
        if os.path.isdir("./data"):
            skweak.utils.docbin_writer(self.docs, "./data/anul_revog.spacy")
        else:
            os.mkdir("./data")
            skweak.utils.docbin_writer(self.docs, "./data/anul_revog.spacy")

    def get_IOB(self):
        '''
        retorna os resultados das entidades extraidas em IOB
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/anul_revog.spacy")
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
        doc_bin = DocBin().from_disk("./data/anul_revog.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"processo_gdf": "", "modalidade_licitacao": "", "numero_licitacao": "","data_escrito": "", "valor_contrato": "","text": "", "labels": ""}
            

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
