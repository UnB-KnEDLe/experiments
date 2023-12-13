# CODIGO DAS LABEL FUNCIONS E SUPERVISÃO FRACA

import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin
import os
import pickle
import spacy
from spacy_tokenizer import spacy_tokenizer


class LabelFunctionsAditamento:
    '''
    Classe que armazena as Label Functions para a aplicacao da supervisao fraca em contratos

    Atributos: 
        self.docs = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.vet = vetor auxiliar para detecção de entidades nas Label Functions
    '''

    def __init__(self, dados):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(dados))
        self.spacy_tokenizer = spacy_tokenizer()

    def contrato_(self, doc):
        '''
        label function para extracao de contratos usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        
        expression = r"([C|c][O|o][N|n|][T|t][R|r][A|a][T|t][O|o][\s\S].+?(?=[0-9]).*?|[C|c][O|o][N|n|][V|v][E|e|Ê|ê][N|n][I|i][O|o][\s\S].*?[s|:]?.+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*?)"
        
        match = re.search(expression, str(doc))
        #print("Teste 4")
        if match:
            flag = 0
            for token in doc:
                if match.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx >= match.span(2)[1] and flag == 1 and token.i > start.i:
                    if(doc[token.i-1].text in ['.', '-', '—', ':','" "']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    
                    yield start.i, end.i, "numero_contrato"
                    break
        expression = r"([C|c][O|o][N|n|][T|t][R|r][A|a][T|t][O|o][\s\S][N|n][\s|o|º|.].+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*?)"
        
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if match.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx >= match.span(2)[1] and flag == 1 and token.i > start.i:
                    if(doc[token.i-1].text in ['.', '-', '—', ':','" "']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    
                    yield start.i, end.i, "numero_contrato"
                    break        
        
        expression = r"([A|a][P|p][O|o|][I|i][O|o][\s\S][F|f][I|i][N|n][A|a][N|n][C|c][E|e][I|i][R|r][O|o][\s\S][A|a][O|o][\s|s][P|p][R|r][O|o][J|j][E|e][T|t][O|o][\s\S][N|n][\s|o|º|.].+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*?)"
        
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if match.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx >= match.span(2)[1] and flag == 1 and token.i > start.i:
                    if(doc[token.i-1].text in ['.', '-', '—', ':','" "']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    
                    yield start.i, end.i, "numero_contrato"
                    break       

        expression = r"([T|t][E|e][R|r|][M|m][O|o][\s\S][D|d][E|e][\s\S][O|o][U|u][T|t][O|o][R|r][G|g][A|a][\s\S][E|e][\s\S][A|a][C|c][E|e][I|i][T|t][A|a][C|c|Ç|ç][A|a|Ã|ã][O|o][\s\S][N|n][\s|o|º|.].+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*?)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if match.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx >= match.span(2)[1] and flag == 1 and token.i > start.i:
                    if(doc[token.i-1].text in ['.', '-', '—', ':','" "']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    
                    yield start.i, end.i, "numero_contrato"
                    break 
                
        expression = r"([T|t][E|e][R|r|][M|m][O|o][\s\S][D|d][E|e][\s\S][A|a][J|j][U|u][S|s][T|t][E|e][\s\S][N|n][\s|o|º|.|.o].+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*?)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if match.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx >= match.span(2)[1] and flag == 1 and token.i > start.i:
                    if(doc[token.i-1].text in ['.', '-', '—', ':','" "']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    
                    yield start.i, end.i, "numero_contrato"
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
                
    def data_assinatura_(self, doc):
        '''
        label function para extracao de data de assinatura usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a][\s\S].+?(\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"

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
    def contrato_detector_fun(self, doc):
            '''
            label function para extracao de contrato com comparacoes de listas

            parametros:
                doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
            '''
            flag = 0
            for token in doc:
                #print("1 - token.text ", token.text)
                if token.text in ['CONTRATO', 'Contrato', 'CONVÊNIO', 'CONVENIO', 'Convênio', 'Convenio']:
                    if token.i+2 < len(doc):
                        
                        for x in range(1, len(doc)-token.i):
                            if doc[token.i+x].text[0].isdigit() and doc[token.i+x].i < doc[token.i+x].i+1:
                                k = 0
                                if token.i+x+1 < len(doc):
                                    if(doc[token.i+x+1].text[0].isdigit()):
                                        k = 1
                                    #print("2 - start token", doc[token.i+x].i)
                                    #print("3 - end token", doc[token.i+x].i+1+k)
                                    yield doc[token.i+x].i, doc[token.i+x].i+1+k, "numero_contrato",
                                    flag = 1
                                    break
                        if flag == 1:
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
                            #print("Teste Matheus", "inicio", doc[token.i+x].i, "fim", doc[token.i+x].i+1+k )
                            yield doc[token.i+x].i, doc[token.i+x].i+1+k, "processo_gdf"
                            break  
                  
    def processo_ml_fun(self, doc):
        
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
        with open('./crf_modelo_extrato_contrato.pkl', 'rb') as f:          
            model = pickle.load(f)
        
        #tokenização tratar entrada do modelo
        spacy = spacy_tokenizer()
        tokens = []
        
        #doc é uma série pandas (indice+valor)
        for documento in doc:
            print("entrada----", documento)
            #documento é uma string
            tokens_documento = spacy.tokenize(documento) #Lista de tokens
            tokens.append(tokens_documento) #Lista de lista de tokens
            '''[['EXTRATO',
                'DO',
                'CONTRATO',
                'Nº',
                '21/2021',
                'Processo',
                ':',
                '00401-00008404/',
                '2021-71',
                '.','''
        
        tokens_validacao = tokens.copy()
        
        for i in range(len(tokens)):
            tokens[i] = _get_features(tokens[i])
            
        processo_lb = model.predict(tokens)
        #Apenas para validar o token mais ›a tag extraida do modelo
        #resultados = []
        #for token_list, tag_list in zip(tokens_validacao, processo_lb):
        #    for token, tag in zip(token_list, tag_list):
        #        resultados.append((token, tag))
        #return resultados
        
        resultados = []
        comeco_tag = 0
        final_tag = 0
        
        #Apenas para validar o token mais a tag extraida do modelo
        for token_list, tag_list in zip(tokens_validacao, processo_lb):
            for (idx_token, token),(idx_tag, tag) in zip(enumerate(token_list), enumerate(tag_list)):
                print(token, tag, idx_tag)
                if tag == 'B-processo_gdf':
                    #print(tag)
                    acumulador = 0
                    comeco_tag = idx_tag
                    for tag_seguinte in tag_list[idx_tag+1:]:
                        if tag_seguinte == 'I-processo_gdf':
                            acumulador+=1        
                        else:
                            break
                    #print(acumulador)
                    final_tag = comeco_tag + acumulador
                    yield comeco_tag, final_tag, "processo_gdf"
                    break   
                    #print(tag, comeco_tag, final_tag, 'processo_gdf' )
                    
        


                            
    def data_detector_fun(self, doc):
        '''
        label function para extracao de data de assinatura com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+5 < len(doc):
                for y in ['DATA', 'Data', 'data', 'Da', 'DA']:
                    if y in token.text and doc[token.i+1].text in ['Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'assinatura'] and doc[token.i+2].text in ['Do', 'DO', 'do'] and doc[token.i+3].text in ['CONTRATO', 'Contrato', 'contrato', 'Contrato:', 'contrato:', 'CONVÊNIO', 'CONVENIO', 'Convênio', 'Convenio', 'convênio', 'convenio', 'Convênio:', 'Convenio:', 'convênio:', 'convenio:']:
                        k = 0
                        if(doc[token.i+4].text == ':'):
                            k += 1
                        for x in range(4, len(doc)-token.i):
                            if (doc[token.i+x].text in ['.', ',', ';'] or (doc[token.i+x].text in ['Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK', 'EOB']) or ((doc[token.i+x].i+1 < len(doc)) and (doc[token.i+x].text in ['Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao:',  'Unidade', 'UNIDADE'] and doc[token.i+x+1].text in ['Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'])) or ((doc[token.i+x].i+2 < len(doc)) and (doc[token.i+x].text in ['Programa', 'PROGRAMA', 'Natureza', 'NATUREZA', 'Data', 'DATA'] and doc[token.i+x+1].text in ['de', 'do', 'da', 'DE', 'DO', 'DA'] and doc[token.i+x+2].text in ['trabalho', 'Trabalho', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:', 'despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAS', 'despesas:', 'Despesas:', 'DESPESAS:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']))) and token.i+4+k < token.i+x:
                                yield token.i+4+k, token.i+x, "data_escrito"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+4+k < token.i+x+1:
                                yield token.i+4+k, token.i+x+1, "data_escrito"
                                break
            if token.i+4 < len(doc):
                for y in ['DATA', 'Data', 'data']:
                    if y in token.text and doc[token.i+1].text in ['de', 'da', 'De', 'Da', 'DE', 'DA'] and doc[token.i+2].text in ['Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']:
                        k = 0
                        if(doc[token.i+3].text == ':'):
                            k += 1
                        for x in range(3, len(doc)-token.i):
                            if (doc[token.i+x].text in ['.', ',', ';'] or (doc[token.i+x].text in ['Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK', 'EOB']) or ((doc[token.i+x].i+1 < len(doc)) and (doc[token.i+x].text in ['Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao:',  'Unidade', 'UNIDADE'] and doc[token.i+x+1].text in ['Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'])) or ((doc[token.i+x].i+2 < len(doc)) and (doc[token.i+x].text in ['Programa', 'PROGRAMA', 'Natureza', 'NATUREZA', 'Data', 'DATA'] and doc[token.i+x+1].text in ['de', 'do', 'da', 'DE', 'DO', 'DA'] and doc[token.i+x+2].text in ['trabalho', 'Trabalho', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:', 'despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAS', 'despesas:', 'Despesas:', 'DESPESAS:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']))) and token.i+3+k < token.i+x:
                                yield token.i+3+k, token.i+x, "data_escrito"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+3+k < token.i+x+1:
                                yield token.i+3+k, token.i+x+1, "data_escrito"
                                break
            if token.i+2 < len(doc):
                for y in ['Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']:
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k += 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in ['.', ',', ';'] or (doc[token.i+x].text in ['Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK', 'EOB']) or ((doc[token.i+x].i+1 < len(doc)) and (doc[token.i+x].text in ['Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao:',  'Unidade', 'UNIDADE'] and doc[token.i+x+1].text in ['Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'])) or ((doc[token.i+x].i+2 < len(doc)) and (doc[token.i+x].text in ['Programa', 'PROGRAMA', 'Natureza', 'NATUREZA', 'Data', 'DATA'] and doc[token.i+x+1].text in ['de', 'do', 'da', 'DE', 'DO', 'DA'] and doc[token.i+x+2].text in ['trabalho', 'Trabalho', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:', 'despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAS', 'despesas:', 'Despesas:', 'DESPESAS:', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']))) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "data_escrito"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1, "data_escrito"
                                break


class SkweakAditamento(LabelFunctionsAditamento):
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
        self.df = pd.DataFrame(columns=["numero_contrato", "processo_gdf", "data_escrito", "text", "labels"])

    def apply_label_functions(self):
        '''
        Aplica as label functions na base de contratos e extrai as entidades
        '''
        doc = self.docs

        detec_contrato = skweak.heuristics.FunctionAnnotator(
            "detec_contrato", self.contrato_)
        doc = list(detec_contrato.pipe(doc))
        
        #LB Regex
        detec_processo = skweak.heuristics.FunctionAnnotator(
            "detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))
        
        #Regex data
        detec_data = skweak.heuristics.FunctionAnnotator(
            "detec_data", self.data_assinatura_)
        doc = list(detec_data.pipe(doc))

        contrato_detector = skweak.heuristics.FunctionAnnotator(
            "contrato_detector", self.contrato_detector_fun)
        doc = list(contrato_detector.pipe(doc))

        #LB de posição
        data_detector = skweak.heuristics.FunctionAnnotator(
            "data_detector", self.data_detector_fun)
        doc = list(data_detector.pipe(doc))
        
        processo_detector = skweak.heuristics.FunctionAnnotator(
            "processo_detector", self.processo_detector_fun)
        doc = list(processo_detector.pipe(doc))

        ##LB ML
        detec_processo_ml = skweak.heuristics.FunctionAnnotator(
            "detec_processo_ml", self.processo_ml_fun)
        doc = list(detec_processo_ml.pipe(doc))
        
        self.docs = doc

        

    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''
        model = skweak.aggregation.HMM("hmm", ["numero_contrato", "processo_gdf", "data_escrito"], sequence_labelling=True)

        self.docs = model.fit_and_aggregate(self.docs)
        #print(self.docs)
        for doc in self.docs:
            #print(f"\n\ndoc.spans \n{doc.spans}" )
            if "hmm" in doc.spans:
                doc.ents = doc.spans["hmm"]
                #print(f"\n\ndoc.ents \n{doc.ents}" )
                
            else:
                doc.ents = []

        ''' Salvando modelo HMM em uma pasta data '''
        if os.path.isdir("./data"):
            skweak.utils.docbin_writer(self.docs, "./data/aditamento.spacy")
        else:
            os.mkdir("./data")
            skweak.utils.docbin_writer(self.docs, "./data/aditamento.spacy")

    def get_IOB(self):
        '''
        retorna os resultados das entidades extraidas em IOB
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/aditamento.spacy")
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
        doc_bin = DocBin().from_disk("./data/aditamento.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"numero_contrato": "", "processo_gdf": "", "data_escrito": "", "text": "", "labels": ""}

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
