# CODIGO DAS LABEL FUNCIONS E SUPERVISÃO FRACA

import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin
import os


class LabelFunctionsConvenio:
    '''
    Classe que armazena as Label Functions para a aplicacao da supervisao fraca em contratos

    Atributos: 
        self.docs = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.vet = vetor auxiliar para detecção de entidades nas Label Functions
    '''

    def __init__(self, dados):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(dados))

    def convenio_(self, doc):
        '''
        label function para extracao de convenio usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''

        expression = r"([C|c][O|o][N|n|][V|v][E|e|Ê|ê][N|n][I|i][O|o])[\s\S][N|n][\s|o|º|.].+?(?=[0-9].*?)(\d*[^;|,|a-zA-Z]*?)"
        
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
                    if(doc[token.i-1].text in ['.', '-', '—', ':']):
                        end = doc[token.i-1]
                    else:
                        end = token
                    yield start.i, end.i, "numero_convenio"
                    break

    def convenio_detector_fun(self, doc):
        '''
        label function para extracao de contrato com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        flag = 0
        for token in doc:
            #print("1 - token.text ", token.text)
            if token.text in ['CONVÊNIO', 'CONVENIO', 'Convênio', 'Convenio']:
                if token.i+2 < len(doc):
                    
                    for x in range(1, len(doc)-token.i):
                        if doc[token.i+x].text[0].isdigit() and doc[token.i+x].i < doc[token.i+x].i+1:
                            k = 0
                            if token.i+x+1 < len(doc):
                                if(doc[token.i+x+1].text[0].isdigit()):
                                    k = 1
                                yield doc[token.i+x].i, doc[token.i+x].i+1+k, "numero_convenio",
                                flag = 1
                                break
                    if flag == 1:
                        break

    def processo_(self, doc):
        '''
        label function para extracao de processo usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[P|p][R|r][O|o][C|c][E|e][S|s][S|s][O|o].*?[\s\S|:]?.+?(?=[0-9]).*?(\d*[^;|,|a-zA-Z]*)"    
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
                        yield start.i, end.i, "data_assinatura_convenio"
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
                        yield start.i, end.i, "valor_convenio"
                        break
        expression = r"[V|v][a|A][l|L][o|O][r|R].*?[\s\S].*?([\d\.]*,\d{2})"
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
                        yield start.i, end.i, "valor_convenio"
                        break

    def unidade_orcamento_(self, doc):
        """ '''
        label function para extracao de unidade orcamentaria usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        """
        expression = r"[u|U][n|N][i|I][d|D][a|A][d|D][e|E][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)"
        match = re.finditer(expression, str(doc))
       
        if match:
            for grupo in match:
                #print ("I - grupo Unidade Orcamentária",grupo)
                flag = 0
                for token in doc:
                    #print ("token",token)
                    if grupo.span(1)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        #print("1 - Unidade Orcamentaria doc[token.i].text ", doc[token.i].text)
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                            #print("2 - Orcamentaria - start token", start)
                        flag = 1
                    if token.idx >= grupo.span(1)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':', '/', ',']):
                            end = doc[token.i-1]
                            #print("3 - Unidade Orcamentaria end token", end)
                        else:
                            end = token
                            
                        yield start.i, end.i, "unidade_orcamentaria"
                        #print("4 - Unidade Orcamentaria start e end", start.i, end.i)
                        break
        
        expression_x = r"([U][.][O].*?[\s\S].*?|[U][O][\s|:].*?[\s\S].*?)(\d+.\d+)"
        match_1 = re.finditer(expression_x, str(doc))
        if match_1:
            for grupo in match_1:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        #print("1 - doc[token.i].text", doc[token.i].text)
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                            #print("2 - start token", start)
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':', '/', ',']):
                            end = doc[token.i-1]
                        else:
                            end = token
                            #print("3.1 - end token", end)
                        yield start.i, end.i, "unidade_orcamentaria"
                        #print("4 - start e end", start.i, end.i)
                        break

    def programa_trabalho_(self, doc):
        '''
        label function para extracao de programa de trabalho usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        #Programa de Tr a b a l h o :
        expression = r"[P|p][R|r][O|o][g|G][r|R][a|A][m|M][a|A][\s|\S][d|D][e|E|O|o|A|a][\s|\S][T|t][R|r][\s][A|a][\s][B|b][\s][A|a][\s][L|l][\s][H|h][\s][O|o][\s|\S].*?[:|;|[\s\S].*?(\d*[^;|,|–|a-zA-Z]*)"
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
                        yield start.i, end.i, "programa_trabalho"
                        break
         #Programa de Trabalho - PT: 
        expression = r"[P|p][R|r][O|o][g|G][r|R][a|A][m|M][a|A][\s|\S][d|D][e|E|O|o|A|a][\s|\S][T|t][R|r][A|a][B|b][A|a][L|l][H|h][O|o][\s\S][-][\s\S][P][T][\s|\S].*?[:|;|[\s\S].*?(\d*[^;|,|–|a-zA-Z]*)" 
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
                            
                        yield start.i, end.i, "programa_trabalho"
                        break
        #Programa de Trabalho:
        expression = r"([P|p][R|r][O|o][g|G][r|R][a|A][m|M][a|A][\s|\S][d|D][e|E|O|o|A|a][\s|\S][T|t][R|r][A|a][B|b][A|a][L|l][H|h][O|o][:][\s\S]).*?(\d*[^;|,|–|a-zA-Z]*)"
                                                
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
                        yield start.i, end.i, "programa_trabalho"
                        break
    def natureza_despesa_(self, doc):
        '''
        label function para extracao de natureza de despesa usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[N|n][a|A][t|T][u|U][r|R][e|E][z|Z][a|A][\s\S][D|d][e|E|a|A][\s\S][d|D][e|E][s|S][p|P][e|E][s|S][a|A][\s\S].*?(\d*[^;|,|–|(|a-zA-Z]*)"
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
                        yield start.i, end.i, "natureza_despesa"
                        break
        expression = r"([N][.][D].*?[\s\S].*?|[N][D][\s|:].*?[\s\S].*?)(\d*[^;|,|–|(|a-zA-Z]*)"
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
                        yield start.i, end.i, "natureza_despesa"
                        break

    def nota_empenho_(self, doc):
        '''
        label function para extracao de nota de empenho usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"(\d+NE\d+)"
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "nota_empenho"
                #         break
                # break

    
    def valor_detector_fun(self, doc):
        '''
        label function para extracao de valor com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+3 < len(doc):
                for y in ['Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:']:
                    if y in token.text:
                        for x in range(1, len(doc)-token.i-2):
                            if (doc[token.i+x].text in ['R$', '$', '$$'] and doc[token.i+x+1].text[0].isdigit()) and doc[token.i+x].i+1 < doc[token.i+x].i+2:
                                yield doc[token.i+x].i+1, doc[token.i+x].i+2,"valor_convenio",
                                break
            
   

    def programa_trab_detector_fun(self, doc):
        '''
        label function para extracao de programa de trabalho com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+5 < len(doc):
                for y in ['Programa', 'PROGRAMA', 'programa']:
                    if y in token.text and doc[token.i+1].text in ['de', 'do', 'da', 'DE', 'DO', 'DA'] and doc[token.i+2].text in ['trabalho', 'Trabalho', 'Tr a b a l h o', 'Tr a b a l h o :', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:']:
                        k = 0
                        if(len(doc[token.i+3].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+4].text) <= 2):
                            k += 1
                        if(doc[token.i+3+k].text.isalpha()):
                            break
                        for x in range(3+k, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';']) and token.i+3+k < token.i+x:
                                yield token.i+3+k, token.i+x,  "programa_trabalho"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1,  "programa_trabalho"
                                break
            if token.i+3 < len(doc):
                for y in ["P.T", "PT"]:
                    if y == token.text:
                        k = 0
                        if(len(doc[token.i+1].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+2].text) <= 2):
                            k += 1
                        if(doc[token.i+1+k].text.isalpha()):
                            break
                        for x in range(1+k, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';']) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x,  "programa_trabalho"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1,  "programa_trabalho"
                                break

    def natureza_desp_detector_fun(self, doc):
        '''
        label function para extracao de natureza de despesa com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+5 < len(doc):
                for y in ['Natureza', 'NATUREZA', 'natureza']:
                    if y in token.text and doc[token.i+1].text in ['de', 'do', 'da', 'DE', 'DO', 'DA', 'DAS', 'das'] and doc[token.i+2].text in ['despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAS', 'despesas:', 'Despesas:', 'DESPESAS:']:
                        k = 0
                        if(len(doc[token.i+3].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+4].text) <= 2):
                            k += 1
                        if(doc[token.i+3+k].text.isalpha()):
                            break
                        for x in range(3, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';']) and token.i+3+k < token.i+x:
                                yield token.i+3+k, token.i+x, "natureza_despesa"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+3+k < token.i+x+1:
                                yield token.i+3+k, token.i+x+1, "natureza_despesa"
                                break
            if token.i+3 < len(doc):
                for y in ["N.D", "ND"]:
                    if y == token.text:
                        k = 0
                        if(len(doc[token.i+1].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+2].text) <= 2):
                            k += 1
                        if(doc[token.i+1+k].text.isalpha()):
                            break
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';']) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "natureza_despesa"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1, "natureza_despesa"
                                break

    def nota_emp_detector_fun(self, doc):
        '''
        label function para extracao de nota de empenho com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+2 < len(doc):
                for y in ['Empenho', 'EMPENHO', 'Empenho:', 'EMPENHO:', 'empenho:']:
                    if y in token.text:
                        k = 0
                        if((len(doc[token.i+1].text) <= 2 or doc[token.i+1].text in ['No', 'NO', 'no', 'Nº', 'nº', 'N°', 'n°', 'n', 'n.', 'n.º'])):
                            if 'R$' in doc[token.i+1].text:
                                break
                            k += 1
                        if(k >= 1 and (len(doc[token.i+2].text) <= 2 or doc[token.i+2].text in ['No', 'NO', 'no', 'Nº', 'nº', 'N°', 'n°', 'n', 'n.', 'n.º'])):
                            if 'R$' in doc[token.i+2].text:
                                break
                            k += 1
                        if(doc[token.i+1+k].text.isalpha()):
                            break
                        for x in range(1+k, len(doc)-token.i):
                            if 'R$' in doc[token.i+x].text:
                                break
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';']) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "nota_empenho"
                                break
                            elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1, "nota_empenho"
                                break
    
    def unidade_orc_detector_fun(self, doc):
        '''
        label function para extracao de unidade orcamentaria com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+4 < len(doc):
                for y in ['Unidade', 'UNIDADE', 'unidade']:
                    if y in token.text and doc[token.i+1].text in ['Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:']:
                        k = 0
                        if(len(doc[token.i+2].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+3].text) <= 2):
                            k += 1
                        if(doc[token.i+2+k].text.isalpha()):
                            break
                        for x in range(2+k, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';', " "] and token.i+2+k < token.i+x):
                                yield token.i+2+k, token.i+x, "unidade_orcamentaria"
                                break
                            ''' elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1, "unidade_orcamentaria"
                                break '''
            if token.i+3 < len(doc):
                for y in ["U.O", "UO", "UO:", "U.O:"]:
                    if y == token.text:
                        k = 0
                        if(len(doc[token.i+1].text) <= 2):
                            k += 1
                        if(k >= 1 and len(doc[token.i+2].text) <= 2):
                            k += 1
                        if(doc[token.i+1+k].text.isalpha()):
                            break
                        for x in range(1+k, len(doc)-token.i):
                            if (doc[token.i+x].text.isalpha() or doc[token.i+x].text in ['.', ',', ';'," "]) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "unidade_orcamentaria"
                                break
                            ''' elif token.i+x+1 >= len(doc) and token.i+1+k < token.i+x+1:
                                yield token.i+1+k, token.i+x+1, "unidade_orcamentaria"
                                break '''
                            

class SkweakConvenio(LabelFunctionsConvenio):
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
        self.df = pd.DataFrame(columns=["numero_convenio", "processo_gdf",  "valor_convenio", "unidade_orcamentaria", "programa_trabalho", "natureza_despesa", "nota_empenho", "data_assinatura_convenio", "text", "labels"])

    def apply_label_functions(self):
        '''
        Aplica as label functions na base de contratos e extrai as entidades
        '''
        doc = self.docs

        detec_convenio = skweak.heuristics.FunctionAnnotator(
            "detec_convenio", self.convenio_)
        doc = list(detec_convenio.pipe(doc))

        detec_processo = skweak.heuristics.FunctionAnnotator(
            "detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))
        
        #Regex data
        detec_data = skweak.heuristics.FunctionAnnotator(
            "detec_data", self.data_assinatura_)
        doc = list(detec_data.pipe(doc))

        detec_valor = skweak.heuristics.FunctionAnnotator(
            "detec_valor", self.valor_)
        doc = list(detec_valor.pipe(doc))

        detec_unidade = skweak.heuristics.FunctionAnnotator(
            "detec_unidade", self.unidade_orcamento_)
        doc = list(detec_unidade.pipe(doc))

        detec_programa = skweak.heuristics.FunctionAnnotator(
            "detec_programa", self.programa_trabalho_)
        doc = list(detec_programa.pipe(doc))

        detec_natureza = skweak.heuristics.FunctionAnnotator(
            "detec_natureza", self.natureza_despesa_)
        doc = list(detec_natureza.pipe(doc))

        detec_nota = skweak.heuristics.FunctionAnnotator(
            "detec_nota", self.nota_empenho_)
        doc = list(detec_nota.pipe(doc))

        convenio_detector = skweak.heuristics.FunctionAnnotator(
            "contrato_detector", self.convenio_detector_fun)
        doc = list(convenio_detector.pipe(doc))

        unidade_orc_detector = skweak.heuristics.FunctionAnnotator(  "unidade_orc_detector", self.unidade_orc_detector_fun)
        doc = list(unidade_orc_detector.pipe(doc)) 
        
        #LB de posição
        #data_detector = skweak.heuristics.FunctionAnnotator("data_detector", self.data_detector_fun)
        #doc = list(data_detector.pipe(doc))
        
       # processo_detector = skweak.heuristics.FunctionAnnotator(
       #     "processo_detector", self.processo_detector_fun)
       # doc = list(processo_detector.pipe(doc))
        

        valor_detector = skweak.heuristics.FunctionAnnotator(
            "valor_detector", self.valor_detector_fun)
        doc = list(valor_detector.pipe(doc))

        programa_trab_detector = skweak.heuristics.FunctionAnnotator(
            "programa_trab_detector", self.programa_trab_detector_fun)
        doc = list(programa_trab_detector.pipe(doc))

        natureza_desp_detector = skweak.heuristics.FunctionAnnotator(
            "natureza_desp_detector", self.natureza_desp_detector_fun)
        doc = list(natureza_desp_detector.pipe(doc))

        nota_emp_detector = skweak.heuristics.FunctionAnnotator(
            "nota_emp_detector", self.nota_emp_detector_fun)
        doc = list(nota_emp_detector.pipe(doc))
     
        self.docs = doc

        

    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''

        model = skweak.aggregation.HMM("hmm", ["numero_convenio", "processo_gdf", "valor_convenio", "unidade_orcamentaria", "programa_trabalho","natureza_despesa", "nota_empenho", "data_assinatura_convenio"], sequence_labelling=True)

        self.docs = model.fit_and_aggregate(self.docs)

        for doc in self.docs:
            if "hmm" in doc.spans:
                doc.ents = doc.spans["hmm"]
            else:
                doc.ents = []

        ''' Salvando modelo HMM em uma pasta data '''
        if os.path.isdir("./data"):
            skweak.utils.docbin_writer(self.docs, "./data/convenio.spacy")
        else:
            os.mkdir("./data")
            skweak.utils.docbin_writer(self.docs, "./data/convenio.spacy")

    def get_IOB(self):
        '''
        retorna os resultados das entidades extraidas em IOB
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/convenio.spacy")
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
        doc_bin = DocBin().from_disk("./data/convenio.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"numero_convenio": "", "processo_gdf": "", "valor_contrato": "", "unidade_orcamentaria": "", "programa_trabalho": "","natureza_despesa": "", "nota_empenho": "", "data_assinatura_convenio": "", "text": "", "labels": ""}

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
