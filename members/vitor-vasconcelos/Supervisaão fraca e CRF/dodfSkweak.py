import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin
import os


class LabelFunctionsContratos:
    '''
    Classe que armazena as Label Functions para a aplicacao da supervisao fraca em contratos

    Atributos: 
        self.docs = base de dados de contratos, no formato de um vetor de strings onde cada string representa um contrato
        self.vet = vetor auxiliar para detecção de entidades nas Label Functions
    '''

    def __init__(self, dados):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(dados))
        self.vet = ['Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA',
                    'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK']

    def contrato_(self, doc):
        '''
        label function para extracao de contratos usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"(, doc\d+/\d{4})"
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                tamanho = len(grupo.split())
                for token in doc:
                    if token.text in {grupo}:
                        yield token.i-1, token.i+tamanho, "CONTRATO"
                break

    def processo_(self, doc):
        '''
        label function para extracao de processo usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[P|p][R|r][O|o][C|c][E|e][S|s][S|s][O|o][\s\S].*?(\d*[^;|,|a-zA-Z]*)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "PROCESSO"
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "PROCESSO"
                    break

    def data_assinatura_(self, doc):
        '''
        label function para extracao de data de assinatura usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a]:.*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4})"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "DATA_ASS."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "DATA_ASS."
                    break

    def valor_(self, doc):
        '''
        label function para extracao de valor usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[v|V][a|A][l|L][o|O][r|R].*?[\s\S].*?([R$ \d\.]*,\d{2})"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "VALOR"
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "VALOR"
                    break

    def unidade_orcamento_(self, doc):
        '''
        label function para extracao de unidade orcamentaria usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[u|U][n|N][i|I][d|D][a|A][d|D][e|E][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "UNI_ORC."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "UNI_ORC."
                    break
        expression = r"[U][.][O].*?[\s\S].*?(\d+.\d+)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "UNI_ORC."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "UNI_ORC."
                    break

    def programa_trabalho_(self, doc):
        '''
        label function para extracao de programa de trabalho usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[P|p][R|r][O|o][g|G][r|R][a|A][m|M][a|A][\s|\S][d|D][e|E|O|o|A|a][\s|\S][T|t][R|r][A|a][B|b][A|a][L|l][H|h][O|o].*?[:|;|[\s\S].*?(\d*[^;|,|–|(|Nat|Not|Uni|Ent]*)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "PROG_TRAB."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "PROG_TRAB."
                    break

    def natureza_despesa_(self, doc):
        '''
        label function para extracao de natureza de despesa usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[N|n][a|A][t|T][u|U][r|R][e|E][z|Z][a|A][\s\S][D|d][e|E|a|A][\s\S][d|D][e|E][s|S][p|P][e|E][s|S][a|A][:|\s|\S][\s\S].*?(\d*[^;|,|–|(|a-zA-Z]*)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "NAT_DESP."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "NAT_DESP."
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
                        yield token.i, token.i+tamanho+len(x), "NOTA_EMP."
                #         break
                # break

    def lei_orc_(self, doc):
        '''
        label function para extracao de lei orcamentaria usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[L|l][E|e][I|i][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?([N|n][o|O|º|°] \d+.\d+\/d{4}|[N|n][o|O|º|°] \d+.\d+)"
        match = re.search(expression, str(doc))
        if match:
            flag = 0
            for token in doc:
                if token.idx == match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx > match.span(1)[0] and flag == 0:
                    if(doc[token.i].text == ':'):
                        start = doc[token.i+1]
                    else:
                        start = token
                    flag = 1
                if token.idx == match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "LEI_ORC."
                    break
                if token.idx > match.span(1)[1] and flag == 1 and token.i > start.i:
                    end = token
                    yield start.i, end.i, "LEI_ORC."
                    break

    def contrato_detector_fun(self, doc):
        '''
        label function para extracao de contrato com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.text in {'CONTRATO'}:
                for x in range(1, len(doc)-token.i-1):
                    if (doc[token.i+x].text in {'No', 'NO', 'no', 'Nº', 'nº', 'N°', 'n°'} and doc[token.i+x+1].text[0].isdigit()) and doc[token.i+x].i < doc[token.i+x].i+2:
                        yield doc[token.i+x].i, doc[token.i+x].i+2, "CONTRATO",
                        break

    def processo_detector_fun(self, doc):
        '''
        label function para extracao de processos com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            for y in ['Processo', 'PROCESSO', 'Processo:', 'PROCESSO:']:
                if token.i+2 < len(doc):
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or doc[token.i+x].text in self.vet or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "PROCESSO"
                                break

    def partes_detector_fun(self, doc):
        '''
        label function para extracao de partes com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            for y in ['Partes', 'PARTES', 'partes:']:
                if token.i+2 < len(doc):
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', ';'} or doc[token.i+x].text in self.vet or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "PARTES"
                                break

    def objeto_detector_fun(self, doc):
        '''
        label function para extracao de objeto com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        flag = 0
        for token in doc:
            if token.i+2 < len(doc):
                if token.text in {'Objeto', 'OBJETO'}:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range(1, len(doc)-token.i-1):
                        if (doc[token.i+x].text in self.vet) and token.i+1+k < token.i+x:
                            flag = 1
                            yield token.i+1+k, token.i+x, "OBJETO"
                            break
                if token.text in {'tem', 'Tem'} and doc[token.i+1].text in {'por', 'POR'} and doc[token.i+2].text in {'objeto', 'Objeto', 'OBJETO'} and flag == 0:
                    for x in range(1, len(doc)-token.i-1):
                        if (doc[token.i+x].text in self.vet or token.i+x+1 >= len(doc)) and token.i < token.i+x:
                            yield token.i, token.i+x, "OBJETO"
                            break

    def valor_detector_fun(self, doc):
        '''
        label function para extracao de valor com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            for y in ['Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:']:
                if y in token.text:
                    for x in range(1, len(doc)-token.i-1):
                        if (doc[token.i+x].text in {'R$', '$', '$$'} and doc[token.i+x+1].text[0].isdigit()) and doc[token.i+x].i < doc[token.i+x].i+2:
                            yield doc[token.i+x].i, doc[token.i+x].i+2, "VALOR",
                            break

    def lei_orc_detector_fun(self, doc):
        '''
        label function para extracao de lei orcamentaria com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+3 < len(doc):
                for y in ['Lei', 'LEI', 'lei:', 'Lei:', 'LEI:']:
                    if y in token.text and doc[token.i+1].text in {'Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'}:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or token.i+x+1 >= len(doc)) and token.i+2+k < token.i+x:
                                yield token.i+2+k, token.i+x, "LEI_ORC."
                                break

    def unidade_orc_detector_fun(self, doc):
        '''
        label function para extracao de unidade orcamentaria com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+3 < len(doc):
                for y in ['Unidade', 'UNIDADE', 'unidade']:
                    if token.i+1 < len(doc):
                        if y in token.text and doc[token.i+1].text in {'Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'}:
                            k = 0
                            if(doc[token.i+2].text == ':'):
                                k = 1
                            for x in range(1, len(doc)-token.i):
                                if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or token.i+x+1 >= len(doc) and token.i+2+k < token.i+x):
                                    yield token.i+2+k, token.i+x, "UNI_ORC."
                                    break
                if "U.O" in token.text and token.i+2 < len(doc):
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range(1, len(doc)-token.i):
                        if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                            yield token.i+1+k, token.i+x, "UNI_ORC."
                            break

    def programa_trab_detector_fun(self, doc):
        '''
        label function para extracao de programa de trabalho com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+4 < len(doc):
                for y in ['Programa', 'PROGRAMA', 'programa']:
                    if y in token.text and doc[token.i+1].text in {'de', 'do', 'da', 'DE', 'DO', 'DA'} and doc[token.i+2].text in {'trabalho', 'Trabalho', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:'}:
                        k = 0
                        if(doc[token.i+3].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or token.i+x+1 >= len(doc)) and token.i+3+k < token.i+x:
                                yield token.i+3+k, token.i+x, "PROG_TRAB."
                                break

    def natureza_desp_detector_fun(self, doc):
        '''
        label function para extracao de natureza de despesa com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+4 < len(doc):
                for y in ['Natureza', 'NATUREZA', 'natureza']:
                    if y in token.text and doc[token.i+1].text in {'de', 'do', 'da', 'DE', 'DO', 'DA', 'DAS', 'das'} and doc[token.i+2].text in {'despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAS', 'despesas:', 'Despesas:', 'DESPESAS:'}:
                        k = 0
                        if(doc[token.i+3].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', ',', '–', ';', '('} or token.i+x+1 >= len(doc)) and token.i+3+k < token.i+x:
                                yield token.i+3+k, token.i+x, "NAT_DESP."
                                break

    def data_detector_fun(self, doc):
        '''
        label function para extracao de data de assinatura com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+2 < len(doc):
                for y in ['Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']:
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or doc[token.i+x].text in self.vet or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "DATA_ASS."
                                break

    def signatarios_detector_fun(self, doc):
        '''
        label function para extracao de signatarios com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+2 < len(doc):
                for y in ['SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', 'Signatários']:
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in self.vet or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "SIGNATARIOS"
                                break

    def vigencia_detector_fun(self, doc):
        '''
        label function para extracao de vigencia com comparacoes de listas

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        for token in doc:
            if token.i+2 < len(doc):
                for y in ['VIGÊNCIA', 'VIGENCIA', 'Vigência:', 'VIGÊNCIA:', 'VIGENCIA:', 'Vigencia:', 'Vigencia', 'Vigência']:
                    if y in token.text:
                        k = 0
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ';'} or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "VIGENCIA"
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
                        if(doc[token.i+1].text == ':'):
                            k = 1
                        for x in range(1, len(doc)-token.i):
                            if (doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or token.i+x+1 >= len(doc)) and token.i+1+k < token.i+x:
                                yield token.i+1+k, token.i+x, "NOTA_EMP."
                                break


class SkweakContratos(LabelFunctionsContratos):
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
        self.df = pd.DataFrame(columns=["CONTRATO", "PROCESSO", "PARTES", "OBJETO", "VALOR", "LEI_ORC.", "UNI_ORC.",
                                        "PROG_TRAB.", "NAT_DESP.", "NOTA_EMP.", "DATA_ASS.", "SIGNATARIOS", "VIGENCIA", "text", "labels"])

    def apply_label_functions(self):
        '''
        Aplica as label functions na base de contratos e extrai as entidades
        '''
        doc = self.docs
        detec_contrato = skweak.heuristics.FunctionAnnotator(
            "detec_contrato", self.contrato_)
        doc = list(detec_contrato.pipe(doc))

        detec_processo = skweak.heuristics.FunctionAnnotator(
            "detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))

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

        # Aplicando cada uma no dataset
        detec_lei_orc = skweak.heuristics.FunctionAnnotator(
            "detec_lei_orc", self.lei_orc_)
        docs = list(detec_lei_orc.pipe(doc))

        contrato_detector = skweak.heuristics.FunctionAnnotator(
            "contrato_detector", self.contrato_detector_fun)
        doc = list(contrato_detector.pipe(doc))

        processo_detector = skweak.heuristics.FunctionAnnotator(
            "processo_detector", self.processo_detector_fun)
        doc = list(processo_detector.pipe(doc))

        partes_detector = skweak.heuristics.FunctionAnnotator(
            "partes_detector", self.partes_detector_fun)
        doc = list(partes_detector.pipe(doc))

        objeto_detector = skweak.heuristics.FunctionAnnotator(
            "objeto_detector", self.objeto_detector_fun)
        doc = list(objeto_detector.pipe(doc))

        valor_detector = skweak.heuristics.FunctionAnnotator(
            "valor_detector", self.valor_detector_fun)
        doc = list(valor_detector.pipe(doc))

        lei_orc_detector = skweak.heuristics.FunctionAnnotator(
            "lei_orc_detector", self.lei_orc_detector_fun)
        doc = list(lei_orc_detector.pipe(doc))

        unidade_orc_detector = skweak.heuristics.FunctionAnnotator(
            "unidade_orc_detector", self.unidade_orc_detector_fun)
        doc = list(unidade_orc_detector.pipe(doc))

        programa_trab_detector = skweak.heuristics.FunctionAnnotator(
            "programa_trab_detector", self.programa_trab_detector_fun)
        doc = list(programa_trab_detector.pipe(doc))

        natureza_desp_detector = skweak.heuristics.FunctionAnnotator(
            "natureza_desp_detector", self.natureza_desp_detector_fun)
        doc = list(natureza_desp_detector.pipe(doc))

        data_detector = skweak.heuristics.FunctionAnnotator(
            "data_detector", self.data_detector_fun)
        doc = list(data_detector.pipe(doc))

        signatarios_detector = skweak.heuristics.FunctionAnnotator(
            "signatarios_detector", self.signatarios_detector_fun)
        doc = list(signatarios_detector.pipe(doc))

        vigencia_detector = skweak.heuristics.FunctionAnnotator(
            "vigencia_detector", self.vigencia_detector_fun)
        doc = list(vigencia_detector.pipe(doc))

        nota_emp_detector = skweak.heuristics.FunctionAnnotator(
            "nota_emp_detector", self.nota_emp_detector_fun)
        doc = list(nota_emp_detector.pipe(doc))

        self.docs = doc

    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''
        model = skweak.aggregation.HMM("hmm", ["CONTRATO", "PROCESSO", "PARTES", "OBJETO", "VALOR", "LEI_ORC.", "UNI_ORC.", "PROG_TRAB.",
                                               "NAT_DESP.", "NOTA_EMP.", "DATA_ASS.", "SIGNATARIOS", "VIGENCIA"], sequence_labelling=True)

        self.docs = model.fit_and_aggregate(self.docs)

        for doc in self.docs:
            if "hmm" in doc.spans:
                doc.ents = doc.spans["hmm"]
            else:
                doc.ents = []

        ''' Salvando modelo HMM em uma pasta data '''
        if os.path.isdir("./data"):
            skweak.utils.docbin_writer(self.docs, "./data/reuters_small.spacy")
        else:
            os.mkdir("./data")
            skweak.utils.docbin_writer(self.docs, "./data/reuters_small.spacy")

    def get_IOB(self):
        '''
        retorna os resultados das entidades extraidas em IOB
        '''
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/reuters_small.spacy")
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
        doc_bin = DocBin().from_disk("./data/reuters_small.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"CONTRATO": "", "PROCESSO": "", "PARTES": "", "OBJETO": "", "VALOR": "", "LEI_ORC.": "", "UNI_ORC.": "", "PROG_TRAB.": "",
                   "NAT_DESP.": "", "NOTA_EMP.": "", "DATA_ASS.": "", "SIGNATARIOS": "", "VIGENCIA": "", "text": "", "labels": ""}

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
