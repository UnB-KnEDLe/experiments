# CODIGO DAS LABEL FUNCIONS E SUPERVISÃO FRACA

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

    def contrato_(self, doc):
      
        '''
        label function para extracao de contratos usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''

        expression = r"([C|c][O|o][N|n|][T|t][R|r][A|a][T|t][O|o][\s\S].*?[s|:]?.+?(?=[0-9]).*?|[C|c][O|o][N|n|][V|v][E|e|Ê|ê][N|n][I|i][O|o][\s\S].*?[s|:]?.+?(?=[0-9]).*?)(\d*[^;|,|a-zA-Z]*)"
        
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
        #expression = re.compile(r"(data d[e|a] assinatura.*\d{1,})|(assinatura\n*\d{1,})|(data assinatura.*\d{1,})|(assinatura.*\d{1,})|(data.*\d{1,}.*vig[e|ê]ncia)",
        #                     flags=re.IGNORECASE | re.UNICODE | re.DOTALL)    
        
        #expression = r"(data d[e|a] assinatura.*\d{1,})|(assinatura\n*\d{1,})|(data assinatura.*\d{1,})|(assinatura.*\d{1,})|(data.*\d{1,}.*vig[e|ê]ncia)"
        #expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a]:.*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4})"
        expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a].*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|\w{2}[\s\S]\w+[\s\S]\w+[\s\S]\w+[\s\S]\d{4}|[\s\S](\d{2}\.\d{2}\.\d{4})|\w{2}/\d{2}\/\d{4})"
        
        #padrao = re.compile(expression, str(doc),flags=re.IGNORECASE | re.UNICODE | re.DOTALL)
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
                        yield start.i, end.i, "data_assinatura_contrato"
                        break



    def valor_(self, doc):
        '''
        label function para extracao de valor usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[V][a|A][l|L][o|O][r|R].*?[\s\S].*?([\d\.]*,\d{2})"
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

    def unidade_orcamento_(self, doc):
        """ '''
        label function para extracao de unidade orcamentaria usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        """
        #expression = r"[u|U][n|N][i|I][d|D][a|A][d|D][e|E][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)"

        expression = r"[u|U][n|N][i|I][d|D][a|A][d|D][e|E][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)|[D|d][O|o][T|t][A|a][c|C|ç|Ç][A|a|ã|Ã][O|o][\s\S][o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)" 
        match = re.finditer(expression, str(doc))
        if match:
            for grupo in match:
                flag = 0
                for token in doc:
                    if grupo.span(1)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        print("1 - doc[token.i].text ", doc[token.i].text)
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                            print("2 - start token", start)
                        flag = 1
                    if token.idx >= grupo.span(1)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':', '/', ',']):
                            end = doc[token.i-1]
                        else:
                            end = token
                            print("3 - end token", end)
                        yield start.i, end.i, "unidade_orcamentaria"
                        print("4 - start e end", start.i, end.i)
                        break
        
        expression_x = r"([U][.][O].*?[\s\S].*?|[U][O][\s|:].*?[\s\S].*?)(\d+.\d+)"
                         
        match_1 = re.finditer(expression_x, str(doc))
        if match_1:
            for grupo in match_1:
                flag = 0
                for token in doc:
                    if grupo.span(2)[0]+1 in range(token.idx, (token.idx+len(token))+1) and flag == 0:
                        print("1.1 - doc[token.i].text", doc[token.i].text)
                        if(doc[token.i].text == ':'):
                            start = doc[token.i+1]
                        else:
                            start = token
                            print("2.1 - start token", start)
                        flag = 1
                    if token.idx >= grupo.span(2)[1] and flag == 1 and token.i > start.i:
                        if(doc[token.i-1].text in ['.', '-', '—', ':', '/', ',']):
                            end = doc[token.i-1]
                        else:
                            end = token
                            print("3.1 - end token", end)
                        yield start.i, end.i, "unidade_orcamentaria"
                        print("4.1 - start e end", start.i, end.i)
                        break

    def programa_trabalho_(self, doc):
        '''
        label function para extracao de programa de trabalho usando regex

        parametros:
            doc: uma string respresentando o texto de um dos contratos oferecidos no vetor da base de dados
        '''
        expression = r"[P|p][R|r][O|o][g|G][r|R][a|A][m|M][a|A][\s|\S][d|D][e|E|O|o|A|a][\s|\S][T|t][R|r][A|a][B|b][A|a][L|l][H|h][O|o].*?[:|;|[\s\S].*?(\d*[^;|,|–|a-zA-Z]*)"
        
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
                        yield start.i, end.i, "PROG_TRAB."
                        break
        expression = r"([P][.][T].*?[\s\S].*?|[P][T][\s|:].*?[\s\S].*?)(\d*[^;|,|–|a-zA-Z]*)"
        
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
                        yield start.i, end.i, "PROG_TRAB."
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
                        yield start.i, end.i, "NAT_DESP."
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
                        yield token.i, token.i+tamanho+len(x), "nota_empenho"
                #         break
                # break

    
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
        #self.df = pd.DataFrame(columns=["numero_contrato", "processo_gdf", "PARTES", "CONTRATANTE", "CONTRATADA", "OBJETO", "valor_contrato", "unidade_orcamentaria",
        #                                "programa_trabalho", "natureza_despesa", "nota_empenho", "data_assinatura_contrato", "VIGENCIA", "text", "labels"])

        self.df = pd.DataFrame(columns=["numero_contrato", "processo_gdf",  "valor_contrato", "unidade_orcamentaria",
                                        "programa_trabalho", "natureza_despesa", "nota_empenho", "data_assinatura_contrato", "text", "labels"])                        

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

        """ contrato_detector = skweak.heuristics.FunctionAnnotator(
            "contrato_detector", self.contrato_detector_fun)
        doc = list(contrato_detector.pipe(doc))

        processo_detector = skweak.heuristics.FunctionAnnotator(
            "processo_detector", self.processo_detector_fun)
        doc = list(processo_detector.pipe(doc))

        valor_detector = skweak.heuristics.FunctionAnnotator(
            "valor_detector", self.valor_detector_fun)
        doc = list(valor_detector.pipe(doc))

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

        nota_emp_detector = skweak.heuristics.FunctionAnnotator(
            "nota_emp_detector", self.nota_emp_detector_fun)
        doc = list(nota_emp_detector.pipe(doc))
 """
        self.docs = doc



    def train_HMM_Dodf(self):
        '''
        treina o modelo HMM para refinar e agregar a entidades extraidas pelas label functions
        '''
        #model = skweak.aggregation.HMM("hmm", ["numero_contrato", "processo_gdf", "PARTES", "CONTRATANTE", "CONTRATADA", "OBJETO", "valor_contrato", "unidade_orcamentaria", "programa_trabalho",
        #                                       "natureza_despesa", "nota_empenho", "data_assinatura_contrato",  "VIGENCIA"], sequence_labelling=True)

        model = skweak.aggregation.HMM("hmm", ["numero_contrato", "processo_gdf", "valor_contrato", "unidade_orcamentaria", "programa_trabalho",
                                               "natureza_despesa", "nota_empenho", "data_assinatura_contrato"], sequence_labelling=True)

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
            aux = {"numero_contrato": "", "processo_gdf": "",  "valor_contrato": "", "unidade_orcamentaria": "", "programa_trabalho": "",
                   "natureza_despesa": "", "nota_empenho": "", "data_assinatura_contrato": "","text": "", "labels": ""}


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
