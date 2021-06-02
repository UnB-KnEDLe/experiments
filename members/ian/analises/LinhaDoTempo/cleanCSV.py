import glob
import re
import pandas as pd
from yattag import Doc
import unicodedata
import numpy as np
from collections import Counter


def clean_process():
    """Funcao que padroniza a coluna 'processo' do dataframe 'df', deixando os valores com apenas numeros,
       e acrescentando 0's a esquerda, deixando todos com o tamanho 25"""
    for idx in range(len(df)):
        if not pd.isna(df['processo'][idx]):
            df['processo'][idx] = re.sub('[^0-9]', '', df['processo'][idx])

            df['processo'][idx] = '{:0>25}'.format(df['processo'][idx])


def clean_date():
    """Funcao que padroniza a coluna 'data' do dataframe 'df', deixando o valores no padrao:
       XX/nome_mes/XXXX"""
    for idx in range(len(df)):
        if not pd.isna(df['data'][idx]):
            df['data'][idx] =  df['data'][idx].replace('\n', ' ').replace(' de ', '/')


doc, tag, text = Doc().tagtext()
txt_files = glob.glob('../data/**/**/*.txt')


# Extraindo os atos de licitacoes dos DODF's
all_contracts = []
for txt in txt_files:
    instances = re.findall('''(?:AVISO(?:S)?\s+D[EO]\s+ABERTURA\s+D[EO]\s+LICITACAO|AVISO(?:S)?\s+D[EO]\s+ADJUDICACAO\s+E\s+HOMOLOGACAO|AVISO(?:S)?\s+D[EO]\s+HOMOLOGACAO\s+E\s+ADJUDICACAO|RESULTADO(?:S)?\s+D[EO]\s+JULGAMENTO|AVISO\s+D[EO]\s+RESULTADO\s+D[EO]\s+JULGAMENTO|AVISO\s+D[EO]\s+DECLARACAO\s+D[EO]\s+VENCEDOR|AVISO(?:S)?\s+D[EO]\s+REVOGACAO\s+D[EO]\s+LICITACAO|AVISO(?:S)?\s+D[EO]\s+REABERTURA\s+D[EO]\s+LICITACAO)(?:[\s\S]+?
)(?:[A-ZÁÃÂÓÍÔÎÊÉÕ]+\s.+?(?:[A-Z]{3,})(?:(?:"|\n(?!(?:S|DE)\s+MEDICAMENTOS|\d|INTERESSADO|. |A Comissao|formado|EIRELI -|BRASILEIRO S/A|AGENCIA|COMPANHIA IMOBILIARIA|EXTERIOR|EIRELI no|AUDITORES|HEMOCARE|HOSPITALARES|HOSPITALAR|GERAL|COMERCIAL|EXTRATO|CIRURGICOS|BARRETO COMERCIAL|CONSULTORES|LTDA|NUMERADOS|FARMACEUTICOS|FARMACEUTICA|DE PRODUTOS|PRODUTOS|CONCORRENCIA|FEDERAL|INDUSTRIA|PROCESSO|COMERCIO|COM TRATAMENTO FAVORECIDO E DIFERENCIADO A ME E EPP\nPROCESSO SEI-GDF|Repeticao dos itens fracassados))))''',
                            unicodedata.normalize('NFKD', open(txt, encoding='utf-8').read()))
    all_contracts.extend(instances)


# Extraindo as entidades dos atos de licitacoes
regex_dict = {
    'ato': '(AVISO(?:S)?\s+D[EO]\s+ABERTURA\s+D[EO]\s+LICITACAO|AVISO(?:S)?\s+D[EO]\s+ADJUDICACAO\s+E\s+HOMOLOGACAO|AVISO(?:S)?\s+D[EO]\s+HOMOLOGACAO\s+E\s+ADJUDICACAO|RESULTADO(?:S)?\s+D[EO]\s+JULGAMENTO|AVISO\s+D[EO]\s+RESULTADO\s+D[EO]\s+JULGAMENTO|AVISO\s+D[EO]\s+DECLARACAO\s+D[EO]\s+VENCEDOR|AVISO(?:S)?\s+D[EO]\s+REVOGACAO\s+D[EO]\s+LICITACAO|AVISO(?:S)?\s+D[EO]\s+REABERTURA\s+D[EO]\s+LICITACAO)',
    'processo': '(?:(?:(?:P|p)rocesso(?:\s+)?(?:(?:\()?SEI(?:\)?))?(?:\s+)?(?:(?:no|n\.o)?)?)|(?:P|p)rocesso:|(?:P|p)rocesso|Processo.|(?:P|p)rocesso\s+no|(?:P|p)rocesso\s+no.|(?:P|p)rocesso\s+no:|(?:P|p)rocesso\s+SEI\s+no:|(?:P|p)rocesso\s+SEI:|(?:P|p)rocesso\s+SEI\s+no|(?:P|p)rocesso SEI|(?:P|p)rocesso\s+SEI\s+no.|(?:P|p)rocesso\s+SEI\.|PROCESSO:|PROCESSO|PROCESSO.|PROCESSO\s+no|PROCESSO\s+no.|PROCESSO\s+no:|PROCESSO\s+SEI\s+no:|PROCESSO\s+SEI:|PROCESSO\s+SEI|PROCESSO\s+SEI\s+no|PROCESSO\s+SEI\s+no.|PROCESSO\s+SEI.)((?:(?!\s\d{2}.\d{3}.\d{3}/\d{4}-\d{2}))(?:(?:\s)(?:(?:[\d.]+)|(?:[\d\s]+))[.-]?(?:(?:\d)|(?:[.\d\sSEI-]+))(?:/|-|\b)(?:(?:(?:\d)+|(?:[\d\s]+)))?(?:-(?:(?:\d)+|(?:[\d\s]+)))?(?:-SECOM/DF|/CBMDF|F J Z B / D F)?))',
    'data': 'Bras[[ií]lia(?:\/?DF)?,?\s+(\d{2}\s+de+\s\w+\s+de\s+\d{4})',
}

df_dict = {
    'ato': [],
    'processo': [],
    'data': [],
    'texto': [],
}


# Criando o dataframe
for contract in all_contracts:
    df_dict['texto'].append(contract)
    for field in regex_dict:
        match = re.search(regex_dict[field], contract)
        if match:
            res = tuple(x for x in match.groups() if x is not None)
            df_dict[field].append(res[0])
        else:
            df_dict[field].append(np.nan)

df = pd.DataFrame.from_dict(df_dict)


# Chamando as funcoes que limpam as colunas 'processo' e 'data'
clean_process()
clean_date()


df.to_csv('atosLicitacao.csv')
