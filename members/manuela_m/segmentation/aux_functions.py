import re
import pandas as pd
import json

patterns_clean = json.loads(
    r'{"patterns": [["\\nP\u00c1GINA\\s([0-9]{1,5})", "\\nDI\u00c1RIO\\sOFICIAL\\sDO\\sDISTRITO\\sFEDERAL", "\\nN\u00ba(.+?)2([0-9]{3})", "\\nxx([a-z]{0,10}) Di\u00e1rio Oficial do Distrito Federal xx([a-z]{0,10})", "\\nDi\u00e1rio Oficial do Distrito Federal", "Documento assinado digitalmente conforme MP n\u00ba 2.200-2 de 24/08/2001, que institui a", "Infraestrutura de Chaves P\u00fablicas Brasileira ICP-Brasil", "Este documento pode ser verificado no endere\u00e7o eletr\u00f4nico", "http://wwwin.gov.br/autenticidade.html", "pelo c\u00f3digo ([0-9]{15,18})", "\\nDocumento assinado digitalmente, original em https://www.dodf.df.gov.br", "http:/Awwwin.gov.br/autenticidade.html", "Documento assinado digitalmente conforme MP n\u00ba 2.200-2 de 24/08/2001,", "\\nque institui a\\n", "\\nhttp://www.in.gov.br/autenticidade.html", "\\nhttp://www.in.gov.brautenticidade html", "Documento assinado digitalmente conforme MP n 2.200-2 de 24/08/2001, que institui a .", "http://www.in.gov.brautenticidade html,", "xx([a-z]{1,10}) ", " xx([a-z]{1,10})", "xx([a-z]{1,10})"], ["\\n-\\n", "\\n- -\\n", "\\n- - -\\n", "\\n[\\.\\,\\-\\\u2014]\\n", "\u2014 -", ". -", "\\r[\\.\\,\\-\\\u2014]\\r", "\\n-\\r", "\\r"]], "rep": [" ", "\n"]}'
)


def merge_data(data):
    # Concatenar os dados do DODF, juntando todas as páginas em uma linha
    return (
        data.sort_values(by=["file_name", "page"])
        .groupby(["file_name", "number", "day", "month", "year"], as_index=False)
        .agg({"text": "\n".join})
    )


def join_block(text):
    # Transformar texto em linhas de folha de documento em texto corrido em parágrafo
    a = "\n".join([l for l in text.split("\\n") if l != ""])
    words = a.replace("\n", " ").split(" ")
    words = [w for w in words if w != ""]

    m_words = []
    dash_cut = False

    for i in range(len(words)):
        word = words[i]

        if (word[-1] == "-") and (i + 1) < len(words):
            word = word[:-1] + words[i + 1]
            i += 1

        m_words.append(word)

    return " ".join(m_words)


def clean_text(text, filter_patterns=patterns_clean, rep=""):
    filter_patterns = "|".join(filter_patterns)

    text = pd.Series(text).str.replace(filter_patterns, rep, regex=True)[0]

    return join_block(text)


def get_filtered_blocks(df):
    # Separar texto em blocos e limpá-los

    pos_texts = []

    # Identificação de blocos é feita com as tags xxbob e xxeob
    for number, text in df[["number", "text"]].itertuples(index=False):
        begin = list(re.finditer("xxbob", text))
        end = list(re.finditer("xxeob", text))
        pos_texts.append([number, text, begin, end])

    full_blocks = []

    for p in pos_texts:
        blocks = []

        number = p[0]
        text = p[1]
        begin = p[2]
        end = p[3]

        if (number % 50) == 0:
            print(number)

        for i in range(len(begin)):
            block = text[begin[i].start() : end[i].end()]

            for i in range(len(patterns_clean["patterns"])):
                block = clean_text(
                    text=block,
                    filter_patterns=patterns_clean["patterns"][i],
                    rep=patterns_clean["rep"][i],
                )

            blocks.append(block)

        full_blocks.append([number, blocks])

    return full_blocks
