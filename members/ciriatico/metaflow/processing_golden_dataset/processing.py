import pandas as pd
from nltk.tokenize import word_tokenize
import json

class JSON:
    def read_json(path):
        with open(path) as json_file:
            data = json.load(json_file)
        json_file.close()

        return data

    def write_json(string, path):
        with open(path, 'w') as outfile:
            json.dump(string, outfile)
        outfile.close()

class Processing:
    def preprocess_dataset(labeled_acts, labeled_entities, filter_act, eq_column, tokens_eq_rep):
        # Retorna lista com infos necessárias para o IOB:
        # texto, (token, endereço inicial do token, endereço final)

        labeled_entities = labeled_entities[labeled_entities["act_type"] == filter_act]
        labeled_entities = labeled_entities.sort_values(["file_id", "loc_end"], ascending=False).reset_index(drop=True)
        labeled_entities[eq_column] = labeled_entities[eq_column].replace(tokens_eq_rep)

        acts = []

        for file in labeled_acts["file_id"].unique():
            token_ents = labeled_entities[labeled_entities["file_id"] == file][[eq_column, "loc_begin", "loc_end"]]
            loc_tokens = [tuple(x) for x in token_ents.to_numpy()]
            text = labeled_acts[labeled_acts["file_id"] == file]["raw_text"].iloc[0]
            
            acts.append((text, loc_tokens))

        return acts

    def mark_text(text, loc_tokens, begin_marker="B__B", end_marker="E__E"):
        # loc_tokens deve estar ordenado de forma decrescente com base em end
        # Se começa em ordem crescente, altera o tamanho da string e o próximo endereço já não é válido

        for token, begin, end in loc_tokens:
            begin_marker_token = " {}-{}-{} ".format(begin_marker, token, begin_marker)
            end_marker_token = " {}-{}-{} ".format(end_marker, token, end_marker)

            text = text[:end] + end_marker_token +  text[end:]
            text = text[:begin] + begin_marker_token +  text[begin:]

        return text
    
    def join_block(text):
        # Transformar texto em linhas de folha de documento em texto corrido em parágrafo

        a = "\n".join([l for l in text.split("\n") if l != ""])
        words = a.replace("\n", " ").split(" ")
        words = [w for w in words if w != ""]

        m_words = []
        dash_cut = False

        for i in range(len(words)):
            word = words[i]

            if (word[-1] == "-") and (i+1)<len(words):
                word = word[:-1] + words[i+1]
                i += 1

            m_words.append(word)

        return " ".join(m_words)

    def clean_text(text, filter_patterns, rep=""):
        filter_patterns = "|".join(filter_patterns)

        text = pd.Series(text).str.replace(filter_patterns, rep, regex=True)[0]

        return Processing.join_block(text)
    
    def tokenize_text(text, begin_marker="B__B", end_marker="E__E"):
        inside = False
        start = False
        text_lines_mod = []

        text_lines = word_tokenize(text)

        for line in text_lines:
            if "{}-".format(begin_marker) in line:
                start = True
                inside = True
                token = line.split(begin_marker)[1].split("-")[1]
            elif "{}-".format(end_marker) in line:
                inside = False
            else:
                if not inside:
                    line += " X X O"
                else:
                    if start:
                        line += " X X B-" + token
                    else:
                        line += " X X I-" + token
                    start = False

                text_lines_mod.append(line)

        return text_lines_mod