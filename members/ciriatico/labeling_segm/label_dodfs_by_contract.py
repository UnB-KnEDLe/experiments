import pandas as pd
import re
import json
import os
import re

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Difere do extrator no DODFMiner por não ter a etapa de limpeza inclusa, o que permite localizar no DODF bruto

class ContractExtractorRaw:
    """Extract contract statements from DODFs and export to file (PARQUET, JSON, TXT or CSV).

    Extracts contract statements from DODF dataframe through REGEX patterns.

    Note:
        This class is not constructable, it cannot generate objects.

    """
    
    @classmethod
    def extract_text(cls, file_path, file_type="dataframe"):
        """Extract texts of contract statements from dataframe or file

        Args:
            file_path: The path of the file that stores the dataframe.
            file_type: The type of the file that is being read. It can be
            passed directly a dataframe or a path to a file (PARQUET,
            TXT or CSV).

        Returns:
            Dataframe in which each row has a contract statement, with additional
            informations such as page, year, month and day of the DODF from which
            the text was extracted.
        """
        
        base_df = file_path

        contract_pattern_1 = r"(\nxx([a-z]{0,10})\sEXTRAT([A-Z]{0,3})\sD([A-Z]{0,3})\sCONTRAT([A-Z]{0,3}))"
        contract_pattern_2 = r"(\nxx([a-z]{0,10})\sEXTRAT([A-Z]{0,3})\sCONTRAT([A-Z]{0,3}))"

        contract_pattern = [contract_pattern_1, contract_pattern_2]
        contract_pattern = "|".join(contract_pattern)
        p_ext = re.compile(contract_pattern)

        block_pattern = r"(\nxx([a-z]{0,10})\s([A-ZÀ-Ú0-9º\/x\-\.\,\(\)\|\*&\s\']{5,}) xx([a-z]{0,10}))"
        block_pattern = block_pattern + "|" + contract_pattern
        p_blk = re.compile(block_pattern)

        nl_pattern = r"\n"
        p_nl = re.compile(nl_pattern)

        blocks_extracted = cls._extract_text_blocks(base_df, p_ext, p_blk, p_nl)

        return blocks_extracted
    
    @classmethod
    def _extract_text_blocks(cls, base_df, contract_pattern, block_pattern, newline_pattern):
        base_df = base_df.sort_values(["file_name", "page"])

        # Concatenar todas as páginas em uma única linha
        base_df = base_df.groupby(["file_name", "number", "day", "month", "year"])["text"].apply('\n'.join).reset_index()

        matched_texts = base_df.apply(lambda row: cls._row_list_regex(row.name, row["text"], contract_pattern, block_pattern, newline_pattern), axis=1)
        matched_texts = [matched_text for matched_text in matched_texts if matched_text != None]

        ext_blk_list = cls._mapped_positions_regex(matched_texts)

        extracted_texts_df = cls._extract_texts_from_mapped_positions(ext_blk_list, base_df, "text")

        ext_df = cls._extracted_texts_to_df(extracted_texts_df, base_df[["file_name", "number", "day", "month", "year"]], ["file_name", "number", "day", "month", "year", "text"])

        return ext_df

    @classmethod
    def _extracted_texts_to_df(cls, texts_indexes, base_df, column_names):
        # Gera dataframe com os textos extraídos, com informações do documento onde foram encontrados
        extracted_texts_df = []

        for extracted_text in texts_indexes:
            index_text = extracted_text[0]
            ext_text = extracted_text[1]

            list_text = list(base_df.loc[index_text, base_df.columns].values)
            list_text.append(ext_text)

            extracted_texts_df.append(list_text)

        return pd.DataFrame(extracted_texts_df, columns=column_names)

    @classmethod
    def _row_list_regex(cls, index, text, pattern_ext, pattern_blk, pattern_nl):
        # Lista com as buscas por regex. O findall permite ver se no documento há pelo menos 1 extrato detectado
        row_list = [index, re.findall(pattern_ext, text), re.finditer(pattern_ext, text), re.finditer(pattern_blk, text), re.finditer(pattern_nl, text)]

        # Se findall não achou nenhum, então o documento não tem nada que interessa
        if len(row_list[1]) > 0:
            return row_list

    @classmethod
    def _mapped_positions_regex(cls, matched_texts):
        # Mapeia as posições do que foi encontrado pelo regex
        # Lista de 2 dimensões: na primeira, de extrato; na segunda, de bloco
        mapped_positions_ext_blk = []

        for match in matched_texts:
            a_ext = match[2]
            a_blk = match[3]
            a_nl = match[4]

            a_ext_list = []
            a_blk_list = []
            a_nl_list = []

            for i in a_ext:
                a_ext_list.append(i.start())

            for i in a_blk:
                a_blk_list.append(i.start())

            for i in a_nl:
                a_nl_list.append(i.start())

            mapped_positions_ext_blk.append([match[0], a_ext_list, a_blk_list, a_nl_list])

        return mapped_positions_ext_blk

    @classmethod
    def _extract_texts_from_mapped_positions(cls, mapped_positions, base_df, text_column):
        extracted_texts = []

        mapped_positions = cls._extract_titles_blocks(mapped_positions)

        for mapped_position in mapped_positions:
            mapped_text = base_df.loc[mapped_position[0]][text_column]

            for ia in mapped_position[1]:
                ia_b = ia[0]
                ia_e = ia[-1]
                index_ia_b = mapped_position[2].index(ia_b)
                index_ia_e = mapped_position[2].index(ia_e)

                if (index_ia_e + 1) <= (len(mapped_position[2])-1):
                    ib = mapped_position[2][index_ia_e+1]
                    extracted_text = mapped_text[ia_b:ib]
                else:
                    extracted_text = mapped_text[ia_b:]

                extracted_texts.append([mapped_position[0], extracted_text])

        return extracted_texts

    @classmethod
    def _extract_titles_blocks(cls, mapped_positions):
        new_mapped_positions = []

        for mapped_page in mapped_positions:
            ext = mapped_page[1].copy()
            blk = mapped_page[2].copy()
            nl = mapped_page[3].copy()

            tls = []

            for t in ext:
                tl = []
                tl.append(t)

                loop_title = True

                # Um título acaba quando o próximo bloco não é a próxima linha
                while loop_title:
                    prox_blk_index = blk.index(t)+1
                    prox_nl_index = nl.index(t)+1

                    # Se o fim do documento não tiver sido alcançado
                    if prox_blk_index < len(blk):
                        prox_blk = blk[prox_blk_index]
                        prox_nl = nl[prox_nl_index]

                        # Se a próxima linha for vista como um próximo bloco
                        if prox_blk == prox_nl:
                            t = prox_blk
                            tl.append(t)
                        else:
                            loop_title = False

                    else:
                        loop_title = False

                tls.append(tl)

            new_mapped_positions.append([mapped_page[0], tls, blk, nl])

        return new_mapped_positions
    
    @classmethod
    def _log(cls, msg):
        print(f"[EXTRACTOR] {msg}")

    @classmethod
    def _create_single_folder(cls, path):
        if os.path.exists(path):
            cls._log(os.path.basename(path) + " folder already exist")
        else:
            try:
                os.mkdir(path)
            except OSError as error:
                cls._log("Exception during the directory creation")
                cls._log(str(error))
            else:
                basename = os.path.basename(path)
                cls._log(basename + " directory successful created")

    @classmethod
    def _read_file(cls, path_file, file_type="dataframe"):
        if file_type == "dataframe":
            return path_file

        if file_type == "parquet":
            return pd.read_parquet(path_file)

        if file_type == "csv":
            return pd.read_csv(path_file)

    @classmethod
    def _save_single_file(cls, base_df, folder, file_type):
        base_df_date = base_df.copy()
        base_df_date["date"] = pd.to_datetime(base_df_date[["year", "month", "day"]])
        min_date = min(base_df_date["date"]).strftime("%d-%m-%Y")
        max_date = max(base_df_date["date"]).strftime("%d-%m-%Y")

        file_name = "DODF_EXTRATOS_CONTRATO_{}_{}.{}"
        file_name = file_name.format(min_date, max_date, file_type)

        cls._create_single_folder(os.path.join(folder, RESULTS_PATH))
        path_file = os.path.join(folder, RESULTS_PATH, file_name)

        if file_type == "parquet":
            base_df.to_parquet(path_file)
            return

        if file_type == "csv":
            base_df.to_csv(path_file, index=False)
            return

        if file_type == "json":
            base_dict = base_df.to_dict("records")
            base_file = json.dumps(base_dict, ensure_ascii=False)

        if file_type == "txt":
            base_file = "\n\n".join(base_df.apply(lambda row: "----file_name: {}----number: {}----day: {}----month: {}----year: {}----text:\n{}".format(row["file_name"], row["number"], row["day"], row["month"], row["year"], row["text"]), axis=1))

        with open(path_file, "w", encoding="UTF-8") as file:
            file.write(base_file)

def label_dodfs_contracts(dodfs_full, dodfs_contracts, start_mark, end_mark):    
    dodf_texts = []

    for dodf_edicao in dodfs_full["file_name"].unique():
        temp_contratos = dodfs_contracts[dodfs_contracts["file_name"] == dodf_edicao].copy().reset_index(drop=True)
        temp_dodf = dodfs_full[dodfs_full["file_name"] == dodf_edicao].copy().reset_index(drop=True)

        contratos_validos = temp_contratos.apply(lambda row: True if (sum(temp_dodf["text"].str.contains(row["text"], regex=False)) == 1) and (temp_dodf["text"].str.count(re.escape(row["text"])) <= 1).all() else False, axis=1)
        temp_contratos = temp_contratos[contratos_validos].reset_index(drop=True)

        temp_dodf_text = temp_dodf["text"].iloc[0]

        for contrato in temp_contratos["text"]:
            contrato_start = temp_dodf_text.find(contrato)
            contrato_end = contrato_start + len(contrato)

            temp_dodf_text = temp_dodf_text[:contrato_end] + end_mark + temp_dodf_text[contrato_end:]
            temp_dodf_text = temp_dodf_text[:contrato_start] + start_mark + temp_dodf_text[contrato_start:]

        dodf_texts.append(temp_dodf_text)
        
    dodfs_full["marked_text"] = pd.Series(dodf_texts)
        
    return dodfs_full

def get_running_text(dodf_text):
    text_lines = dodf_text.split("\n")
    
    # Concatenando linhas de títulos
    previous_line_title = None
    running_lines = []

    for line in text_lines:
        if "xxbcet" in line:
            if previous_line_title == None:
                previous_line_title = line
            else:
                previous_line_title += " " + line

        else:
            if previous_line_title != None:
                running_lines.append(previous_line_title)
                previous_line_title = None
            running_lines.append(line)

    # Concatenando linhas de blocos
    previous_line_block = None
    full_running_lines = []

    for line in running_lines:
        if ("xxbob" not in line) and ("xxeob" not in line) and ("xxbcet" not in line):
            if previous_line_block == None:
                previous_line_block = line
            else:
                previous_line_block += " " + line

        else:
            if previous_line_block != None:
                full_running_lines.append(previous_line_block)
                previous_line_block = None
            full_running_lines.append(line)
            
    return "\n".join(full_running_lines)

def clean_text_dodfs(dodfs_texts):

    start_page_patterns = [r"\nPÁGINA\s([0-9]{1,5})", r"\nDIÁRIO\sOFICIAL\sDO\sDISTRITO\sFEDERAL",
                           r"\nNº(.+?)2([0-9]{3})", r"\nxx([a-z]{0,10}) Diário Oficial do Distrito Federal xx([a-z]{0,10})",
                           r"\nDiário Oficial do Distrito Federal"]

    end_page_patterns = [r"Documento assinado digitalmente conforme MP nº 2.200-2 de 24/08/2001, que institui a",
                            r"Infraestrutura de Chaves Públicas Brasileira ICP-Brasil",
                            r"Este documento pode ser verificado no endereço eletrônico",
                            r"http://wwwin.gov.br/autenticidade.html",
                            r"pelo código ([0-9]{15,18})",
                            r"\nDocumento assinado digitalmente, original em https://www.dodf.df.gov.br",
                        r"http:/Awwwin.gov.br/autenticidade.html",
                        r"Documento assinado digitalmente conforme MP nº 2.200-2 de 24/08/2001,",
                        r"\nque institui a\n",
                        r"\nhttp://www.in.gov.br/autenticidade.html",
                        r"\nhttp://www.in.gov.brautenticidade html",
                        r"Documento assinado digitalmente conforme MP n 2.200-2 de 24/08/2001, que institui a .",
                        r"http://www.in.gov.brautenticidade html,"]

    middle_page_patterns = [r"xx([a-z]{1,10}) ", r" xx([a-z]{1,10})", r"\n-\n",
                         r"xx([a-z]{1,10})", r"\n- -\n", r"\n- - -\n",
                        r"\n[\.\,\-\—]\n", r"— -", r". -"]
    
    start_page_patterns = "|".join(start_page_patterns)
    middle_page_patterns = "|".join(middle_page_patterns)
    end_page_patterns = "|".join(end_page_patterns)
    
    page_patterns = [start_page_patterns, middle_page_patterns, end_page_patterns]
    page_patterns = "|".join(page_patterns)
    
    return dodfs_texts.str.replace(page_patterns, "", regex=True)

def tokenize_text_dodf(dodf_text, labeled, token_type):    
    
    clean_text = "\n".join([part.strip() for part in dodf_text.split("\n") if not pd.Series(part).isin(["", ".", ",", "-"]).any()])

    if token_type == "sentence":
        tokenized_text = [sent_tokenize(part) for part in clean_text.split("\n")]

        flat_text = []

        for token in tokenized_text:
            flat_text += token

        tokenized_text = "\n".join(flat_text)
    
    elif token_type == "word":
        tokenized_text = [word_tokenize(part) for part in clean_text.split("\n")]

        flat_text = []

        for token in tokenized_text:
            flat_text += token

        tokenized_text = "\n".join(flat_text)
        
    else:
        tokenized_text = clean_text
    
    if not labeled:
        return "\n\n".join(tokenized_text.split("\n"))
    
    blk_pattern = re.compile("bb_extrato_bb((.|\n)*?)ee_extrato_ee")
    
    blk_found = re.search(blk_pattern, tokenized_text)
    
    while blk_found != None:
        contract = tokenized_text[blk_found.start():blk_found.end()]

        labeled_lines = [line for line in contract.split("\n") if ("bb_extrato_bb" not in line) and ("ee_extrato_ee" not in line)]
        labeled_lines = ["I " + line if idx != 0 else "B " + line for idx, line in enumerate(labeled_lines)]
        labeled_act = "\n".join(labeled_lines)

        tokenized_text = tokenized_text[:blk_found.start()] + labeled_act + tokenized_text[blk_found.end():]

        blk_found = re.search(blk_pattern, tokenized_text)
        
    tokenized_text = "\n\n".join(["O " + line if (line[0:2] != "I ") and (line[0:2] != "B ") else line for line in tokenized_text.split("\n")])
    
    return tokenized_text

def tokenize_dodfs_general(dodfs, labeled=True, token_type="sentence"):
    tokenized_dodfs = []
    dodfs = dodfs.sort_values(["year", "month", "file_name"]).reset_index(drop=True)
        
    for year in dodfs["year"].unique():
        temp_year = dodfs[dodfs["year"] == year]

        for month in temp_year["month"].unique():
            temp_month = temp_year[temp_year["month"] == month]

            tokenized_dodfs += list(temp_month.apply(lambda row: tokenize_text_dodf(row["marked_text"], labeled, token_type), axis=1))
            print("{}-{} feito.".format(year, month))
                        
    dodfs["tokenized_text"] = pd.Series(tokenized_dodfs)
    
    return dodfs

def label_dodf_by_contract(dodfs_raw, dodfs_contracts_raw, token_type):
    start_mark_contrato = "\nbb_extrato_bb\n"
    end_mark_contrato = "\nee_extrato_ee\n"
    
    dodfs = dodfs_raw.sort_values(["file_name", "page"]).reset_index(drop=True)
    dodfs = dodfs.groupby(by=["file_name", "number", "day", "month", "year"], as_index=False).agg({"text": "\n".join})
    
    dodfs = label_dodfs_contracts(dodfs, dodfs_contracts_raw, start_mark_contrato, end_mark_contrato)
    
    if token_type == "sentence":
        dodfs["marked_text"] = dodfs.apply(lambda row: get_running_text(row["marked_text"]), axis=1)
        
    dodfs["marked_text"] = clean_text_dodfs(dodfs["marked_text"])
    dodfs = tokenize_dodfs_general(dodfs, token_type=token_type)
    
    return dodfs.drop(columns=["marked_text"])

def format_dodf_generic(dodfs_raw, token_type):
    dodfs = dodfs_raw.sort_values(["file_name", "page"]).reset_index(drop=True)
    dodfs = dodfs.groupby(by=["file_name", "number", "day", "month", "year"], as_index=False).agg({"text": "\n".join})
    
    dodfs["marked_text"] = dodfs["text"].copy()
    
    if token_type == "sentence":
        dodfs["marked_text"] = dodfs.apply(lambda row: get_running_text(row["marked_text"]), axis=1)
        
    dodfs["marked_text"] = clean_text_dodfs(dodfs["marked_text"])
    dodfs = tokenize_dodfs_general(dodfs, labeled=False, token_type=token_type)
    
    return dodfs.drop(columns=["marked_text"])

raw_files_path = "./data/raw_full_dodfs/"
raw_files = os.listdir(raw_files_path)

treated_line_path = "./data/processed/line/"
treated_sentence_path = "./data/processed/sentence/"
treated_word_path = "./data/processed/word/"

for file in raw_files:
    dodfs_year_raw = pd.read_parquet(raw_files_path + file) 
    dodf_year_contratos_raw = ContractExtractorRaw.extract_text(dodfs_year_raw)
    
    dodf_year_labeled_sentence = label_dodf_by_contract(dodfs_year_raw, dodf_year_contratos_raw, "sentence")
    print("\nPor frase rotulado.")
    
    dodf_year_labeled_word = label_dodf_by_contract(dodfs_year_raw, dodf_year_contratos_raw, "word")
    print("\nPor palavra rotulado.")
    
    dodf_year_labeled_line = label_dodf_by_contract(dodfs_year_raw, dodf_year_contratos_raw, "line")
    print("\nPor linha rotulado.")

    print("\n\n{} feito.\n\n".format(file))
    
    dodf_year_labeled_sentence.to_parquet(treated_sentence_path + "labeled/" + "sentence_labeled_" + file)
    dodf_year_labeled_word.to_parquet(treated_word_path + "labeled/" + "word_labeled_" + file)
    dodf_year_labeled_line.to_parquet(treated_line_path + "labeled/" + "line_labeled_" + file)