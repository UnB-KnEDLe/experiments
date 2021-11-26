import pandas as pd
import re
import json
import os

RESULTS_PATH = "results/"

class ContractExtractor:
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

        # Padrões de começo e fim de página abarcam apenas os padrões observados entre 2000 e 2021
        start_page_patterns = [r"\nPÁGINA\s([0-9]{1,5})", r"\nDIÁRIO\sOFICIAL\sDO\sDISTRITO\sFEDERAL",
                               r"\nNº(.+?)2([0-9]{3})", r"\nxx([a-z]{0,10}) Diário Oficial do Distrito Federal xx([a-z]{0,10})",
                               r"\nDiário Oficial do Distrito Federal"]

        end_page_patterns = [r"Documento assinado digitalmente conforme MP nº 2.200-2 de 24/08/2001, que institui a",
                                r"Infraestrutura de Chaves Públicas Brasileira ICP-Brasil",
                                r"Este documento pode ser verificado no endereço eletrônico",
                                r"http://wwwin.gov.br/autenticidade.html",
                                r"pelo código ([0-9]{15,18})",
                                r"\nDocumento assinado digitalmente, original em https://www.dodf.df.gov.br"]

        middle_page_patterns = [r"xx([a-z]{1,10}) ", r" xx([a-z]{1,10})", r"\n-\n",
                             r"xx([a-z]{1,10})", r"\n- -\n", r"\n- - -\n",
                            r"\n[\.\,\-\—]\n", r"— -"]

        contract_texts = cls._clean_text(blocks_extracted, start_page_patterns, end_page_patterns, middle_page_patterns)

        return contract_texts
    
    @classmethod
    def extract_to_file(cls, read_path_file, read_file_type="dataframe", folder="./", save_file_type="parquet"):
        """Extract texts of contract statements from dataframe or file
            and save them in file (PARQUET, CSV, JSON or TXT)

        Args:
            read_path_file: The path of the file that stores the dataframe.
            read_file_type: The type of the file that is being read. It can be
            passed directly a dataframe or a path to a file (PARQUET, TXT or
            CSV).
            folder: Folder where the file will be written.
            save_file_type: type of the file that will be exported (it can be
            PARQUET, JSON, TXT or CSV).

        Returns:
            Dataframe in which each row has a contract statement, with additional
            informations such as page, year, month and day of the DODF from which
            the text was extracted.
        """
        
        contract_texts = cls.extract_text(read_path_file, read_file_type)
        cls._save_single_file(contract_texts, folder, save_file_type)
        
        return contract_texts
    
    @classmethod
    def _extract_text_blocks(cls, base_df, contract_pattern, block_pattern, newline_pattern):
        base_df = base_df.sort_values(["file_name", "page"])

        # Concatenar todas as páginas em uma única linha
        base_df = base_df.groupby(["file_name", "number", "day", "month", "year"])["text"].apply('\n'.join).reset_index()

        matched_texts = base_df.apply(lambda row: cls._row_list_regex(row.name, row["text"], contract_pattern, block_pattern, newline_pattern), axis=1)
        matched_texts = [matched_text for matched_text in matched_texts if matched_text != None]

        ext_blk_list = cls._mapped_positions_regex(matched_texts)

        extracted_texts_df = cls._extract_texts_from_mapped_positions(ext_blk_list, base_df, "text")

        ext_df = cls._extracted_texts_to_df(extracted_texts_df, base_df[["file_name", "number", "day", "month", "year"]], ["file_name", "number", "day", "month", "year", "full_text", "title_text", "corpus_text"])

        return ext_df
    
    @classmethod
    def _clean_text(cls, ext_df, start_page_patterns, end_page_patterns, middle_page_patterns):
        start_page_patterns = "|".join(start_page_patterns)
        middle_page_patterns = "|".join(middle_page_patterns)
        end_page_patterns = "|".join(end_page_patterns)

        page_patterns = [start_page_patterns, middle_page_patterns, end_page_patterns]
        page_patterns = "|".join(page_patterns)

        ext_df["full_text"] = ext_df["full_text"].str.replace(page_patterns, "", regex=True)
        ext_df["title_text"] = ext_df["title_text"].str.replace(page_patterns, "", regex=True)
        ext_df["corpus_text"] = ext_df["corpus_text"].str.replace(page_patterns, "", regex=True)

        return ext_df

    @classmethod
    def _extracted_texts_to_df(cls, texts_indexes, base_df, column_names):
        # Gera dataframe com os textos extraídos, com informações do documento onde foram encontrados
        extracted_texts_df = []

        for extracted_text in texts_indexes:
            index_text = extracted_text[0]
            ext_text = extracted_text[1]
            title_text = extracted_text[2]
            corpus_text = extracted_text[3]

            list_text = list(base_df.loc[index_text, base_df.columns].values)
            list_text.append(ext_text)
            list_text.append(title_text)
            list_text.append(corpus_text)

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

                index_ia_e_nl = mapped_position[3].index(ia_e) + 1
                inl = mapped_position[3][index_ia_e_nl]
                title_text = mapped_text[ia_b:inl]

                if (index_ia_e + 1) <= (len(mapped_position[2])-1):
                    ib = mapped_position[2][index_ia_e+1]
                    corpus_text = mapped_text[inl:ib]
                else:
                    corpus_text = mapped_text[inl:]

                extracted_text = title_text + corpus_text

                extracted_texts.append([mapped_position[0], extracted_text, title_text, corpus_text])

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