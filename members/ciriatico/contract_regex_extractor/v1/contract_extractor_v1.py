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
        
        base_df = cls._read_file(file_path, file_type)
        
        contract_pattern = r"(\nxx([a-z]{0,10})\sEXTRAT([A-Z]{0,3})\sD([A-Z]{0,3})\sCONTRAT([A-Z]{0,3}))"
        p_ext = re.compile(contract_pattern)

        block_pattern = r"(\nxx([a-z]{0,10})\s([A-ZÀ-Ú0-9º\/x\-\.\,\(\)\|\*&\s\']{5,}) xx([a-z]{0,10}))"
        block_pattern = block_pattern + "|" + contract_pattern
        p_blk = re.compile(block_pattern)
        
        blocks_extracted = cls._extract_text_blocks(base_df, p_ext, p_blk)
        
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
    def _extract_text_blocks(cls, base_df, contract_pattern, block_pattern):
        base_df = base_df.sort_values(["file_name", "page"])

        # Concatenar todas as páginas em uma única linha
        base_df = base_df.groupby(["file_name", "number", "day", "month", "year"])["text"].apply('\n'.join).reset_index()

        matched_texts = base_df.apply(lambda row: cls._row_list_regex(row.name, row["text"], contract_pattern, block_pattern), axis=1)
        matched_texts = [matched_text for matched_text in matched_texts if matched_text != None]

        ext_blk_list = cls._mapped_positions_regex(matched_texts)

        extracted_texts_df = cls._extract_texts_from_mapped_positions(ext_blk_list, base_df, "text")

        ext_df = cls._extracted_texts_to_df(extracted_texts_df, base_df[["file_name", "number", "day", "month", "year"]], ["file_name", "number", "day", "month", "year", "text"])
    
        return ext_df
    
    @classmethod
    def _clean_text(cls, ext_df, start_page_patterns, end_page_patterns, middle_page_patterns):
        start_page_patterns = "|".join(start_page_patterns)
        middle_page_patterns = "|".join(middle_page_patterns)
        end_page_patterns = "|".join(end_page_patterns)

        page_patterns = [start_page_patterns, middle_page_patterns, end_page_patterns]
        page_patterns = "|".join(page_patterns)

        ext_df["text"] = ext_df["text"].str.replace(page_patterns, "", regex=True)

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
    def _row_list_regex(cls, index, text, pattern_ext, pattern_blk):
        # Lista com as buscas por regex. O findall permite ver se no documento há pelo menos 1 extrato detectado
        row_list = [index, re.findall(pattern_ext, text), re.finditer(pattern_ext, text), re.finditer(pattern_blk, text)]

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

            a_ext_list = []
            a_blk_list = []

            for i in a_ext:
                a_ext_list.append(i.start())

            for i in a_blk:
                a_blk_list.append(i.start())

            mapped_positions_ext_blk.append([match[0], a_ext_list, a_blk_list])

        return mapped_positions_ext_blk

    @classmethod
    def _extract_texts_from_mapped_positions(cls, mapped_positions, base_df, text_column):
        extracted_texts = []

        for mapped_position in mapped_positions:
            mapped_text = base_df.loc[mapped_position[0]][text_column]

            for ia in mapped_position[1]:
                # Um texto começa no bloco detectado pelo regex para extrato e termina no próximo bloco qualquer detectado
                index_ia = mapped_position[2].index(ia)

                if (index_ia + 1) <= (len(mapped_position[2])-1):
                    ib = mapped_position[2][index_ia+1]
                    extracted_text = mapped_text[ia:ib]
                else:
                    extracted_text = mapped_text[ia:]

                extracted_texts.append([mapped_position[0], extracted_text])

        return extracted_texts

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