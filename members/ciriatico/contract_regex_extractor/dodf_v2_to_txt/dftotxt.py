import os
import pandas as pd

class DFtoTXT:
    
    @classmethod
    def extract_txts(cls, path, base_df):
        base_df = cls._clean_df(base_df)
        files_path = os.listdir(path)

        for year in base_df["year"].unique():
            path_year = os.path.join(path, str(year))

            if str(year) not in files_path:
                os.mkdir(path_year)

            temp_year = base_df[base_df["year"] == year].copy()

            for month in temp_year["month"].unique():
                path_month = os.path.join(path_year, str(month))
                files_path_year = os.listdir(path_year)

                temp_month = temp_year[temp_year["month"] == month].copy()

                if str(month) not in files_path_year:
                    os.mkdir(path_month)

                cls._df_to_txt(path_month, temp_month)
    
    @classmethod
    def _clean_df(cls, base_df):
        month_str = {
            1: "01_Janeiro",
            2: "02_Fevereiro",
            3: "03_Março",
            4: "04_Abril",
            5: "05_Maio",
            6: "06_Junho",
            7: "07_Julho",
            8: "08_Agosto",
            9: "09_Setembro",
            10: "10_Outubro",
            11: "11_Novembro",
            12: "12_Dezembro"
        }
        
        base_df_mod = cls._extract_pages_dodf(base_df)
        base_df_mod["file_name"] = base_df_mod["file_name"].str.split("/").str[0]
        base_df_mod["month"] = base_df_mod["month"].replace(month_str)
        base_df_mod = base_df_mod.sort_values(["year", "month", "file_name"]).reset_index(drop=True)
        
        return base_df_mod
        
    @classmethod
    def _extract_pages_dodf(cls, base_df):
        modified_df = base_df.sort_values(["file_name", "year", "month", "day", "page"]).reset_index(drop=True)
        modified_df = modified_df.groupby(["file_name", "number", "day", "month", "year"])["text"].apply('\n'.join).reset_index()

        modified_df = modified_df.drop(columns=["number", "day"])

        return modified_df
    
    @classmethod
    def _str_to_txt(cls, path, filename, text):
        with open(os.path.join(path, filename), "w") as text_file:
            text_file.write(text)
        return ""
    
    @classmethod
    def _df_to_txt(cls, path, base_df):
        base_df.apply(lambda row: cls._str_to_txt(path, row["file_name"]+".txt", row["text"]), axis=1)