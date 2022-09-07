import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup
import unidecode


class Acts:
    def __init__(self, orgao, section, name, type_act, raw_txt):
        self.orgao = orgao
        self.section = section
        self.name = name
        self.type_act = type_act
        self.raw_txt = raw_txt


class DODF:
    def __init__(self, link_dodf, date, nro):
        # self.list_of_acts = list_of_acts
        self.link_dodf = link_dodf
        self.date = date
        self.nro = nro

    
class DODFCorpus:
    def __init__(self, dodf_file):
        self.dodf_file = dodf_file


    def read(self):
        if self.dodf_file[self.dodf_file.index('.'):] == '.json':
            return pd.DataFrame(self.read_from_json(), columns = ['link_dodf', 'date', 'nro', 'orgao', 'section', 'name', 'type_act', 'raw_txt'])
        if self.dodf_file[self.dodf_file.index('.'):] == '.txt':
            return self.read_from_txt()


    def parse_html_text(self, text):
        if text is not None:
            return unidecode.unidecode(BeautifulSoup(text, 'html.parser').get_text())
        return text


    def read_from_json(self):
        dodf = open(self.dodf_file)
        dodf = json.load(dodf)

        dodf_obj = []
        section = "Seção III"

        r_json = dodf["json"]
        if section in r_json["INFO"]:
            for orgao in r_json["INFO"][section]:
                for doc in r_json["INFO"][section][orgao]["documentos"]:
                    acts_info = r_json["INFO"][section][orgao]["documentos"][doc]
                    dodf_obj.append([DODF(r_json["linkJornal"], re.search(r'\d{2}-\d{2}-\d{4}', self.dodf_file).group(), re.search(r'\d{3}', self.dodf_file).group()), \
                                      Acts(orgao, section, acts_info["titulo"], acts_info["tipo"], self.parse_html_text(acts_info["texto"]))])

        dodf_acts = []
        for obj1, obj2 in dodf_obj:
            dodf_acts.append([obj1.link_dodf, obj1.date, obj1.nro, obj2.orgao, obj2.section, obj2.name, obj2.type_act, obj2.raw_txt])

        return dodf_acts
