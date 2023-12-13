import pandas as pd
import json
import requests
import os
from bs4 import BeautifulSoup
import unidecode
from datetime import datetime


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
            # return pd.DataFrame(self.read_from_json(), columns = ['link_dodf', 'date', 'nro', 'orgao', 'section', 'name', 'type_act', 'raw_txt'])
            df = pd.DataFrame(self.read_from_json(), columns = ['linkJornal', 'dt_previsao_publicacao', 'nu_numero', 'orgao', 'secao', 'titulo', 'tipo', 'texto'])
            df['processo'] = df['texto'].str.extract(r'''(?:(?:(?:P|p)rocesso(?:\s+)?(?:(?:\()?SEI(?:\)?))?(?:\s+)?(?:(?:no|n\.o)?)?)|(?:P|p)rocesso:|(?:P|p)rocesso|Processo.|(?:P|p)rocesso\s+no|(?:P|p)rocesso\s+n.? ?o.?|(?:P|p)rocesso\s+no:|(?:P|p)rocesso\s+SEI\s+no:|(?:P|p)rocesso\s+SEI:|(?:P|p)rocesso\s+nº|(?:P|p)rocesso:?\s*—?\s*-?|(?:P|p)rocesso\s+(?:N|n).?º?|(?:P|p)rocesso\s+SEI-GDF:|(?:P|p)rocesso\s+SEI-GDF|(?:P|p)rocesso\s+SEI\s+no|(?:P|p)rocesso\s+SEI\s+n|(?:P|p)rocesso\s+SEI|(?:P|p)rocesso-\s+SEI|(?:P|p)rocesso\s+SEI\s+no.|(?:P|p)rocesso\s+\(SEI\)\s+no.|(?:P|p)rocesso\s+SEI\.|(?:P|p)rocesso\s+\(SEI-DF\)\s+no.|(?:P|p)rocesso\s+SEI-GDF no|(?:P|p)rocesso\s+n|(?:P|p)rocesso\s+N|(?:P|p)rocesso\s+administrativo no|(?:P|p)rocesso\s+-\s+de\s+Licitação\s+n.º|(?:P|p)rocesso\s+n:|PROCESSO ?: ?N?o?|PROCES-? ?SO|PROCESSO.|PROCESSO\s+no|PROCESSO\s+No|PROCESSO\s+N.o:?|PROCESSO\s+no.|PROCESSO\s+no:|PROCESSO\s+No:|PROCESSO\s+SEI\s+no:|PROCESSO N.?º?:?|PROCESSO\s+SEI:|PROCESSO\s+SEI|PROCESSO\s+SEI-GDF:|PROCESSO\s+SEI-GDF|PROCESSO\s+SEI\s+no|PROCESSO\s+SEI\s+No|PROCESSO\s+SEI\s+no.|PROCESSO\s+SEI.|PROCESSO\s+TCB\s+(?:N|n).?º?)((?:(?!\s\d{2}.\d{3}.\d{3}/\d{4}-\d{2}))(?:(?:\s*)(?:(?:[\d.]+)|(?:[\d\s,]+))[.-]?(?:(?:\d)|(?:[.\d\sSEI-|]+))(?:/|-
\b)(?:(?:(?:\d)+|(?:[\d\s]+)))?(?:-(?:(?:\d)+|(?:[\d\s]+)))?(?:-SECOM/DF|-?CEB|/CBMDF|F J Z B / D F)?))''').replace(r'[^0-9]', '', regex=True)
            return df
        if self.dodf_file[self.dodf_file.index('.'):] == '.txt':
            pass


    def parse_html_text(self, text):
        if text is not None:
            return unidecode.unidecode(BeautifulSoup(text, 'html.parser').get_text())
        return text


    def read_from_json(self):
        if os.path.isfile(self.dodf_file):
            r_json = json.load(open(self.dodf_file))["json"]
        else:
            try:
                r_json = requests.get("http://164.41.76.30/dodf_watcher/" + self.dodf_file[-24:]).json()["json"]
            except:
                return []

        dodf_obj = []
        section = "Seção III"

        if section in r_json["INFO"]:
            for orgao in r_json["INFO"][section]:
                for doc in r_json["INFO"][section][orgao]["documentos"]:
                    acts_info = r_json["INFO"][section][orgao]["documentos"][doc]
                    data = datetime.strptime(r_json['dt_previsao_publicacao'][:10], '%Y-%m-%d').strftime("%d/%m/%Y")
                    dodf_obj.append([DODF(r_json["linkJornal"], data, r_json['nu_numero']), \
                                      Acts(orgao, section, acts_info["titulo"], acts_info["tipo"], self.parse_html_text(acts_info["texto"]))])

        dodf_acts = []
        for obj1, obj2 in dodf_obj:
            dodf_acts.append([obj1.link_dodf, obj1.date, obj1.nro, obj2.orgao, obj2.section, obj2.name, obj2.type_act, obj2.raw_txt])

        return dodf_acts
