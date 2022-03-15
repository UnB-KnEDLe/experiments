import json
import re
import os
import pandas as pd
from unidecode import unidecode

CLEANR = re.compile('<.*?>')
CLEANR2 = re.compile('&.*?;')

def primeira_parte():
    
    string = "<?xml version='1.0' encoding='UTF-8'?>\n"
    string = string+'<!DOCTYPE collection SYSTEM "BioC.dtd">\n'
    string = string+'<collection>\n'
    string = string+'\t<source>BC5CDR</source>\n'
    string = string+'\t<date></date>\n'
    string = string+'\t<key></key>\n'
    string = string+'\t<document>\n'
    
    return string
    
def ultima_parte():
    
    string ='\t</document>\n'
    string = string+'</collection>\n'
    
    return string

def escreve_publicacao(pub):
    
    string = '\t<passage>\n'
    string = string+'\t<text>\n'
    
    string = string+"\t\t"+pub+"\n"
    
    string = string+'\t</text>\n'
    string = string+'\t</passage>\n'
    
    return string

def write_biocxml(nome_arq,edicao):
    
    output_file = open(nome_arq,'w', encoding="utf-8")
    
    output_file.write(primeira_parte())
    
    for publicacao in edicao:
        output_file.write(escreve_publicacao(publicacao))
    
    output_file.write(ultima_parte())
    
    output_file.close()

def write_csv(filename,orgao,publicacao):
    zipped = list(zip(orgao,publicacao))
    df = pd.DataFrame(zipped, columns=['Orgao','Publicacao'])
    
    df.to_csv(filename, sep='\t',index=False)  

def atos_validos(titulo_pub):
    
    lista = ["aviso","contrato","alteração","extrato","convênio","convênios", "contratual","convenio","convenios", "licitação", "licitacao", "abertura","pregão", "retificação","pregao","revogação","revogacao","suspensão","suspensao","anulacao","anulação","aditivo", "instrumento contratual", "instrumentos contratuais"]
    
    lista2 = ["homologação","rescisão","ratificação","alteracao","alteração","inabilitação","plano","divulgação","divulgacao","chamamento","convocação","dispensa","relatório","relatorio","convocacao"]
       
    palavras = titulo_pub.lower().split()
    
    for word in lista2:
        if word in palavras:
            return False,'outro'
    
    qt = 0
    
    lista_saida = []
    for word in lista:
        if word in palavras:
            lista_saida.append(word)
            qt+=1
    
    tipo = 'outro'
    for i in lista_saida:
        if i == "aditivo":
            tipo = 'aditamento_contrato'
            break
        elif i == "contrato" or i == "contratos" or i == "contratual" or i == "instrumento contratual" or i == "instrumentos contratuais":
            tipo = 'extrato_contrato'
            break
        elif i == "convênio" or i == "convenio" or i == "convênios" or i == "convenios":
            tipo = 'extrato_convenio'
            break
        elif i == "licitacao" or i == "licitação" or i == "licitaçao" or i == "licitacão" or i == "pregão" or i == "pregao" or i == "abertura" or i == "retificacao" or i == "retificaçao" or i == "retificação":
            tipo = 'aviso_licitacao'
            break
        elif i == "revogacao" or i == "revogação" or i == "revogacão" or i == "revogaçao" or i == "anulacao" or i == "anulação" or i == "anulacão" or i == "anulaçao":
            tipo = 'aviso_revogacao_anulacao_licitacao'
            break
        elif i == "suspensao" or i == "suspensão":
            tipo = 'aviso_suspensao_licitacao'
            break
        else:
            pass
    
    if qt >= 2:
        return True,tipo
    
    return False,tipo

def cleanhtml(raw_html):
    
    cleantext = re.sub(CLEANR, '\n', raw_html)
    cleantext = re.sub(CLEANR2, '', cleantext)
    return cleantext

def is_pua_codepoint(c):
    
    pua_ranges = ( (0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD) )

    return any(a <= c <= b for (a,b) in pua_ranges)

def remove_PrivateUserArea(string):
    
    output = ""

    for i in range(0,len(string)):
        if ord(string[i])>1000:
            output=output+' '
        else:
            output=output+string[i]
            
    return output
            
    

json_path = "dodf_jsons/"
xml_path = "dodf_xmls/"
csv_path = "dodf_csvs/"

edicoes_json = [publicacao_xml for publicacao_xml in os.listdir(json_path) if os.path.isfile(os.path.join(json_path,publicacao_xml))]

publi_str = []
orgao_str = []

histograma = {}

outros = []

titulos = []

for edicao_json in edicoes_json:

    json_filename = json_path+edicao_json
    xml_filename = xml_path+edicao_json[:-5]+'.xml'
    csv_filename = csv_path+edicao_json[:-5]+'.csv'
    
    f_in = open(json_filename,'r', encoding="utf-8")
    
    data = json.load(f_in)

    print("Processing "+edicao_json+" ...")
    secaoIII = data['json']['INFO']['Seção III']

    edicao = []

    for orgao in secaoIII:
        
        i = 0

        for doc in secaoIII[orgao]['documentos']:
            
            for pub in secaoIII[orgao]['documentos'][doc]:
                
                titulo = secaoIII[orgao]['documentos'][doc]['titulo']
                i+=1
                
                ehvalido,tipo_ato = atos_validos(titulo)
                
                if (titulo not in titulos) and ehvalido == True:
                    
                    if tipo_ato not in histograma:
                        histograma[tipo_ato] = 0
                    else:
                        histograma[tipo_ato] += 1 
                    
                    if i == 1:
                        edicao.append("\n"+orgao)
                    
                    orgao_str.append(orgao)
                    
                    pub_xml = remove_PrivateUserArea(titulo)
                    pub_xml = pub_xml+remove_PrivateUserArea(cleanhtml(secaoIII[orgao]['documentos'][doc]['texto']))
                    #pub_xml = pub_xml.encode("ascii", "ignore")
                    #pub_xml = pub_xml.decode()
                    edicao.append(pub_xml)
                    titulos.append(remove_PrivateUserArea(titulo))
                    if tipo_ato == "outro":
                        outros.append(pub_xml)

    write_biocxml(xml_filename,edicao)
    
    write_csv(csv_filename,orgao_str,edicao)
    
    f_in.close()
    
    print(histograma)
    print()
    print()
    
#print(histograma)

#for i in outros:
#    print(i+"\n")
