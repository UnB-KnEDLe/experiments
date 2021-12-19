import pandas as pd
import re
from pandas import DataFrame
import numpy as np


def dfa_licitacao():
    regex1 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+ABERTURA\s+D[EO]\s+LICITA[CÇ][AÃ]O|AVISO\s+D[EO]\s+ABERTURA)'
    regex2 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+ADJUDICAC[CÇ][AÃ]O\s+E\s+HOMOLOGA[CÇ][AÃ]O|AVISO\s+D[EO]\s+HOMOLOGA[CÇ][AÃ]O\s+E\s+ADJUDICA[CÇ][AÃ]O)'
    regex3 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+HOMOLOGA[CÇ][AÃ]O\s+E\s+CONVOCA[CÇ][AÃ]O)'
    regex4 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+CONVOCA[CÇ][AÃ]O)'
    regex5 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+DECLARA[CÇ][AÃ]O\s+D[EO]\s+VENCEDOR)'
    regex6 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+RESULTADO)'
    regex7 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+RESULTADO\s+D[EO]\s+JULGAMENTO)'
    regex8 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+JULGAMENTO)'
    regex9 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+LICITA[CÇ][AÃ]O)'
    regex10 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+JULGAMENTO\s+D[EO]\s+HABILITA[CÇ][AÃ]O)'
    regex11 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+RESULTADO\s+D[EO]\s+RECURSO\s+E\s+JULGAMENTO)'
    regex12 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+SUSPENS[AÃ]O\s+D[EO]\s+LICITA[CÇ][AÃ]O)'
    regex13 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+ADIAMENTO\s+D[EO]\s+LICITA[CÇ][AÃ]O)'
    regex14 = r'(?:xxbcet\s+)?(?:AVISO\s+D[EO]\s+ALTERA[CÇ][AÃ]O)'
    regex15 = r'(?:xxbcet\s+)?(?:EXTRATO\s+D[EO]\s+CONTRATO)'
    regex16 = r'(?:xxbcet\s+)?(?:AVISOS?\s+D[EO]\s+REABERTURA)'
    regex17 = r'(?:xxbcet\s+)?(?:AVISOS?\s+D[EO]\s+NOVA\s+DATA\s+D(?:E|A)\s+ABERTURA)'

    regex_s = r'(?:xxbcet\s+)(?:AVISOS?|EXTRATOS?|RESULTADOS?|SECRETARIA|SUBSECRETARIA |PREGÃO|TOMADA|COMISSÃO|DIRETORIA|ATO)'

    acts = []
    l_acts = []
    del l_acts
    l_acts = []
    ato = False

    for texto in dodfs:
        i = 0
        # print(len(texto))
        while i != len(texto):
            if re.match(regex1, texto[i]):
                l_acts.append('[1] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 2
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex2, texto[i]):
                l_acts.append('[2] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 2
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex3, texto[i]):
                l_acts.append('[3] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 2
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex4, texto[i]):
                l_acts.append('[4] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 2
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex5, texto[i]):
                l_acts.append('[5] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 2
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex7, texto[i]):
                l_acts.append('[7] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex11, texto[i]):
                l_acts.append('[11] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]
            
            elif re.match(regex6, texto[i]):
                l_acts.append('[6] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex9, texto[i]):
                l_acts.append('[9] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex10, texto[i]):
                l_acts.append('[10] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex8, texto[i]):
                l_acts.append('[8] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex12, texto[i]):
                l_acts.append('[12] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex13, texto[i]):
                l_acts.append('[13] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex14, texto[i]):
                l_acts.append('[14] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex15, texto[i]):
                l_acts.append('[15] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex16, texto[i]):
                l_acts.append('[16] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            elif re.match(regex17, texto[i]):
                l_acts.append('[17] ' + texto[0] + texto[i])
                ato = True
                while ato:
                    i += 1
                    if i == len(texto):
                        break
                    if re.match(regex_s, texto[i]) and ('xxbob' in texto[i-1] or('—' in texto[i-1] and 'xxbob' in texto[i-2])):
                        i -= 1
                        break
                    else:
                        l_acts[-1] += '\n' + texto[i]

            else:
                i+=1

    return l_acts


first = True
cont = 0

for chunk in pd.read_csv("dodfs.csv", chunksize=1):
    
    print(cont)

    chunk = chunk.reset_index(drop=True)

    dodfs = []

    for row in range(len(chunk)):
        dodfs.append(chunk['Dodfs_list'][row].split('\n'))


    acts = dfa_licitacao()


    if first:
        df_acts = DataFrame(acts,columns=['Atos'])
        df_acts.to_parquet('all_acts_with_tags.parquet.gzip', compression='gzip')
        first = False
        del acts, dodfs, df_acts

    else:
        data = DataFrame(acts,columns=['Atos'])
        df_acts = pd.read_parquet("all_acts_with_tags.parquet.gzip", engine='fastparquet') 
        df_acts = pd.concat([df_acts, data])
        df_acts = df_acts.reset_index(drop=True)
        df_acts.to_parquet('all_acts_with_tags.parquet.gzip', compression='gzip') 

        del acts, dodfs, data, df_acts

    cont += 1
