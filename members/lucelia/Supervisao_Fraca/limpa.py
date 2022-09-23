import re
simbolos_pontuacao = ["'", '"', ",", ".", "!", ":", ";", '#', '@', '-']
texto = "EXTRATO DO CONTRATO PARA AQUISIÇÃO DE BENS PELO DISTRITO FEDERAL Nº 08/ 2022-SEJUS - SIGGO Nº 045498 Processo : 00400-00013706/ 2021-99 . Das Partes : SECRETARIA DE ESTADO DE JUSTIÇA E CIDADANIA DO DF X DANTON GABRIEL S. DE S. SILVA ME . Do Objeto : Aquisição de material : caixa acústica : caixa acústica ativa 15 1000w ; e microfone de lapela com as seguintes especificações : transdutor do tipo condensador ; padrão polar omnidirecional ; resposta de frequência 50 hz - 18khz ; sensibilidade de circuito aberto : -54 db . Do Valor do Contrato : O valor total do Contrato é de R$ 1.684,00 ( um mil seiscentos e oitenta e quatro reais ) , devendo a importância ser atendida à conta de dotações orçamentárias consignadas no orçamento corrente Lei Orçamentária nº 6.482 , de 09 de Janeiro de 2020 . DA DOTAÇÃO ORÇAMENTÁRIA : I - Unidade Orçamentária : 44.101 ; II - Programa de Trabalho 14.243.6211.2412. 0003-MANUTENÇÃO E FUNCIONAMENTO DO CENTRO DE ATENDIMENTO INTEGRADO À CRIANÇAS VÍTIMAS DE VIOLÊNCIA SEXUAL-DF.OCA ; III - Natureza da Despesa : 44.90.52 ; IV - Fonte de Recurso : 732020590 . O empenho inicial é de R$ 1.684,00 ( um mil seiscentos e oitenta e quatro reais ) , conforme Nota de Empenho nº 2022NE00394 , emitida em 11/03/2022 , sob o evento nº 400091 , na modalidade Ordinário . DATA DE ASSINATURA : 14/03/2022 . SIGNATÁRIOS : Pelo DISTRITO FEDERAL : JAIME SANTANA DE SOUSA , na qualidade de Secretário-Executivo , da Secretaria de Estado de Justiça do Distrito Federal . Pela CONTRATADA : DANTON GABRIEL SIMPLICIO DE SALES SILVA , na qualidade de Representante Legal ."

#for c in simbolos_pontuacao:
    #r = re.compile('({"|".join(map("\s+$", ""))})',texto)
#r = re.compile('({"|".join(map("\s+$", "")})')
#texto = r.sub('', texto)
#r = any(bool(re.search(r"\s", ele)) for ele in texto)
# =(re.compile(r"\s", "", ele) for ele in texto)
""" for c in texto:
    p = re.sub("\s", "", c) if c[i] in simbolos_pontuacao
    #p = (re.sub("\s", "", ele) for ele in texto) """
    #print(p) # Lorem ipsum dolor sit Amet blabl etc whiskas sache

def to_regex(s):
    #r=''
    #print(s)
    #if s in simbolos_pontuacao:
    #r=re.search("\s+\:")
    
    #return re.sub(r,"")
    r=''
    if (re.search("\s+\:", texto)):
	    r=re.sub(" ","", texto)
    return r
print(to_regex(texto))