s = ['Data de assinatura: 19/07/2019.', 'Assinatura: 23/07/2019.',
     'DATA DE ASSINATURA: 02/10/2018.', 'Data de assinatura: 27/03/2019.',
     'Data da Assinatura: 27/03/2019.', 'ASSINATURA: \n14/06/2019.',
     'Da data da assinatura: 09/08/2019.',' Da data da assinatura: 29 de agosto de 2018',
     'Data da Assinatura: xx de setembro de 2018.', 'Assinatura do Contrato: 10/07/2019.',
     'ASSINATURA: 11.04.2019.', 'Assinatura: XX/12/2021.',
     'Data: 6/22/2021; Vigência:', 'DA ASSINATURA: 22/07/2021.']

s2 = ['unidade_orcamentaria', 'unidade orcamentaria 01101',
      'unidade orçamentária 09135,', 'DOTAÇÃO ORÇAMENTÁRIA: I - Unidade: 26.205;',
      'Unidade Orçamentária: 18101.'
]

s3 = ['EXTRATO DE CONTRATO Nº 042745/2021', 'EXTRATO DE CONTRATO Nº 45400/2021',
      'EXTRATO DO CONTRATO CORRETORA SEGUROS BRB Nº 01/2021',
      'EXTRATOS DE CONTRATO CONTRATO Nº 9295.',
      'EXTRATO DO CONTRATO DE EXECUÇÃO DE OBRAS Nº 03/2020',
      'EXTRATO DO CONTRATO DE PRESTAÇÃO DE SERVIÇOS Nº 074/2020-SSP/DF',
      'EXTRATO DO CONTRATO DE PRESTAÇÃO DE SERVIÇOS Nº 25/2020',
      'EXTRATO DE CONTRATO CONTRATO Nº 9284.', 'EXTRATO DO CONTRATO No 37375/SEDICT/DF',
      'EXTRATO DO CONTRATO N° 044759/2021', 'EXTRATO DO CONTRATO DE PATROCINIO No 01/2019',
      'EXTRATO DO CONTRATO No 34/2019', 
      'EXTRATO DE CONTRATO DE PATROCÍNIO N° 00193-00001298/2021-51',
      'EXTRATO DE CONTRATO Nº 44.999/2021', 'EXTRATO DO CONTRATO Nº 44842/2021',
      'AVISO DE INEXIGIBILIDADE E EXTRATO DE CONTRATO + Contrato 2540.00006/2021',
      'CONTRATO DE PRESTACAO DE SERVICOS no 39.178/2019.',
      'Contrato nº 007/2021-CEBLajeado EXTRATO DE CONTRATO Nº 07/2021'
      'A CEB LAJEADO S/A, torna pública a assinatura do Contrato nº 007/2021-CEBLajeado',
      'EXTRATO DO CONTRATO DE SUBEMPRÉSTIMO Nº 0600.952',
      'EXTRATO DO CONTRATO DE AQUISIÇÃO DE PRODUTOS DA AGRICULTURA FAMILIAR Nº 42/2021',
      'EXTRATO DO CONTRATO PARA AQUISIÇÃO DE BENS PELO DISTRITO FEDERAL Nº 42/2021',
      'CONTRATO SIMPLIFICADO 062/2021-CJU/CEB-H',
      'EXTRATO DO CONTRATO CORRETORA SEGUROS BRB Nº 25/2021',
      'Extrato do Contrato no 16/2019', 'EXTRATO DE CONTRATO No 11/2019',
      'Contrato de patrocinio  + Contrato: 2019/001.',
      'EXTRATO DO CONTRATO BRB Nº 198/2021', 'EXTRATO DO CONTRATO Nº 25/2021'
]

s4 = ['VALOR: R$ 1.221.750,00', 'no valor de R$ 583.829,72',
      'Valor Estimado do Contrato: R$ 180.000,00.',
      'Valor total do contrato: R$ 39.815,91',
      'DO VALOR: O valor do Contrato é de R$ 4.675,00'
]

trecho_teste = """
8,"EXTRATOS DE CONTRATOS
EXTRATO DO CONTRATO DE PATROCINIO No 01/2019, celebrado entre a Companhia de
Planejamento do Distrito Federal - CODEPLAN e o Servico Social da Industria - Departamento Regional
do Distrito Federal - SESI/DF. Processo: 00121.00000.0897/2019-41. Objeto: Patrocinio do Concurso V
PREMIO CODEPLAN DE TRABALHOS TECNICO-CIENTIFICOS, acerca do Desenvolvimento do
Distrito Federal e Regiao Integrada de Desenvolvimento do Distrito Federal e Entorno-RIDE, conforme
Plano de Trabalho. Vigencia: O presente Contrato tera vigencia de 06 (seis) meses, a contar do dia
19/07/2019. O Patrocinador ira premiar os trabalhos vencedores com R$ 11.000,00 (onze mil reais),
correspondente a 50% do valor total do Premio. Data de assinatura: 19/07/2019. Assinam pela
CODEPLAN: Jeansley Charlles de Lima - Presidente, Pelo SESI: Marco Antonio Areias Secco - Diretor
Regional."
"""

# https://docs.python.org/3/library/re.html
# https://regex101.com/
import re

assinatura_rule = re.compile(r"(data d[e|a] assinatura.*\d{1,})|(assinatura\n*\d{1,})|(data assinatura.*\d{1,})|(assinatura.*\d{1,})|(data.*\d{1,}.*vig[e|ê]ncia)",
                             flags=re.IGNORECASE | re.UNICODE | re.DOTALL)

unidade_or_rule = re.compile(r"(unidade[ |_]or[c|ç]ament[a|á]ria.*\d{1,})|(unidade[ |_]orcamentaria)|(dota[c|ç][a|ã]o or[c|ç]ament[a|á]ria.*\d{1,})",
                             flags=re.IGNORECASE | re.DOTALL)

extrato_rule = re.compile(r"(extrato d[e|o] contrato.*\d{1,})|(contrato de presta[c|ç][a|ã]o de servi[c|ç]os.*\d{1,})|(contrato.*\d{1,})",
                          flags=re.IGNORECASE | re.DOTALL)

valor_rule = re.compile(r"(valor.*\d{1,})|(no valor de.*\d{1,})|(do valor.*valor.*\d{1,})", flags=re.IGNORECASE | re.DOTALL)

for phrase in s4:
    print(re.match(valor_rule, phrase))

for rule in [assinatura_rule, unidade_or_rule, extrato_rule, valor_rule]:
    print(re.search(rule, trecho_teste))