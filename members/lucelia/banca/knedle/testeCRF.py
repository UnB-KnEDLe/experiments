# CODIGO DE APLICACAO DO TEINARMENTO PARA MODELO CRF

import dodfCRF
import dodfCRFMEU

import re
import pandas as pd


df = pd.read_csv("/home/lucelia_vieira/Experimentos/knedle/contratosFINAL.csv")

print("LEU DF")

crf = dodfCRF.CRFContratos()

crf.init_model_lbfgs()

print("INIT MODEL-lbfgs")

crf.train_model(df)

print("TRAIN MODEL")

crf.save_model("contratosFINAL")

print("SAVE MODEL")

# #################################################################################

# df = pd.read_csv("contratosFINAL.csv")

# print("LEU DF")

# crf = dodfCRFMEU.CRFContratos()

# crf.init_model_lbfgs()

# print("INIT MODEL-lbfgs")

# crf.train_model(df)

# print("TRAIN MODEL")

# crf.save_model("contratosMEU")

# print("SAVE MODEL")

# #################################################################################

# df = pd.read_csv("contratos.csv")

# print("LEU DF")

# crf = dodfCRF.CRFContratos()

# crf.init_model_l2sgd()

# print("INIT MODEL-l2sgd")

# crf.train_model(df)

# print("TRAIN MODEL")

# crf.save_model("contratos_l2sgd")

# print("SAVE MODEL")

#################################################################################

txt = ["EXTRATO DO CONTRATO Nº 13/2014 Processo 410.000.434/2014 – DAS PARTES:SEPLAN x SERVEGEL – APOIO ADMINISTRATIVO E SUPORTE OPERACIONAL LTDA. DO PROCEDIMENTO: O presente Contrato obedece aos termos do EDITAL DE LICITAÇÃO DE PREGÃO ELETRONICO N.º 151/ 2012 – SULIC/SEPLAN (fls. 03 a 149) oriundo do processo licitatório autuado sob nº 411.000.023/ 2012. DO OBJETO: O Contrato tem por objeto a contratação de empresa especializada para realizar a prestação de serviços continuados de limpeza asseio e conservação nos próprios do Governo do Distrito Federal com fornecimento de mão-de-obra materiais e equipamentos conforme especificações e condições estabelecidas no Anexo I do Edital LOTE 11 consoante especificam o EDITAL DE LICITAÇÃO DE PREGÃO ELETRONICO N.º 151/2012 – SULIC/SEPLAN (fls. 03 a 149) oriundo do processo licitatório autuado sob nº 411.000.023/2012. DO VALOR: O valor total do contrato é de R$ 5.645.716,32 >(cinco milhões seiscentos e quarenta e cinco mil setecentos e dezesseis reais e trinta e dois centavos). DA DOTAÇÃO ORÇAMENTÁRIA: A despesa correrá à conta da seguinte Dotação Orçamentária: I – Unidade Orçamentária: 32.101; II – Programa de Trabalho: 04.122.6003.2990.0006; III – Natureza da Despesa: 3.3.90.37; IV – Fonte de Recursos: 100; V – Nota de Empenho: 2014NE00621. DO PRAZO DE VIGÊNCIA: O contrato terá vigência de 12 (doze) meses a contar da data de sua assinatura. DA ASSINATURA: 08/05/2014. DOS SIGNATÁRIOS: Pela SEPLAN: Paulo Antenor de Oliveira na qualidade de Secretário de Estado e pela CONTRATADA: Marcelo Henry Soares Monteiro, na qualidade de Sócio-Proprietário da Empresa", "EXTRATO DO CONTRATO No 070/2019 Processo: 00392-00010412/2018-74 Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ: 09.335.575/0001-30; Contratada: EXSO Servicos de Engenharia LTDA, inscrita sob o CNPJ: 19.794.877/0001-20. Objeto: O contrato tem por objeto o credenciamento de pessoa juridica, devidamente registrada no CREA ou CAU, para prestar servicos em carater emergencial para construcao de muro e residencia na localidade da Vila Sao Jose em Vicente Pires, nos termos e condicoes estabelecidos no Edital de Credenciamento no. 001/2018 - CODHAB/DF e seus cadernos, que o integram e complementam, sempre que houver interesse previamente manifestado pela CODHAB. Fundamentacao Legal: Credenciamento no 001/2018 - CODHAB/DF. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.3571.0006. Natureza da Despesa: 33.90.39. Fonte: 100. Nota de Empenho no valor de R$ 99.848,75 (noventa e nove mil oitocentos e quarenta e oito reais e setenta e cinco centavos), conforme Nota de Empenho no 2019NE01126, emitida em 20/12/2019. Valor do Contrato de R$ 99.848,75 (noventa e nove mil oitocentos e quarenta e oito reais e setenta e cinco centavos). Modalidade: Global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 60 (sessenta) dias. Signatarios: Pela CODHAB - Wellington Luiz de Souza Silva, como Diretor-Presidente; Pela Contratada: Marcos Brasiliense Pimentel Barros, como representante Legal.<EOB>", " EXTRATO DO CONTRATO No 072/2019 Processo: 00392-00011605/2019-23 - Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ 09.335.575/0001-30; Contratada: PLUGAR MANUTENCAO E REFORMA LTDA - EPP - CNPJ 22.223.664/0001-52. Objeto: execucao de 05 (cinco) modulos estruturais de interesse social - conforme descrito no item 4 do Projeto Basico, denominados , a serem construidos na Regiao Administrativa de Samambaia RA - XII. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.1213.0906. Natureza da Despesa: 44.90.51. Fonte: 107. Nota de Empenho 2019NE01146, emitida em 27/12/2019 no valor de R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Valor do Contrato: R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Modalidade: Global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 06 (seis) meses. Signatarios: Pela CODHAB/DF: Wellington Luiz de Souza Silva, na qualidade de Diretor-Presidente; Pela Contratada: George Alexandre Campos, na qualidade de Representante Legal.<EOB>",
       "EXTRATO DO CONTRATO No 073/2019 Processo: 00392-00011606/2019-78 - Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ 09.335.575/0001-30; Contratada: WR COMERCIAL DE ALIMENTOS E SERVICOS LTDA - EPP - CNPJ 06.091.937/0001-17. Objeto: execucao de 05 (cinco) modulos estruturais de interesse social - conforme descrito no item 4 do Projeto Basico, denominados , a serem construidos na Regiao Administrativa de Samambaia RA - XII. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.1213.0906. Natureza da Despesa: 44.90.51. Fonte: 107. Nota de Empenho 2019NE01145, emitida em 27/12/2019 no valor de R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Valor do Contrato: R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Modalidade: global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 06 (seis) meses. Signatarios: Pela CODHAB/DF: Wellington Luiz de Souza Silva, na qualidade de Diretor-Presidente; Pela Contratada: Renato Marinho Araujo, na qualidade de Representante Legal.<EOB>", "EXTRATO DO CONTRATO No 074/2019 Processo: 00392-00011607/2019-12 - Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ 09.335.575/0001-30; Contratada: GALAXY ENGENHARIA EIRELI - CNPJ 25.451.351/0001-40. Objeto: execucao de 05 (cinco) modulos estruturais de interesse social - conforme descrito no item 4 do Projeto Basico, denominados , a serem construidos na Regiao Administrativa de Samambaia RA - XII. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.1213.0906. Natureza da Despesa: 44.90.51. Fonte: 107. Nota de Empenho 2019NE01144, emitida em 27/12/2019 no valor de R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Valor do Contrato: R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Modalidade: global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 06 (seis) meses. Signatarios: Pela CODHAB/DF: Wellington Luiz de Souza Silva, na qualidade de Diretor-Presidente; Pela Contratada: Leonardo Vinicius Sousa Reis, na qualidade de Representante Legal.<EOB>", "EXTRATO DO CONTRATO No 075/2019 Processo: 00392-00011608/2019-67 - Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ 09.335.575/0001-30; Contratada: CONSTRUTORA BRASIL INTEGRAL EIRELI - CNPJ 20.710.789/0001-81. Objeto: execucao de 05 (cinco) modulos estruturais de interesse social - conforme descrito no item 4 do Projeto Basico, denominados , a serem construidos na Regiao Administrativa de Samambaia RA - XII. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.1213.0906. Natureza da Despesa: 44.90.51. Fonte: 107. Nota de Empenho 2019NE01143, emitida em 27/12/2019 no valor de R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Valor do Contrato: R$ 332.085,90 (trezentos e trinta e dois mil, oitenta e cinco reais e noventa centavos). Modalidade: global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 06 (seis) meses. Signatarios: Pela CODHAB/DF: Wellington Luiz de Souza Silva, na qualidade de Diretor-Presidente; Pela Contratada: Marlos Augusto de Oliveira, na qualidade de Representante Legal.<EOB>", "EXTRATO DO CONTRATO No 076/2019 Processo: 00392-00010286/2019-39 - Contratante: Companhia de Desenvolvimento Habitacional do Distrito Federal - CODHAB/DF - CNPJ 09.335.575/0001-30; Contratada: QUARTZ CONSTRUCOES E SERVICOS DE REFORMAS LTDA - CNPJ 12.886.045/0001-94. Objeto: execucao de 04 (quatro) modulos estruturais de interesse social - conforme descrito no item 4 do Projeto Basico, denominados , a serem construidos na Regiao Administrativa de Samambaia RA - XII. Dotacao Orcamentaria: UO 28.209. Programa de Trabalho: 16.482.6208.1213.0906. Natureza da Despesa: 44.90.51. Fonte: 107. Nota de Empenho 2019NE01147, emitida em 27/12/2019 no valor de R$ 265.668,72 (duzentos e sessenta e cinco mil seiscentos e sessenta e oito reais e setenta e dois centavos). Valor do Contrato: R$ 265.668,72 (duzentos e sessenta e cinco mil seiscentos e sessenta e oito reais e setenta e dois centavos). Modalidade: global. Evento: 400091. Data da Assinatura: 30/12/2019. Vigencia: 06 (seis) meses. Signatarios: Pela CODHAB/DF: Wellington Luiz de Souza Silva, na qualidade de Diretor-Presidente; Pela Contratada: Alessandro Alves Beserra, na qualidade de Representante Legal.<EOB>"]

aux = crf.model_predict(txt)

print("==========================================================================")
for i in aux:
    print(i)
