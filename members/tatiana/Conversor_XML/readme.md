# Conversor de Arquivos XML para CSV

O código "[Conversor_XML](https://github.com/UnB-KnEDLe/experiments/blob/master/members/tatiana/Conversor_XML/Conversor_XML.ipynb)" passa as informações contidas em arquivos XML gerados pelo TeamTat e as organiza em arquivos CSV de acordo com os atos que foram anotados. 

* Link de acesso ao projeto de anotação no TeamTat: https://www.teamtat.org/sessions/a2984585-2b00-4a2c-a6f0-7afc5543d74c

No projeto presente no link acima estão 3 documentos nos quais foram anotados alguns exemplos de 13 tipos de atos do DODF. Através desse link, você tem acesso aos documentos anotados, à lista dos tipos de entidades e à lista dos tipos de relações que foram usados no processo de anotação. 

Caso queira criar um novo projeto, testar funções de upload de documentos e coisas do tipo, talvez o [tutorial](https://github.com/UnB-KnEDLe/tutorial_annotation_teamtat) feito pela equipe de anotação possa ajudar, mas qualquer dúvida é só perguntar. 
  
Na pasta '[xml](https://github.com/UnB-KnEDLe/experiments/tree/master/members/tatiana/Conversor_XML/xml)' estão os arquivos xml que foram gerados pelo TeamTat e que contém todas as entidades e relações anotadas nos 3 documentos do DODF. 

Na pasta '[csv](https://github.com/UnB-KnEDLe/experiments/tree/master/members/tatiana/Conversor_XML/csv)' estão os arquivos csv que foram gerados pelo código, contendo as informações que foram anotadas utilizando o TeamTat. Esses arquivos também podem ser visualizados no jupyter notebook '[Conversor_XML.ipynb](https://github.com/UnB-KnEDLe/experiments/blob/master/members/tatiana/Conversor_XML/Conversor_XML.ipynb)'. 

Os arquivos xml gerados pelo TeamTat seguem o padrão BioC, aqui estão alguns links sobre esse padrão/formato que podem ser úteis:
* http://bioc.sourceforge.net/
* https://biocreative.bioinformatics.udel.edu/media/store/files/2014/2_BioC_bc2014_final.pdf
* https://pypi.org/project/bioc/
* https://github.com/yfpeng/bioc


E aqui tem a documentação da biblioteca 'xml.etree.ElementTree', utilizada no código "[Conversor_XML](https://github.com/UnB-KnEDLe/experiments/blob/master/members/tatiana/Conversor_XML/Conversor_XML.ipynb)": https://docs.python.org/2/library/xml.etree.elementtree.html
