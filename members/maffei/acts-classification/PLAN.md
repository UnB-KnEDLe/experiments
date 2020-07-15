prof, to quebrando a cabeça com esse negócio de classificar os atos,
Digo, o objetivo está claro mas estou com problemas quanto algumas coisas necessárias a saber:	
	+ separação de sentenças. Estou fazendo baseado em ponto final seguido por letra maiúscula, parece funcionar. Mas tipo, tem ato que é bem grande, ao passo q outros são bem pequenos. 

	+ rotulação: minha abordagem, ainda não implementada, deve ser assim:
		1. extraio todos os atos possíveis usando a bilioteca Regex que temos hoje, e faço a separação de senteças usando o procedimento especificado acima. Obtenho portanto dados rotulados 
		2. extraio os JSONs do PDFs usando o dodfminer. Para cada texto no json (excluindo título/subtítulo), faço uma busca por palavras chaves usadas  pelos regex. Se alguma estiver, rotulo de acordo. Senão, rotulo como "não ato". Ao final, haveria uma base de "atos"/"não atos".
		Ao final portanto haveria q se fazer 2 classificações: se é ou não ato e, se for, qual ato


--------------------------

Tenta fazer um classificar binário, por exemplo, pra saber se é aposentadoria ou não. Então pegue todos os atos, e monte a base com atos de aposentadoria e os restantes.