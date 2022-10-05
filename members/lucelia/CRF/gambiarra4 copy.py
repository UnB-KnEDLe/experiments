
#processarLinha faz a troca dos labels e processarLinha2 insere uma linha em branco após cada setença
def processarLinha(sentences):
    #print(sentences)
    #print(sentences[1])
    #count = 0
    # Write the file out again
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'wt') as fileout:

        for linha in sentences:
            #count = count +1
            #print (linha)
            lista = linha.split("\t")
            
            #print (len(lista))
            #print (lista)     
            if len(lista)==2 and lista[1] == 'PERSON\n':
                lista[1]= 'B-PER\n'
                #print (lista)
                fileout.write(lista[0]+'\t'+lista[1])
            elif len(lista)==2 and lista[1] == 'GPE\n': 
                lista[1]= 'B-LOC\n'
                #print (lista)
                fileout.write(lista[0]+'\t'+lista[1])         
            elif len(lista)==2:
                fileout.write(lista[0]+'\t'+lista[1])
            #if count == 15:
              #  break


#atribui uma linha em branco entre as sentenças apos o ponto final
def processarLinha2(sentences1):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat_1.txt', 'wt') as fileout:
        texto = ''
        for linha in sentences1:
            p=linha.find('.')
            #print (p)
            if p==0:
                texto=texto+ linha +'\n'
            else:
                texto=texto+ linha
        fileout.write(texto)

#retira espaço entre palavras quebradas indevidamente
def retirarEspaco(sentences):
#Retira os espaços em branco e as words maiores que duas posições
        
        for i in range(len(sentences) - 1):
            atual = sentences[i].split()
            proximo = sentences[i+1].split()
            if len(atual) == 0:
                continue
            while len(proximo) > 2:
            #print(f'Convertendo ({atual}) e ({proximo}) para ', end = '')
                atual[0] += proximo[0]
                sentences[i] = ' '.join(atual)
                proximo = proximo[1:]
                sentences[i+1] = ' '.join(proximo)
            #print(f'({atual}) e ({proximo})')
        return sentences

#verifica as linhas com mais de duas words e concatena
def concatenarPalavra(sentences):
    for i in range(len(sentences) - 1):
        atual = sentences[i].split()
    #print('atual', atual[0])
    #proximo = test_sentences_label[i+1].split()
    #print('proximo', proximo[0])
    #print(i) 
        if ((len(atual)>2) and (len(atual)<=3)):
            #print('atual', atual)
            sentences[i]=(''.join(atual[0]+atual[1]))+' '+atual[2] 
            #print('sentences', sentences[i])
            #print(i)
        elif ((len(atual)>3) and (len(atual)<=4)):
            #print('atual', atual)
            sentences[i]=(''.join(atual[0]+atual[1]+atual[2]))+' '+atual[3] 
            #print('test_sentences_label', test_sentences_label[i])
            #print(i)
        elif (len(atual)>4):
            sentences[i]=(''.join(atual[0]+atual[1]+atual[2]+atual[3]))+' '+atual[4]
            print(i)
    return sentences

#retira as labels das words
def retirarLabel(sentences):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu.txt', 'wt') as fileout:  
    # for each line in the input file
        texto=''
        count=0

        for linha in sentences:
            if len(linha) != 1:
                x=linha.split()[0]              
                #print('x-->', x)
                texto=texto+x+'\n'
                continue
            
            #verifica se a linha tem mais de 2 palavras
            # if len(x) > 2:
            #     #print('split', x)
            #     print(f' {x} possui {len(x)} elementos.') 
            # if len(x) == 1:
            #     #print('split', x)
            #     print(f' {x} possui {len(x)} elementos.') 
            
        count+=1
        #print(count)
        #print(texto)
        fileout.write(texto)

   #print(sentenca)


#Abre o Ontonotes 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'r') as file:
    sentences1 = list(file.readlines())

#Abre o Ontonotes 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train.txt', 'r') as file:
    sentences = list(file.readlines())

retirarEspaco(sentences)
concatenarPalavra(sentences)
retirarLabel(sentences)
#processarLinha(sentences)
#processarLinha2(sentences1)




