
#1 - processarLinha faz a troca dos labels 
def processarLinha(sentences):
    #print(sentences)
    #print(sentences[1])
    #count = 0
    # Write the file out again
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_label.txt', 'wt') as fileout:
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


#2 - retira espaço entre palavras quebradas indevidamente
def retirarEspaco(sentences):
#Retira os espaços em branco e as words maiores que duas posições
    #with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_quebra.txt', 'wt') as fileout:
      
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
        #fileout.write(str(sentences))

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
            #print(i)
    return sentences

def gerarSentenca(sentences):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'wt') as fileout:
        texto=''
        for linha in sentences:
            texto=texto+linha+'\n'
        fileout.write(texto)

#retira as labels das words
def retirarLabel(sentences):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu.txt', 'wt') as fileout:
        texto=''
        #count=0
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
        fileout.write(texto)        
        #count+=1
        #print(count)
        #print(texto)
        #return (texto)

   #print(sentenca)

#atribui uma linha em branco entre as sentenças apos o ponto final
def processarLinha2(sentences):
    #grava o dataset do Y_test
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'wt') as fileout:
    #grava o dataset do X_test
    #with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu.txt', 'wt') as fileout:
        arq = ''
        for linha in sentences:
            p=linha.find('.')
            #print (p)
            if p==0:
                arq=arq+ linha +'\n'
            else:
                arq=arq+ linha
        fileout.write(arq)

#Abre o Ontonotes para aplicar o tratamento 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train.txt', 'r') as file:
   sentences = list(file.readlines())

with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_label.txt', 'r') as file:
   sentences_label = list(file.readlines())

#Abre o Ontonotes do Y_test após ajuste do label
#with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'r') as file:
#    texto = list(file.readlines())


#Abre o Ontonotes do X_test para adicionar a linha em branco
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu_1.txt', 'r') as file:
    texto = list(file.readlines())

# 1 - faz a troca dos label para ficar no mesmo formato do CONLL
#processarLinha(sentences)

# 2 - retira espaço entre palavras quebradas indevidamente 
retirarEspaco(sentences_label)

# 3 - verifica as linhas com mais de duas words e concatena
concatenarPalavra(sentences)

#gerarSentenca(sentences)

# 4 - retira as labels das words
#retirarLabel(sentences)

# 5 - adiciona linha em branco entre as sentenças
processarLinha2(sentences)



