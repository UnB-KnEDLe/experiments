
#1 - processarLinha faz a troca dos labels 

# 1 - faz a troca dos label para ficar no mesmo formato do CONLL
#Abre o Ontonotes train para aplicar o tratamento 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train.txt', 'r') as file:
   sentences = list(file.readlines())
 
#Abre o Ontonotes do X_test para adicionar a linha em branco
#with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu_1.txt', 'r') as file:
#    texto = list(file.readlines())

def processarLinha(sentences):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_label.txt', 'wt') as fileout:

        #print(sentences)
        #print(sentences[1])
        #count = 0
        # Write the file out again
        label1=''
        for linha in sentences:
            label=''
            
            #count = count +1
            #print (linha)
            lista = linha.split("\t")    
            #print (len(lista))
            #print (lista)     
            if len(lista)==2 and lista[1] == 'PERSON\n':
                lista[1]= 'B-PER\n'
                #print (lista)
                label=label+(lista[0]+'\t'+lista[1])
            elif len(lista)==2 and lista[1] == 'GPE\n': 
                lista[1]= 'B-LOC\n'
                #print (lista)
                label=label+(lista[0]+'\t'+lista[1])         
            elif len(lista)==2:
                    label=label+(lista[0]+'\t'+lista[1])
                #if count == 15:
                #  break 
        fileout.write(label)
        #label1=label1+label
        #print(label1)
        #return label1

processarLinha(sentences)
#print(label[0])

# 2 - retira espaço entre palavras quebradas indevidamente 
#with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_label.txt', 'wt') as file:
#    label = list(file.readlines())

#2 - retira espaço entre palavras quebradas indevidamente
def retirarEspaco(label):
#Retira os espaços em branco e as words maiores que duas posições
    #with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_quebra.txt', 'wt') as fileout:    
        quebra=''
        print('label===>', label[5])
        for i in range(len(label) - 1):
            atual = label[i].split()
            proximo = label[i+1].split()
            if len(atual) == 0:
                continue
            while len(proximo) > 2:
                #print(f'Convertendo ({atual}) e ({proximo}) para ', end = '')
                atual[0] += proximo[0]
                label[i] = ' '.join(atual)
                proximo = proximo[1:]
                label[i+1] = ' '.join(proximo)
                #print(f'({atual}) e ({proximo})')
            #print('label sem a quebra',label)
            quebra=quebra+label
            #print('saida para o arquivo',quebra)
        #fileout.write(quebra)
        return quebra

#word = retirarEspaco(label)


#3 - verifica as linhas com mais de duas words e concatena
def concatenarPalavra(word):
    #with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_espaco.txt', 'wt') as fileout:    
        espaco = ''
        for i in range(len(word) - 1):
            atual = word[i].split()
        #print('atual', atual[0])
        #proximo = test_sentences_label[i+1].split()
        #print('proximo', proximo[0])
        #print(i) 
            if ((len(atual)>2) and (len(atual)<=3)):
                #print('atual', atual)
                word[i]=(''.join(atual[0]+atual[1]))+' '+atual[2] 
                #print('sentences', sentences[i])
                #print(i)
            elif ((len(atual)>3) and (len(atual)<=4)):
                #print('atual', atual)
                word[i]=(''.join(atual[0]+atual[1]+atual[2]))+' '+atual[3] 
                #print('test_sentences_label', test_sentences_label[i])
                #print(i)
            elif (len(atual)>4):
                word[i]=(''.join(atual[0]+atual[1]+atual[2]+atual[3]))+' '+atual[4]
                print(i)
            espaco=espaco+word
        #fileout.write(espaco)
        return espaco
    
"""
#4 - retira as labels das words
def retirarLabel(sentences):
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
            
    #count+=1
    #print(count)
    #print(texto)
    return (texto)

   #print(sentenca)
 """
#5 - atribui uma linha em branco entre as sentenças apos o ponto final
def processarLinha2(texto):
    #grava o dataset do Y_test
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'wt') as fileout:
    #grava o dataset do X_test
    #with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_lu.txt', 'wt') as fileout:
        arq = ''
        for linha in texto:
            p=linha.find(' . ')
            #print (p)
            if p==0:
                arq=arq+ linha +'\n'
            else:
                arq=arq+ linha
        fileout.write(arq)

# 3 - verifica as linhas com mais de duas words e concatena
#linhas = concatenarPalavra(word)

# 4 - retira as labels das words
#w_label= retirarLabel(linhas)

#Se for tratar o Y_test precisa executar apenas os passos 1,2,3, gravar a saĩda do 3 e depois chamar a saĩda do 3 e rodar o passo 5

#grava a saída do passo 3
#with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_espaco.txt', 'wt') as fileout:
#    fileout.write(linhas)   

#Abre o arquivo do passo 3
#with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_espaco.txt', 'r') as file:
#    texto = list(file.readlines())

# 5 - adiciona linha em branco entre as sentenças
#processarLinha2(texto)

##==== Arquivos
#.../Ontonotes/ner_train_lu_1.txt -- X_test -- train sem label e sem \n entre as sentencas
#.../Ontonotes/ner_train_lu.txt -- X_test -- train sem label e com \n entre as sentencas
#../Ontonotes/ner_train_trat.txt -- Y_test -- train com label e sem \n entre as sentencas
#../Ontonotes/ner_train_trat_1.txt -- Y_test -- train com label e com \n entre as sentencas