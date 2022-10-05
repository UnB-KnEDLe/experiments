
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
                lista[1]= 'PER\n'
                #print (lista)
                fileout.write(lista[0]+'\t'+lista[1])
            elif len(lista)==2 and lista[1] == 'GPE\n': 
                lista[1]= 'LOC\n'
                #print (lista)
                fileout.write(lista[0]+'\t'+lista[1])         
            elif len(lista)==2:
                fileout.write(lista[0]+'\t'+lista[1])
            #if count == 15:
              #  break


def processarLinha2(sentences1):
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat_1.txt', 'wt') as fileout:


#Abre o Ontonotes 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'r') as file:
    sentences1 = list(file.readlines())

#Abre o Ontonotes 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_1.txt', 'r') as file:
    sentences = list(file.readlines())

processarLinha(sentences)

processarLinha2(sentences1)




