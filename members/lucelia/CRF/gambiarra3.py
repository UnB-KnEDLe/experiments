
def processarLinha(sentences):
    #print(sentences)
    #print(sentences[1])
    count = 0
    # Write the file out again
    with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_trat.txt', 'wt') as fileout:
        texto=''
        sentenca=''
        for linha in sentences:
            count = count +1
            #print (linha)
            lista = linha.split("\t")
            pos = linha.find('.')
            print ('pos ==',pos)
            if pos > 0:
                p=''.join(linha[:pos])
                print ('p ==',p)
                if len(lista)==2 and lista[1] == 'PERSON\n':
                    lista[1]= 'PER\n'
                    texto=texto+lista[0]+'\t'+lista[1]
                    print ('texto person ==',texto)
                elif len(lista)==2:
                    texto=texto+lista[0]+'\t'+lista[1]
                    print ('texto geral ==',texto)
                texto=texto + p
                print ('texto final ==',texto)
            sentenca = sentenca + texto + '\n'
            print ('sentenca ==',sentenca)
            if count == 30:
                break  

        fileout.write(sentenca)
        
            #print (len(lista))
            #print (lista)     
            #if len(lista)==2 and lista[1] == 'PERSON\n':
            #    lista[1]= 'PER\n'
                #print (lista)
            #    fileout.write(lista[0]+'\t'+lista[1])
            #elif len(lista)==2 and lista[1] == 'GPE\n': 
             #   lista[1]= 'LOC\n'
                #print (lista)
              #  fileout.write(lista[0]+'\t'+lista[1]) 

            #elif len(lista)==2:
             #   fileout.write(lista[0]+'\t'+lista[1])
            #if count == 15:
              #  break
        
#Abre o Ontonotes 
with open('/home/82068895153/POS/skweak/data/Ontonotes/ner_train_1.txt', 'r') as file:
    sentences = list(file.readlines())

processarLinha(sentences)




