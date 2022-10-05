# Read in the file
with open('/home/82068895153/POS/skweak/data/conll2003_dataset/train_out.txt', 'r') as file: 
  filedata = file.readlines()


texto_sentenca = ''
sentenca = ''
lista_token = ['"',',','$','(',')']

for linha in filedata:
  
  """  if linha.strip() == '.':
       #print (linha)
       texto_sentenca = texto_sentenca + sentenca + '\n' 
       sentenca = ''
   else:
       sentenca = sentenca + linha + '\n'
       lista_sentenca = sentenca.split('\n')
       lista_sentenca.append(linha)
       if linha in lista_token:
          sentenca = ''.join(lista_sentenca)
       else:
          sentenca = ' '.join(lista_sentenca)
 """
  if linha.strip() == '.':
      lista_sentenca = sentenca.split('\n')
      sentenca = '' 
      for s in lista_sentenca:
         if s.strip() in lista_token:
            sentenca = sentenca + s
            print ('entrou no if'+ s) 
         else:
            sentenca = sentenca + ' ' + s 
    
      texto_sentenca = texto_sentenca + sentenca + '\n' 
      sentenca = ''
  else:
      sentenca = sentenca + linha + '\n'
       
 
# Write the file out again
with open('/home/82068895153/POS/skweak/data/conll2003_dataset/train_out_1.txt', 'wt') as fileout:
  fileout.write(texto_sentenca)