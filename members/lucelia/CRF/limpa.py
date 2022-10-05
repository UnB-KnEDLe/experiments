simbolos_pontuacao = ["'", '"', ",", ".", "!", ":", ";", '#', '@', '-']
texto = ['Lorem : "ipsum" \'dolor\', sit ! Amet ; bla ... bl @ #etc #, whiskas sache !!']

# incluí um elemento com mais de um caractere
simbolos_pontuacao = ['...', "'", '"', ",", "!", ":", ";", '#', '@']


import re

for c in simbolos_pontuacao:
    r = re.compile('({"|".join(map("\s+$", ""))})',texto)
#texto = r.sub('', texto)

print(r) # Lorem ipsum dolor sit Amet blabl etc whiskas sache
