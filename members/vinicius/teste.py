
string = "especiﬁcações"

nova = ""
f = open("/home/viniciusrpb/Projects/biocxml/dodf_xmls/DODF 005 07-01-2022.xml", "r")
print()

lines = f.readlines()

for string in lines:
    for i in string:
        if ord(i) > 300:
            print(f'char: {i} ascii: {ord(i)}')

for i in range(0,len(string)):
    if ord(string[i]) == 64257:
        nova=nova+"fi"
    else:
        nova=nova+string[i]

print(nova)
    
