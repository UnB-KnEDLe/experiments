from os import listdir
from os.path import abspath
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from math import floor, ceil
from gensim.models import Word2Vec
import krippendorff

def get_csv(path_xmls, name_csv="agreement.csv"):
    """Gets a list of xmls and returns a csv file

    Parameters
    ----------
    path_xmls : str
        Path to folder containing xml files in BioC-XML format
    
    name_csv : str
        Desired name for the output CSV file

    Returns
    -------
    path : path to CSV file
        The path for the created csv file with the annotations of the xmls
    """
    
    xmls = listdir(path_xmls)

    # lista de roots para iteração posterior
    roots = []

    # populando lista de roots e iterando na lista xmls
    for xml in xmls:
        xml = path_xmls + '/' + xml
        tree = ET.parse(xml)
        root = tree.getroot()
        roots.append(root)
    
    # abre csv para escrita de headers
    with open(name_csv, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id_dodf', 'tipo_rel', 'id_rel',
                        'anotador_rel', 'tipo_ent', 'id_ent',
                        'anotador_ent', 'offset', 'length', 'texto'])
        
    # abre csv para escrita em modo append
    with open(name_csv, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        # itera na lista de roots
        for root in roots:
            # coleta id do dodf
            id_dodf = root.find("./document/id")
            id_dodf_text = id_dodf.text
            # cria lista de ids de relações
            ids_rel = []
            for rel in root.findall("./document/passage/relation"):
                id_rel = rel.get('id')
                ids_rel.append(id_rel)
                # coleta tipo e anotador da relação
                for info in rel.findall('infon'):
                    if info.get('key') == 'type':
                        tipoRel = info.text
                        #print(tipoRel)
                    elif info.get('key') == 'annotator':
                        annotatorRel = info.text
                        #print(annotatorRel)
                # cria lista de ids de anotações
                ids_anno = []
                for info in rel.findall('node'):
                    id_anno = info.get('refid')
                    ids_anno.append(id_anno)
                #print(ids_anno)
                # loop na lista de ids
                for id_anno in ids_anno:
                    # encontra e itera sobre todos os elementos annotation do xml
                    for anno in root.findall("./document/passage/annotation"):
                        # para cada anotação definida por um id, coleta o tipo,
                        # anotador, offset, length e texto
                        if anno.get('id') == id_anno:
                            # encontra tipo e anotador
                            for info in anno.findall('infon'):
                                if info.get('key') == 'type':
                                    tipoAnno = info.text
                                    #print(tipoAnno)
                                elif info.get('key') == 'annotator':
                                    annotatorAnno = info.text
                                    #print(annotatorAnno)
                            # encontra offset e length
                            for info in anno.findall('location'):
                                offset = info.get('offset')
                                #print(offset)
                                length = info.get('length')
                                #print(length)
                            # encontra texto
                            for info in anno.findall('text'):
                                texto = info.text
                                #print(texto)
                            # escreve linha no csv
                            writer.writerow([id_dodf_text, tipoRel, id_rel,
                                             annotatorRel, tipoAnno, id_anno,
                                             annotatorAnno, offset, length, texto])

    path = abspath(name_csv)
    return path


def aux_structures(path_csv):
    """Gets a csv file and returns a duplicate mapping and 
    a in-memory dict of entities

    Parameters
    ----------
    path_csv : str
        Path to CSV file

    Returns
    -------
    map_duplicate : dict
        A dict containing all duplicates offsets through the CSV file

    data_dict : dict
        A dict containing tuples for each annotation,
        consisting of the following entities:
        pointer to first char, length, anotator, type of entity, text,
        integer encoding of the type of entity, id of the dodf containing the annotation
        (offset, length, anotator, tipo_ent, text, label_int, id_dodf)
    """

    df = pd.read_csv(path_csv)

    offsets = df.offset.to_list()
    lengths = df.length.to_list()
    texts = df.texto.to_list()
    anotators = df.anotador_ent.to_list()
    tipo_ent = df.tipo_ent.to_list()
    ids_ent = df.id_ent.to_list()

    labels_unique = df.tipo_ent.unique().tolist()
    labels_unique_dict = {}
    for index, possible_label in enumerate(labels_unique):
        labels_unique_dict[possible_label] = index
    
    # passar para função get_csv
    df['tipo_ent_num'] = df.tipo_ent.replace(labels_unique_dict)
    labels_int = df.tipo_ent_num.to_list()
    ids_dodfs = df.id_dodf.to_list()

    off_sem_duplicata = np.unique(df.offset.to_numpy())

    map_duplicata = {}
    for i in off_sem_duplicata:
        map_duplicata[i] = []
    for idx, offset in enumerate(offsets):
        map_duplicata[offset].append(idx)
    
    data_dict = {}
    for idx, _ in enumerate(offsets):
        data_dict[idx] = (offsets[idx], lengths[idx],
                          anotators[idx], tipo_ent[idx],
                          texts[idx], labels_int[idx], ids_dodfs[idx])
    
    map_duplicata_final = map_duplicata
    for off_ref in map_duplicata:
        lista_linha_dataframe = map_duplicata[off_ref]
        # vamos andar pra trás threshold caracteres e depois pra frente threshold caracteres
        for linha_dataframe in lista_linha_dataframe:
            threshold_init = ceil(len(data_dict[linha_dataframe][4])/10)
            threshold = threshold_init if threshold_init > 1 else 1
            for i in range(threshold):
                # andando pra trás
                try:
                    # linha mapeada por offset_referencia - threshold da vez
                    linha_threshold = data_dict[linha_dataframe-(threshold-i)][0]
                    
                    # label da anotação referência
                    label_ref = data_dict[linha_dataframe][3]
                    
                    # label da anotação referente à linha mapeada por offset_referencia - threshold da vez
                    label_threshold = data_dict[linha_dataframe-(threshold-i)][3]
                    
                    # texto da anotação mapeada por linha dataframe
                    anotacao_referencia = data_dict[linha_dataframe][4]
                    
                    # texto da anotação mapeada por linha threshold
                    anotacao_threshold = data_dict[linha_threshold][4]

                    # verifica tamanho das duas anotações para testar se é substring
                    if(len(anotacao_threshold) > len(anotacao_referencia)):
                        # se existe anotação na linha_threshold
                        if(map_duplicata[linha_threshold]):
                            # se o label dela for igual ao label da anotação referência
                            if(label_ref == label_threshold):
                                # se o tamanho dela for no máximo tamanho 
                                # anotação referência + 10% tamanho anotação referência
                                if(len(anotacao_threshold) < floor(len(anotacao_referencia)*(11/10))):
                                    if anotacao_referencia in anotacao_threshold:
                                        map_duplicata_final[linha_dataframe].append(linha_threshold)
                    else:
                        # se existe anotação na linha_threshold
                        if(map_duplicata[linha_threshold]):
                            # se o label dela for igual ao label da anotação referência
                            if(label_ref == label_threshold):
                                # se o tamanho dela for no mínimo 90% do tamanho da anotação referência
                                if(len(anotacao_threshold) > floor(len(anotacao_referencia)*(9/10))):
                                    if anotacao_threshold in anotacao_referencia:
                                        map_duplicata_final[linha_dataframe].append(linha_threshold)
                except:
                    pass
                # andando pra frente
                try:
                    # linha mapeada por offset_referencia - threshold da vez
                    linha_threshold = data_dict[linha_dataframe+(i+1)][0]
                    
                    # label da anotação referência
                    label_ref = data_dict[linha_dataframe][3]
                    
                    # label da anotação referente à linha mapeada por offset_referencia - threshold da vez
                    label_threshold = data_dict[linha_dataframe+(i+1)][3]
                    
                    # texto da anotação mapeada por linha dataframe
                    anotacao_referencia = data_dict[linha_dataframe][4]
                    
                    # texto da anotação mapeada por linha threshold
                    anotacao_threshold = data_dict[linha_threshold][4]
                    
                    # verifica tamanho das duas anotações para testar se é substring
                    if(len(anotacao_threshold) > len(anotacao_referencia)):
                        # se existe anotação na linha_threshold
                        if(map_duplicata[linha_threshold]):
                            # se o label dela for igual ao label da anotação referência
                            if(label_ref == label_threshold):
                                # se o tamanho dela for no máximo tamanho 
                                # anotação referência + 10% tamanho anotação referência
                                if(len(anotacao_threshold) < floor(len(anotacao_referencia)*(11/10))):
                                    if anotacao_referencia in anotacao_threshold:
                                        map_duplicata_final[linha_dataframe].append(linha_threshold)
                    else:
                        # se existe anotação na linha_threshold
                        if(map_duplicata[linha_threshold]):
                            # se o label dela for igual ao label da anotação referência
                            if(label_ref == label_threshold):
                                # se o tamanho dela for no mínimo 90% do tamanho da anotação referência
                                if(len(anotacao_threshold) > floor(len(anotacao_referencia)*(9/10))):
                                    if anotacao_threshold in anotacao_referencia:
                                        map_duplicata_final[linha_dataframe].append(linha_threshold)
                except:
                    pass

    return map_duplicata_final, data_dict

def jaccard_char(str1, str2):
    """Gets two strings and returns the jaccard index of
    similarity between them.

    Parameters
    ----------
    str1 : str
        First string to compare
    
    str2 : str
        Second string to compare

    Returns
    -------
    agreement : float
        The value corresponding to the Jaccard agreement
        index between str1 and str2
    """

    diferenca_char = 0
    # string 2 maior que string 1
    if len(str2) > len(str1):
        inicio = str2.find(str1)
        # teste substring
        if inicio == -1:
            agreement = 0
            return agreement
        else:
            idx_menor = 0
            for idx, _ in enumerate(str2):
                try:
                    # shift na string até encontrar o caractere de inicio
                    if idx < inicio:
                        diferenca_char += 1
                        # contador para shift-reverso
                        idx_menor +=1
                    else:
                        # strings 'alinhadas', diferença char a char
                        if str2[idx] != str1[idx-idx_menor]:
                            diferenca_char += 1
                # acabamos de percorrer str1; computamos a diferença remanescente em relação à str2
                except:
                    dif_reman = len(str2) - idx
                    diferenca_char += dif_reman
                    break
            # computamos o agreement usando jaccard
            agreement = abs(len(str2) - diferenca_char)/len(str2)
            
    # string 1 maior que string 2
    elif len(str1) > len(str2):
        # teste substring
        inicio = str1.find(str2)
        if inicio == -1:
            agreement = 0
            return agreement
        else:
            idx_menor = 0
            for idx, _ in enumerate(str1):
                try:
                    # shift na string até encontrar o caractere de inicio
                    if idx < inicio:
                        diferenca_char += 1
                        # contador para shift-reverso
                        idx_menor +=1
                    else:
                        # strings 'alinhadas', diferença char a char
                        if str1[idx] != str2[idx-idx_menor]:
                            diferenca_char += 1
                # acabamos de percorrer str2; computamos a diferença remanescente em relação à str1
                except:
                    dif_reman = len(str1) - idx
                    diferenca_char += dif_reman
                    break
            # computamos o agreement usando jaccard
            agreement = abs(len(str1) - diferenca_char)/len(str1)
            
    # strings de mesmo tamanho
    else:
        # teste subtring
        inicio = str1.find(str2)
        if inicio == -1:
            agreement = 0
            return agreement
        # str1 e str2 são a mesma string
        elif inicio == 0:
            agreement = 1
        else:
            # strings alinhadas, diferença char a char
            for idx, _ in enumerate(str1):
                if str1[idx] != str2[idx]:
                    diferenca_char += 1
            total_dif = abs(len(str1) - diferenca_char)/len(str1)
            agreement = 1 - total_dif
    
    return agreement

def get_word2vec(path_csv):
    """Gets a csv file and return the Word2Vec
    50 dimensional embedding vectors of it.

    Parameters
    ----------
    path_csv : str
        Path to CSV file
    
    Returns
    -------
    vectors : gensim.models.keyedvectors.Word2VecKeyedVectors
        The value corresponding to the Jaccard agreement
        index between str1 and str2
    """
    
    df = pd.read_csv(path_csv)
    anotacoes = df['texto'].to_list()
    tokens = []

    for texto in anotacoes:
        token_list = texto.split()
        tokens.append(token_list)
    
    def hash_func(astring):
        return hash(astring)
    
    model = Word2Vec(sentences=tokens, size=50, window=3, min_count=1,
                     workers=1, negative=5, seed=14, hashfxn=hash_func)

    vectors = model.wv

    return vectors

def get_vector(text, vectors):
    """Gets a string and return the Word2Vec
    vector representation of it.

    Parameters
    ----------
    text : str
        Texto to be converted to vector representation
    
    vectors : gensim.models.keyedvectors.Word2VecKeyedVectors
        An embedding vector corpus representation
    
    Returns
    -------
    vector : numpy.ndarray
        The vector representation of the input string
    """

    token_list = text.split()
    vector = np.zeros(50)
    for token in token_list:
        vector += vectors[token]
    return vector

def cos_sim(str1, str2):
    """Gets two strings and returns the cosine
    similarity between them.

    Parameters
    ----------
    str1 : str
        First string to compare
    
    str2 : str
        Second string to compare

    Returns
    -------
    cos : numpy.float64
        The value corresponding to the cosine
        similarity between str1 and str2
    """

    dot = np.dot(str1, str2)
    norma = np.linalg.norm(str1)
    normb = np.linalg.norm(str2)
    cos = dot/(norma * normb)
    
    return cos

def euclidean(str1, str2):
    """Gets two strings and returns the euclidean
    distance between them.

    Parameters
    ----------
    str1 : str
        First string to compare
    
    str2 : str
        Second string to compare

    Returns
    -------
    dist : numpy.float64
        The value corresponding to the euclidean
        distance between str1 and str2
    """

    dist = np.linalg.norm(str1-str2)
    return dist

def krippendorff_score(path, map_duplicata, data_dict):
    """Gets the duplicate mapping and returns the
    krippendorf alpha score.

    Parameters
    ----------
    path : str
        Path to CSV file of annotations

    map_duplicata : dict
        Mapping of duplicate annotations
    
    data_dict : dict
        A dict containing the entities in memory

    Returns
    -------
    kripp : numpy.float64
        The value of krippendorf alpha between
        the annotations given
    """

    df = pd.read_csv(path)
    unique_anotators = np.unique(df.anotador_ent.to_numpy())

    map_dupli = {}
    # iteramos em cada entrada do dict
    for key_offset, lista_linhas in map_duplicata.items():
        # percorremos cada linha de cada chave
        for linha in lista_linhas:
            # pegamos o id do dodf da anotação
            id_dodf = data_dict[linha][6]
            # o anotador
            anotator = data_dict[linha][2]
            # o label inteiro dado
            label = data_dict[linha][5]
            # criamos uma tupla com a anotação
            tupla_anotacao = (anotator, label, id_dodf)
            # e uma nova chave, contendo o id do dodf + o offser da anotação
            nova_chave = id_dodf + '__' + str(key_offset)
            # se estiver no mesmo dodf, fazer append na mesma chave
            # caso contrario, criar uma chave nova no dict
            if nova_chave in map_dupli.keys():
                if tupla_anotacao not in map_dupli[nova_chave]:
                    map_dupli[nova_chave].append((anotator, label, id_dodf))
            else:
                map_dupli[nova_chave] = [(anotator, label, id_dodf)]

    map_dupli_usavel = {}
    # iteramos no dict
    for key, item in map_dupli.items():
        # para cada lista de anotações, filtramos para > 1
        if len(item) > 1:
            map_dupli_usavel[key] = item
    
    final_data = {}
    # iteramos em map_dupli_usavel
    for key, item in map_dupli_usavel.items():
        # cada lista deve ter a dimensionalidade igual ao nº de anotadores únicos
        final_row = [np.nan] * len(unique_anotators)
        # como filtramos acima, todos os casos desse dicionário possuem ao menos 2 anotações
        for tupla in item:
            anotator = tupla[0]
            label = tupla[1]
            # comparamos o anotador com a lista de anotadores
            for idx, unq_anotator in enumerate(unique_anotators):
                # se coincidir, damos o label dado. caso contrário, permanace o np.nan
                if anotator == unq_anotator:
                    final_row[idx] = label
            final_data[key] = final_row
    
    agreement_list = list(final_data.values())

    kripp = krippendorff.alpha(reliability_data=agreement_list, level_of_measurement='nominal')

    return kripp

