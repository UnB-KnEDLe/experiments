#encoding: utf-8
txt = ['A\tO\n', 'House\tO\n', '.\tO\n', '\n', 'Pedro\tPER\n', '.\tO\n', '\n', 'Doing\tVERB\n', 'something\tO\n', '\n']

def list_of_lists_to_list_of_strings(ls):
    return [' '.join(l) for l in ls]

def lines_to_sentences(lines):
    """
    Só funciona se último elemento de lines for '\n',
    se não for assim no seu caso adicione um no final
    caso contrário vai pular a última sentença
    """
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in lines:
        if line == '\n':
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            word, word_label = line.strip('\n').split('\t')
            sentence.append(word)
            label.append(word_label)
    
    sentences = list_of_lists_to_list_of_strings(sentences)
    labels = list_of_lists_to_list_of_strings(labels)
    return sentences, labels

print('Sentenças e labels separados')
sentences, labels = lines_to_sentences(txt)
print(sentences)
print(labels)

class LanguageDetector:
    """Alterna a linguagem detectada só pra teste"""
    def __init__(self):
        self.l = True

    def lang(self, sentence):
        if self.l:
            language = 'en'
        else:
            language = 'ko'
        self.l = not self.l
        return language

def filter_sentences(sentences, labels):
    """
    filtra pela linguagem
    """
    filtered_sentences = []
    filtered_labels = []
    ld = LanguageDetector()
    for sentence, label in zip(sentences, labels):
        if ld.lang(sentence) == 'en':
            filtered_sentences.append(sentence)
            filtered_labels.append(label)
    return filtered_sentences, filtered_labels


def sentences_to_lines(sentences, sentence_labels):
    """
    Contrário de lines_to_sentences
    """
    lines = []
    for sentence, labels in zip(sentences, sentence_labels):
        words = sentence.split(' ')
        word_labels = labels.split(' ')
        for word, label in zip(words, word_labels):
            # line = f'{word}\t{label}\n'
            line = word + '\t' + label + '\n'
            lines.append(line)
        lines.append('\n')
    return lines

print('Sentenças e labels filtrados pela linguagem')
filtered_sentences, filtered_labels = filter_sentences(sentences, labels)
print(filtered_sentences)
print(filtered_labels)

print('Linhas após filtragem')
filtered_lines = sentences_to_lines(filtered_sentences, filtered_labels)
print(filtered_lines)
