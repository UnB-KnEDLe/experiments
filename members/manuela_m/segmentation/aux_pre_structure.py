from email.mime import text
import glob
import re
from Bio.Align import PairwiseAligner
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
import re

from aux_functions import *


class LabelStructure:
    """Class with static methods to create a labeled DODF structure by sentences"""

    @staticmethod
    def find_alignment(data, segment: str):
        """Find the alignment between text from path_dodf and the text segment.
        Args:
            path_dodf ([type]): path to text file extracted from DODF
            segment (str): text segmento to match in path_dodf document
        Returns:
            alignment
        """
        aligner = PairwiseAligner()
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = -0.1
        aligner.target_end_gap_score = 0.0
        aligner.query_end_gap_score = 0.0

        alignment = None
        alignment = next(aligner.align(data, segment))
        return alignment

    @staticmethod
    def load_txt_by_numdodf(path_txt) -> dict:
        """Load documents from a directory path and create a dictionary
        Args:
            path_txt ([type]): path to directory with *.txt
        Returns:
            dict: Dictionary with key value the number of DODF e values a list of text file path.
        """
        dic_dodfs = {}
        for p in glob.glob(path_txt):
            num_dodf = int(p.split()[1])
            if num_dodf not in dic_dodfs:
                dic_dodfs[num_dodf] = []
            dic_dodfs[num_dodf].append(p)
        return dic_dodfs

    @staticmethod
    def positions(
        df, path_txt, column_name_num_dodf="NUM_DODF", columns_name_text="text"
    ):
        """[summary]
        Args:
            df ([type]): [description]
            path_txt ([type]): [description]
            column_name_num_dodf (str, optional): [description]. Defaults to 'NUM_DODF'.
            columns_name_text (str, optional): [description]. Defaults to 'text'.
        """
        dic_dodfs = LabelStructure.load_txt_by_numdodf(path_txt)
        act_pos = {}
        for i in range(len(df)):
            num_dodf = int(df.loc[i, column_name_num_dodf])
            act = df.loc[i, columns_name_text]
            paths_dodf = dic_dodfs[num_dodf]
            for path in paths_dodf:
                alig = LabelStructure.find_alignment(path, act)
                if alig.score > len(act) - 10:
                    init = alig.aligned[0][0][0]
                    end = alig.aligned[0][0][1]
                    act_pos[path] = act_pos.get(path, [])
                    act_pos[path].append((i, act, init, end))
        return act_pos

    # segmenta o texto selecionando os atos e tudo entre os atos.
    @staticmethod
    def segmentor(text, pos_list):
        pos_list.sort(key=lambda x: x[2])

        init = 0
        end = len(text)
        segs = []
        for pos in pos_list:
            act = pos[1]
            a_init = pos[2]
            a_end = pos[3]
            segs.append(text[init:a_init])
            segs.append(act)
            init = a_end + 1
        segs.append(text[init:end])
        return segs

    @staticmethod
    def sentence_labeling(segs, labels=["O", "B", "I"]):
        iob = []
        for i, txt in enumerate(segs):
            lines = []
            txt = re.sub(r"(\s+)", " ", txt)
            sents = sent_tokenize(txt)
            if i % 2 == 0:
                lines = [labels[0] + " " + sent for sent in sents]
            else:
                lines += [labels[1] + " " + sents[0]]
                lines += [labels[2] + " " + sent for sent in sents[1:]]

            lines = "\n".join(lines)
            iob.append(lines)

        return "\n\n".join(iob)

    @staticmethod
    def word_labeling(segs, labels=["O", "B", "I"]):
        label_sents = []
        for i, txt in enumerate(segs):
            txt = txt.replace("\n", " ")
            sents = wordpunct_tokenize(txt)
            if i % 2 == 0:
                label_sents += [labels[0] + " " + sent for sent in sents]
            else:
                label_sents.append(labels[1] + " " + sents[0])
                label_sents += [labels[2] + " " + sent for sent in sents[1:]]
        return "\n\n".join(label_sents)
