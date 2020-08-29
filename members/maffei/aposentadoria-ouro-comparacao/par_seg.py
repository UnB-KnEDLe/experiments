from collections import Counter
import gc
import numpy as np
from typing import List, Dict, Iterable
from itertools import chain

import fitz
from fitz.utils import getColor

COLORS = fitz.utils.getColorList()
NOTWHITE = [i for i in COLORS if 'WHITE' not in i]
np.random.seed(seed=42)
np.random.shuffle(NOTWHITE)

FORBIDDEN = [
    'Este documento pode ser',
    'Documento assinado digita',
    'Infraestrutura de Chaves',
    'PÁG.',
    'SEÇÃO',
    'pelo código',
]

def gpk(lis_dic, key):
    """Group iterable of dict by specified `key`.

    Args:
        lis_dic: Iterable[Dict]
        key: whatever hashble value to be used as the key
    Returns:

    """
    gp = {}
    lis_key = []
    for d in lis_dic:
        gp[d[key]] = []
    for d in lis_dic:
        gp[d[key]].append(d)
    return gp


def get_spans_lines(doc, glue_horizon = False):
    _, _, wid, hei = doc[0].MediaBox
    spans = []
    lines = []
    for pnum in range(doc.pageCount):
        d = doc[pnum].getTextPage().extractDICT()
        for block in d['blocks']:
            for line in block['lines']:
                d = {}
                d['bbox'] = line['bbox']
                d['text'] = '\n'.join(
                         [sp['text'] for sp in line['spans']
                         if all([not sp['text'].startswith(i) for i in FORBIDDEN])])
                d['page'] = 2*pnum + (1 if line['bbox'][0] > (wid/2) else 0)
                lines.append(d)
                for span in line['spans']:
                    t = span['text']
                    if len(t) > 2 and '....' not in t\
                        and all([not t.startswith(i) for i in FORBIDDEN]):
                        span['page'] = 2 * pnum + (
                            1 if span['bbox'][0] > (wid/2) else 0
                        )
                        if glue_horizon and spans:
                            if int(span['bbox'][1]) == int(spans[-1]['bbox'][1]):
                                spans[-1]['text'] += span['text']
                                x0, y0, _, _ = spans[-1]['bbox']
                                _, _, x1, y1 = span['bbox']
                                spans[-1]['bbox'] = (x0, y0, x1, y1)
                            else:
                                spans.append(span)
                        else:
                            spans.append(span)       
    return spans, lines


def set_dic_par(seq):
    lis = []
    buf = []
    pcounter = 0
    for sp in sorted(seq, key=lambda x:(x['page'], x['bbox'][1])):
        if not buf:
            sp['par'] = pcounter
            f = sp
            fbox = sp['bbox']
            buf.append(f)
        else:
            bbox = sp['bbox']
            if int(fbox[0]) == int(bbox[0]) \
                and int(fbox[2]) >= int(bbox[2]):
                sp['par'] = pcounter
                buf.append(sp)
            else:
                pcounter += 1
                lis.extend(buf)
                sp['par'] = pcounter
                f = sp
                fbox = sp['bbox']
                buf = [sp]
    lis.extend(buf)
    return lis


def sort_byreading(lis):
    """Sort by page, vertical and horizontal position"""
    return sorted(lis, key=lambda x: (x['page'],  x['bbox'][1], x['bbox'][0]))

def get_par_text(doc, min_linebreak=1, min_parlen=40):
    spans, lines = get_spans_lines(doc, glue_horizon=True)
    spans_sorted = sort_byreading(spans)
    lis = set_dic_par(spans_sorted)
    acc = 0
    valid_texts = []
    for idx, (key, l) in enumerate( gpk( lis, 'par').items() ):
        whole_tex = '\n'.join([sp['text'] for sp in l])
        cond1 = whole_tex.count('\n') < min_linebreak
        cond2 = len(whole_tex) < min_parlen
        if cond1 or cond2:
            continue
        valid_texts.append(whole_tex)
    return valid_texts


