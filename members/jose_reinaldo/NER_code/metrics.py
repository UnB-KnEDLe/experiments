import numpy as np
# from drive.MyDrive.NER_code.utils import find_iobes_entities, find_iobes_entities2
from utils import find_iobes_entities, find_iobes_entities2
import torch

def preprocess_pred_targ(model, dataloader, device):
    """
    Transform predictions and targets from torch.Tensors to lists
    """
    full_pred = []
    full_targ = []
    with torch.no_grad():
        for sent, tag, word, mask in dataloader:
            sent = sent.to(device)
            tag = tag.to(device)
            word = word.to(device)
            mask = mask.to(device)
            pred, _ = model.decode(sent, word, mask)
            
            for i in range(len(pred)):
                full_pred.append(pred[i, :mask[i].sum()].tolist())
                full_targ.append(tag[i, :mask[i].sum()].tolist())
    
    return full_pred, full_targ

def IOBES_tags(predictions, tag2idx):
    """
    Transform tags from indices to class name (string var)
    """
    idx2tag = {}
    for tag in tag2idx:
        idx2tag[tag2idx[tag]] = tag
    
    IOBES_tags = predictions.copy()
    for i in range(len(IOBES_tags)):
        for j in range(len(IOBES_tags[i])):
            IOBES_tags[i][j] = idx2tag[IOBES_tags[i][j]]
    return IOBES_tags

def exact_f1_score(model, dataloader, device, tag2idx):
    TP = np.array([0 for _ in range((model.num_classes-2)//2)])
    FP = np.array([0 for _ in range((model.num_classes-2)//2)])
    FN = np.array([0 for _ in range((model.num_classes-2)//2)])

    for sent, tag, word, mask in dataloader:
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        mask = mask.to(device)
        pred, _ = model.eval().decode(sent, word, mask)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = find_iobes_entities(pred[i], tag2idx)
            real_entities = find_iobes_entities(tag[i], tag2idx)
            for entity in predicted_entities:
                if entity in real_entities:
                    TP[(entity[2]//2)-1] += 1
                else:
                    FP[(entity[2]//2)-1] += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN[(entity[2]//2)-1] += 1

    precision = TP/(TP+FP+0.000001)
    recall = TP/(TP+FN+0.000001)
    macro_f1 = 2*(precision*recall)/(precision+recall)
    macro_f1 = [0 if np.isnan(f1) else f1 for f1 in macro_f1]
    occurrences = TP + FN
    averaged_macro_f1 = (macro_f1*occurrences).sum()/occurrences.sum()

    TP = TP.sum()
    FP = FP.sum()
    FN = FN.sum()
    if TP == 0:
        return 0, 0

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    micro_f1 = 2*(precision*recall)/(precision+recall)

    return averaged_macro_f1, micro_f1

def exact_micro_f1_score(model, dataloader, device, tag2idx):
    TP = 0
    FP = 0
    FN = 0

    for sent, tag, word, mask in dataloader:
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        pred, _ = model.eval().decode(sent, word, mask)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = find_iobes_entities2(pred[i], tag2idx)
            real_entities = find_iobes_entities2(pred[i], tag2idx)
            for entity in predicted_entities:
                if entity in real_entities:
                    TP += 1
                else:
                    FP += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN += 1
    if TP == 0:
      f1 = 0
      return f1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall)/(precision+recall)
    return f1

def exact_macro_f1_score(model, dataloader, device):
    TP = np.array([0 for _ in range((model.num_classes-2)//2)])
    FP = np.array([0 for _ in range((model.num_classes-2)//2)])
    FN = np.array([0 for _ in range((model.num_classes-2)//2)])

    for sent, tag, word, mask in dataloader:
        sent = sent.to(device)
        tag = tag.to(device)
        word = word.to(device)
        pred, _ = model.eval().decode(sent, word)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = find_entities(pred[i])
            real_entities = find_entities(tag[i])
            for entity in predicted_entities:
                if entity in real_entities:
                    TP[(entity[2]//2)-1] += 1
                else:
                    FP[(entity[2]//2)-1] += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN[(entity[2]//2)-1] += 1

    precision = TP/(TP+FP+0.000001)
    recall = TP/(TP+FN+0.000001)
    f1 = 2*(precision*recall)/(precision+recall)

    occurrences = TP + FN
    return occurrences, f1
