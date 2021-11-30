import numpy as np
import torch

def preprocess_pred_targ(model, dataloader, device):
    """
    Generates predictions using the trained model and the dataloader, then transforms predictions and targets from torch.Tensors to lists

    Input:
        dataloader (pytorch): Dataloader for the active_dataset (from data.py)
        model (pytorch): Trained model
        device (string): Device in which the model is (cuda or cpu)

    Output:
        full_pred (python list): List of all predictions for the unlabeled samples
        full_targ (python list): List of all targets (ground-truth)
    
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
    Transforms tags from indices (integer value) to class name (string)

    Input:
        predictions (list): List of predicted tags for all "unlabeled" samples
        tag2idx (python dictionary): maps classes to unique integer values

    Output:
        IOBES_tags (list): List containing the predictions but with class names (string) instead of indices (integer)
    """
    idx2tag = {}
    for tag in tag2idx:
        idx2tag[tag2idx[tag]] = tag
    
    IOBES_tags = predictions.copy()
    for i in range(len(IOBES_tags)):
        for j in range(len(IOBES_tags[i])):
            IOBES_tags[i][j] = idx2tag[IOBES_tags[i][j]]
    return IOBES_tags

