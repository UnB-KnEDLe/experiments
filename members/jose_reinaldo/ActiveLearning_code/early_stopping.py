import numpy as np
# NER open packages
import torch
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
# NER my packages
# from drive.MyDrive.partial_DAL.metrics import preprocess_pred_targ, IOBES_tags
from metrics import preprocess_pred_targ, IOBES_tags

def naive_var_computation(model, sent, word, tag, mask):
    """
    Naive implementation of a function to compute the variance of the gradients over a mini-batch (very inefficient)
    """
    aux = None
    flag_start = False
    try:
        model.word_encoder.flag_WordDrop = False
    except:
        pass
    for s, w, t, m in zip(sent, word, tag, mask):
        model.zero_grad()
        loss = model(s.unsqueeze(dim=0), w.unsqueeze(dim=0), t.unsqueeze(dim=0), m.unsqueeze(dim=0))
        loss.backward()
        if not flag_start:
            aux = model.decoder.linear.weight.grad.unsqueeze(dim=0)
            flag_start = True
        else:
            aux = torch.cat((aux, model.decoder.linear.weight.grad.unsqueeze(dim=0)), dim=0)
    return aux.var(dim=0)

def DUTE_ES(curr_epochs, confidence):
    """
    DUTE early stopping strategy

    Input:
        curr_epochs (integer): Number of training epochs for the previous iteration
        confidence (float in [0,1]): Confidence of the trained model on the unlabeled dataset

    Output:
        Number of training epochs to be used in the current epoch
    """
    return max(3, round(0.1 * (curr_epochs * (1 - confidence.item())) + 0.9 * curr_epochs))

def DevSetLoss_ES(model, validation_dataloader, performance_hist, patience, tag2idx):
    """
    Early stopping using the mean loss on the validation (development) set

    Input:
        model (pytorch): Trained model
        validation_dataloader (pytorch): dataloader (for the validation set) based on the active_dataset (from data.py)
        performance_hist (python list): List containing the history of the model's performance (validation loss)
        patience (integer value): Number of iterations without improvement to stop training
        tag2idx (python dictionary): maps NER classes to unique integer values

    Output:
        early_stop_flag (bool): Flag indicating whether to early stop training
        performance_hist (python list): List containing the history of the model's performance (validation loss)
    """
    # Compute mean loss over all mini-batches of the validation loss
    model.eval()
    loss_hist = []
    with torch.no_grad():
        for sent, tag, word, mask in validation_dataloader:
            sent = sent.to(next(model.parameters()).device)
            tag = tag.to(next(model.parameters()).device)
            word = word.to(next(model.parameters()).device)
            mask = mask.to(next(model.parameters()).device)
            loss = model(sent, word, tag, mask)
            loss_hist.append(loss.cpu())
    mean_loss = np.array(loss_hist, dtype=float).mean()
    performance_hist.append(mean_loss)
    min_loss = float('inf')
    min_idx = -1
    for idx, loss in enumerate(performance_hist):
        if loss <= min_loss:
            min_loss = loss
            min_idx = idx
    early_stop_flag = True if len(performance_hist) - min_idx > patience else False
    return early_stop_flag, performance_hist


def DevSetF1_ES(model, validation_dataloader, performance_hist, patience, tag2idx):
    """
    Early stopping using the f1-score computed on the validation (development) set

    Input:
        model (pytorch): Trained model
        validation_dataloader (pytorch): dataloader (for the validation set) based on the active_dataset (from data.py)
        performance_hist (python list): List containing the history of the model's performance (f1-score)
        patience (integer value): Number of iterations without improvement to stop training
        tag2idx (python dictionary): maps NER classes to unique integer values

    Output:
        early_stop_flag (bool): Flag indicating whether to early stop training
        performance_hist (python list): List containing the history of the model's performance (f1-score)
    """
    # Compute span-based f1-score on the validation set
    model.eval()
    with torch.no_grad():
        predictions, targets = preprocess_pred_targ(model, validation_dataloader, next(model.parameters()).device)
        predictions = IOBES_tags(predictions, tag2idx)
        targets = IOBES_tags(targets, tag2idx)
        micro_f1 = f1_score(targets, predictions, mode='strict', scheme=IOBES)
        performance_hist.append(micro_f1)

    # Check if patience has been reached (model not improving over consecutive epochs)
    max_f1 = -1
    max_idx = -1
    for idx, f1 in enumerate(performance_hist):
        if f1 >= max_f1:
            max_f1 = f1
            max_idx = idx
    early_stop_flag = True if len(performance_hist) - max_idx > patience else False

    return early_stop_flag, performance_hist

def BatchDisparity_ES(model, dataloader, S=5):
    """
    Function to compute the batch gradient disparity (BGD) between batches metric to be used for early stopping, 
    proposed by Forouzesh and Thiran (https://arxiv.org/pdf/2107.06665.pdf)

    Input:
        model (pytorch): Trained model
        dataloader (pytorch): dataloader (for the labeled set) based on the active_dataset (from data.py)
        S (integer value): Number of batches to be used to compute the BGD metric

    Output:
        returns the computed BGD metric
    """
    # Step 1: get loss std across all training samples
    loss_list = []
    for idx, (sent, tag, word, mask) in enumerate(dataloader):
        if idx >= 40:
            break
        sent = sent.to(next(model.parameters()).device)
        tag = tag.to(next(model.parameters()).device)
        word = word.to(next(model.parameters()).device)
        mask = mask.to(next(model.parameters()).device)
        loss = model(sent, word, tag, mask)
        loss_list.append(loss)
    loss_std = np.array(loss_list, dtype=float).std()
    
    # Step 2: Compute gradients for S samples
    grad_S = []
    for idx, (sent, tag, word, mask) in enumerate(dataloader):
        if idx >= S:
            break

        model.zero_grad()
        sent = sent.to(next(model.parameters()).device)
        tag = tag.to(next(model.parameters()).device)
        word = word.to(next(model.parameters()).device)
        mask = mask.to(next(model.parameters()).device)
        loss = model(sent, word, tag, mask)
        norm_loss = loss/loss_std
        norm_loss.backward()

        grad_temp = []
        for name, param in model.named_parameters():
            if param.grad != None:
                grad_temp.append(param.grad.view(-1))
        grad_S.append(torch.cat(grad_temp).cpu())
        
    # Step 3: Compute gradient disparity for all pairs of samples in S 
    D = 0.0 
    # print(f'S is {S}\nlen(grad_S) is {len(grad_S)}')
    for i in range(len(grad_S)):
        for j in range(i):
            D += (grad_S[i] - grad_S[j]).norm(p=2)
    return D/(S*(S-1))