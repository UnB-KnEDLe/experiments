import torch
# from drive.MyDrive.partial_DAL.utils import budget_limit
from utils import budget_limit
import random
from random import sample

def least_confidence(model, dataloader, budget, device):
    """
    Least confidence sampling function

    Input:
        model (pytorch): Trained model
        dataloader (pytorch): Dataloader for the active_dataset (from data.py)
        budget (integer value): Number of words that can be queried in 1 iteration of the active learning algorithm
        device (string): Device in which the model is (cuda or cpu)

    Output:
        least_confidence_idx (python list): List of ordered indices for the unlabeled samples (ordered by the model's confidence from most to least uncertain sample)
        residual_budget (integer value): Remaining budget (unused for this iteration)
    
    """
    with torch.no_grad():
        full_prob = torch.Tensor().to(device)
        for sent, tag, word, mask in dataloader:
            sent = sent.to(device)
            tag = tag.to(device)
            word = word.to(device)
            mask = mask.to(device)
            prediction, prob = model.decode(sent, word, mask)

            # output = torch.nn.functional.softmax(output, dim=2).max(dim=2).values * mask
            # prob = torch.ones(output.shape[0]).to(device)
            # for i, j in torch.nonzero(output, as_tuple=False):
                # prob[i] *= output[i, j]

            full_prob = torch.cat((full_prob, prob), dim=0)
        idx = torch.argsort(full_prob)
        least_confidence_idx, residual_budget = budget_limit(idx, budget, dataloader)
        return least_confidence_idx, residual_budget

def random_sampling(dataloader, budget, model=None, device=None, seed = None):
    """
    Random sampling function

    Input:
        dataloader (pytorch): Dataloader for the active_dataset (from data.py)
        budget (integer value): Number of words that can be queried in 1 iteration of the active learning algorithm
        model (pytorch): unused
        device (string): unused
        seed (integer value): Used to generate reproducible "random" sampling

    Output:
        random_idx (python list): List of ordered indices for the unlabeled samples (ordered randomly)
        residual_budget (integer value): Remaining budget (unused for this iteration)
    
    """
    if seed:
        random.seed(seed)
    idx = sample(range(len(dataloader.dataset)), len(dataloader.dataset))
    random_idx, residual_budget = budget_limit(idx, budget, dataloader)
    return random_idx, residual_budget

def normalized_least_confidence(model, dataloader, budget, device):
    """
    Normalized least confidence (also called MNLP from https://arxiv.org/abs/1707.05928) sampling function

    Input:
        dataloader (pytorch): Dataloader for the active_dataset (from data.py)
        budget (integer value): Number of words that can be queried in 1 iteration of the active learning algorithm
        model (pytorch): Trained model
        device (string): Device in which the model is (cuda or cpu)

    Output:
        norm_least_confidence_idx (python list): List of ordered indices for the unlabeled samples (ordered by the model's normalized confidence from most to least uncertain sample)
        residual_budget (integer value): Remaining budget (unused for this iteration)
    
    """
    with torch.no_grad():
        full_prob = torch.Tensor().to(device)
        for sent, tag, word, mask in dataloader:
            sent = sent.to(device)
            tag = tag.to(device)
            word = word.to(device)
            mask = mask.to(device)
            prediction, prob = model.decode(sent, word, mask)
            
            norm_prob = (prob.log() / mask.sum(dim=1)).exp()
            # output = torch.nn.functional.log_softmax(output, dim=2).max(dim=2).values * mask
            # prob = output.sum(dim=1) / mask.sum(dim=1)
            full_prob = torch.cat((full_prob, norm_prob), dim=0)
        idx = torch.argsort(full_prob)
        norm_least_confidence_idx, residual_budget = budget_limit(idx, budget, dataloader)
        return norm_least_confidence_idx, residual_budget