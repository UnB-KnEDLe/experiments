import torch

def self_label(model, dataloader, query_idx, min_confidence, collate_object, refinement_iter, word2idx, device):
    """
    Labels high confidence tokens with self-labeling and low confidence tokens using the oracle

    Returns:
        1 - Number of active-labeled tokens (i.e. annotated by the oracle)
        2 - Number of self-labeled tokens (i.e. annotated by the trained model)
        3 - Number of wrongly self-labeled tokens
    """
    with torch.no_grad():
        # Step 1: Create a batch with the queried samples
        aux = [dataloader.dataset.unlabeled_sentences[i] for i in query_idx]
        ssent = [dataloader.dataset.sentences[i] for i in aux]
        wword = [dataloader.dataset.words[i] for i in aux]
        ttags = [dataloader.dataset.tags[i] for i in aux]
        unpadded_batch = [[s, t, w] for s, w, t in zip(ssent, wword, ttags)]
        batch = collate_object(unpadded_batch)

        # Step 2: Identify tokens to be actually labeled by the oracle
        ssent, ttags, wword, mmask = batch[0], batch[1], batch[2], batch[3]
        ssent, ttags, wword, mmask = ssent.to(device), ttags.to(device), wword.to(device), mmask.to(device)
        pred, prob = model.decode(ssent, wword, mmask, return_token_probs=True)
        safe_token_mask = prob >= min_confidence
        safe_token_mask = safe_token_mask * mmask

        # Step 3: Change labels (Replace human labeled tags by real tags) (Replace self-labeled tags by model predictions)
        new_labels = torch.where(safe_token_mask==True, pred, ttags)

        # Step 4: Prediction refinement. Inspired by the iterative refinement used in (https://arxiv.org/pdf/2104.07284.pdf)
        for _ in range(refinement_iter):
            new_labels = model.refined_decode(ssent, wword, new_labels, safe_token_mask)
            new_labels = torch.where(safe_token_mask==True, new_labels, ttags)
        
        new_labels[ssent == word2idx['<START>']] = 1
        new_labels[ssent == word2idx['<END>']] = 1
        safe_token_mask[ssent == word2idx['<START>']] = False
        safe_token_mask[ssent == word2idx['<END>']] = False
        # Step 5: token confidence aware active-self labeling
        for idx, new_tags, new_mask in zip(aux, new_labels, mmask):
            dataloader.dataset.tags[idx] = new_tags[:new_mask.sum()].tolist()

    # Step 5: Return labeling statistics 
    # 1 - Number of active labeled tokens -  ignoring special <START> <END> tokens
    # 2 - Number of self-labeled tokens
    # 3 - Number of wrongly self-labeled tokens - ignoring special <START> <END> tokens

    return ((safe_token_mask != True)*mmask).sum().item() - 2*ssent.shape[0], safe_token_mask.sum().item(), (((new_labels != ttags)*mmask).sum()).item()