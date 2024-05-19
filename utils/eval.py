# Import necessary modules
import time
import torch
from .data import get_loaders,test_loaders
import numpy as np
from tqdm import tqdm

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = test_loaders(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    begin=time.time()
    for batch in tqdm(test_lodaer):
        batch = batch.to("cuda")
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    end=time.time()
    print("evaluate time:%f"%(end-begin))
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()