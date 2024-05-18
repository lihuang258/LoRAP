# Import necessary modules
import time
import torch
import torch.nn as nn
from utils.log import exported_logger as logger
# Import get_loaders function from data module within the same directory
from .data import get_loaders,test_loaders

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer,mode="test"
    )
    print("testloader load finished")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 400 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits      #输出的维度为[batch_size,seqlen,vocab_size][1,2048,32000]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
    print("nlls load finished")
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()
def model_eval(model,tokenizer,device,dataset="wikitext2"):
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer, mode="test")
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl


import torch
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
    # print(torch.cat(nlls, dim=-1).mean())
    end=time.time()
    print("evaluate time:%f"%(end-begin))
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()