# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from utils.log import exported_logger as logger
from torch import nn
# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2_2(nsamples, seed, seqlen, tokenizer,mode="train"):
    if mode=="train":
        random.seed(seed)
        trainloader = []
        # Load train and test datasets
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader, _
    if mode=="test":
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text'][:-1]), return_tensors='pt')
        logger.info("test data load finished")
        return 0, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer,mode="train"):
    # Load train and validation datasets
    if mode=="train":
        traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # Generate samples from training set
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                # if len(traindata[i]['text']) > 2000:
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        logger.info("train data load finished")
        return trainloader, _


    if mode=="test":
        valdata = load_dataset('allenai/c4', 'allenai--c4',
                               data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        # Prepare validation dataset
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return 0, valenc
def get_bookcorpus(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("bookcorpus")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # if len(traindata[i]['text']) > 2000:
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    logger.info("train data load finished")
    return trainloader, _
def get_ptb(nsamples, seed, seqlen, tokenizer):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    testenc = tokenizer("\n\n".join(valdata['sentence'][:-1]), return_tensors='pt')
    return 0, testenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None,mode="train"):
    if 'wikitext2' in name:
        return get_wikitext2_2(nsamples, seed, seqlen, tokenizer,mode=mode)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer,mode=mode)
    if 'bookcorpus' in name:
        return get_bookcorpus(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)

########从LLM-pruner中提取的代码
import random
import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from torch.utils.data.dataset import Dataset


def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata


def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def test_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader



def prepare_calibration_input(args,model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    if hasattr(model, 'hf_device_map'):
        # dev = model.hf_device_map["model.embed_tokens"]
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp     #将所有的数据都存入inps中
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask'] #更像是系统自己生成的mask
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:     #出现该错误是因为在forward中抛出了ValueError，所以这里直接pass掉
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids
