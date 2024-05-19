import torch
import math
import time
import random
from torch import nn
from utils.data import get_loaders, prepare_calibration_input
from utils.layerwrapper import WrappedGPT
from utils.prune import prune_mlp
from utils.decomposition import cal_saved_rank,decopose
from transformers.activations import ACT2FN
from utils.modeling_llama import LinearLoW
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        name1
        child
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def lorap(args, model, tokenizer, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dataloader, _ = get_loaders(dataset=args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,
                                tokenizer=tokenizer, mode="train")
    print(f"calibdation data {args.dataset} loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args,model, dataloader,device)
    layers = model.model.layers
    if "mlp" in args.sublayer:
        allow_name = ['up_proj', 'gate_proj', 'down_proj']
    else:
        allow_name = []
    if "self_attn" in args.sublayer:
        allow_name = allow_name+['q_proj', 'k_proj', 'v_proj', 'o_proj']
    for i in range(len(layers)):
        layer = layers[i]
        all_set = find_layers(layer)
        subset={key: all_set[key] for key in all_set if key.split(".")[1] in allow_name}
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:  # 对每个子层添加一个钩子函数，这样在每次前向计算时该函数都会被调用
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        self_attn_set = {key: subset[key] for key in subset if key.split(".")[0] == "self_attn"}
        mlp_set = {key: subset[key] for key in subset if key.split(".")[0] == "mlp"}
        k_att, k_min, k_max = cal_saved_rank(args.sparsity_ratio, model, args.para_allocate)
        #########compress self_attn##########
        if "self_attn" in args.sublayer:
            self_aten = layer.self_attn
            for name in self_attn_set:
                if name in ["self_attn.q_proj", "self_attn.k_proj"]:
                    print(f"Decompose layer {i} name {name} k{k_min}")
                    L,R=decopose(name,subset,wrapped_layers,saved_rank=k_min, method=args.deco_method,return_dict=False)
                    if args.real_com:
                        new_layer = LinearLoW(subset[name].weight.shape[1], subset[name].weight.shape[0], saved_rank=k_min)
                        new_layer.initialize_weight(L, R)
                        setattr(self_aten, name.split(".")[-1], new_layer)
                    else:
                        subset[name].weight.data=L@R
                elif name in ["self_attn.v_proj", "self_attn.o_proj"] and k_max < int(model.config.hidden_size/2):
                    print(f"Decompose layer {i} name {name} k{k_max}")
                    L, R = decopose(name, subset, wrapped_layers, saved_rank=k_max, method=args.deco_method,
                                    return_dict=False)
                    if args.real_com:
                        new_layer = LinearLoW(subset[name].weight.shape[1], subset[name].weight.shape[0], saved_rank=k_max)
                        new_layer.initialize_weight(L, R)
                        setattr(self_aten, name.split(".")[-1], new_layer)
                    else:
                        subset[name].weight.data = L @ R
        if "mlp" in args.sublayer:
            if args.mlp_compress_method == "decom":
                shape = layer.mlp.up_proj.weight.data.shape
                k_mlp = int((1 - args.sparsity_ratio) * len(subset) * shape[0] * shape[1] / (shape[0] + shape[1]))
                for name in mlp_set:
                    print(f"Decompose layer {i} name {name} k{k_mlp}")
                    L,R=decopose(name,subset,wrapped_layers,saved_rank=k_min, method=args.deco_method,return_dict=False)
                    subset[name].weight.data = L @ R
            elif args.mlp_compress_method == "prune":
                prune_mlp(args, subset, wrapped_layers, device)
                layer.mlp.intermediate_size= layer.mlp.up_proj.weight.data.shape[0]    ###No practical impact
                print(f"Prune layer {i}")
            else:
                raise ValueError("Unknown mlp compression method")
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
    if "mlp" in args.sublayer:
        model.config.intermediate_size = model.model.layers[0].mlp.up_proj.weight.data.shape[0]
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def after_tune_SVD(model,lora_rank,sparsity_ratio,allocate_ratio):
    layers = model.model.layers
    k_att, k_min, k_max = cal_saved_rank(sparsity_ratio, model,allocate_ratio)
    k_min+=lora_rank
    k_max+=lora_rank
    for i in range(len(layers)):
        print("Decompose layer %d"%i)
        layer = layers[i]
        self_aten = layer.self_attn
        subset = find_layers(layer)
        for name in subset:
            if name.startswith("self"):
                if name.split(".")[1] in ["q_proj", "k_proj"]:
                    print(f"SVD layer {i} name {name} k{k_min}")
                    L, R = decopose(name,subset,wrapped_layers=None,saved_rank=k_min,method="SVD",return_dict=False)
                    new_layer = LinearLoW(subset[name].weight.shape[1], subset[name].weight.shape[0],saved_rank=k_min)
                    new_layer.initialize_weight(L, R)
                    setattr(self_aten, name.split(".")[1], new_layer)
                elif name.split(".")[1] in ["v_proj", "o_proj"] and k_max < int(model.config.hidden_size / 2):
                    print(f"SVD layer {i} name {name} k{k_max}")
                    L, R = decopose(name, subset, wrapped_layers=None, saved_rank=k_min, method="SVD",
                                    return_dict=False)
                    new_layer = LinearLoW(subset[name].weight.shape[1], subset[name].weight.shape[0], saved_rank=k_max)
                    new_layer.initialize_weight(L, R)
                    setattr(self_aten, name.split(".")[1], new_layer)
