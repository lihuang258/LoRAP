import torch
from torch import nn
def prune_in_channels(layer: nn.Module, idxs: list[int],device) -> nn.Module:
    mask = torch.ones(layer.weight.shape[1], dtype=torch.bool).to(device)
    mask[idxs] = 0
    new_matrix = torch.index_select(layer.weight.data, 1, mask.nonzero().squeeze(1))
    layer.in_features = layer.in_features-len(idxs)
    layer.weight = nn.Parameter(new_matrix)
    return layer

def prune_out_channels(layer: nn.Module, idxs: list[int],device) -> nn.Module:
    mask = torch.ones(layer.weight.shape[0], dtype=torch.bool).to(device)
    mask[idxs] = 0
    new_matrix = torch.index_select(layer.weight.data, 0, mask.nonzero().squeeze(1))
    layer.out_features = layer.out_features-len(idxs)
    layer.weight = nn.Parameter(new_matrix)
    if layer.bias is not None:
        masks = torch.ones(layer.bias.shape, dtype=torch.bool).to(device)
        masks[idxs] =0
        new_bias=torch.index_select(layer.bias, 0,masks.nonzero().squeeze())
        # 使用掩码获取非零元素
        layer.bias = nn.Parameter(new_bias)
    return layer

def prune_mlp(args,subset,wrapped_layers,device):
    sparsity = args.sparsity_ratio
    saved_ratio = args.mlp_least_save_ratio
    up = torch.abs(subset["mlp.up_proj"].weight.data) * torch.sqrt(wrapped_layers["mlp.up_proj"].scaler_row.reshape((1, -1)))
    down = torch.abs(subset["mlp.down_proj"].weight.data) * torch.sqrt(wrapped_layers["mlp.down_proj"].scaler_row.reshape((1, -1)))
    gate = torch.abs(subset["mlp.gate_proj"].weight.data) * torch.sqrt(wrapped_layers["mlp.gate_proj"].scaler_row.reshape((1, -1)))

    row_norms_up = torch.norm(up, p=2, dim=1)
    row_norms_gate = torch.norm(gate, p=2, dim=1)
    row_norms_down = torch.norm(down, p=2, dim=0)

    toal = row_norms_gate + row_norms_down + row_norms_up
    sorted_values, sorted_indices = torch.sort(toal)
    indices_to_zero = sorted_indices[int(up.shape[0] * saved_ratio):int((sparsity + saved_ratio) * up.shape[0])]
    prune_in_channels(layer=subset["mlp.down_proj"], idxs=indices_to_zero, device=device)
    prune_out_channels(layer=subset["mlp.up_proj"], idxs=indices_to_zero, device=device)
    prune_out_channels(layer=subset["mlp.gate_proj"], idxs=indices_to_zero, device=device)
    # if args.real_com:
    #     prune_in_channels(layer=subset["mlp.down_proj"], idxs=indices_to_zero,device=device)
    #     prune_out_channels(layer=subset["mlp.up_proj"], idxs=indices_to_zero,device=device)
    #     prune_out_channels(layer=subset["mlp.gate_proj"], idxs=indices_to_zero,device=device)
    # else:
    #     mask = torch.ones(up.shape[0], dtype=torch.bool).to(device)
    #     mask[indices_to_zero] = 0
    #     subset["mlp.down_proj"].weight.data*= mask
    #     subset["mlp.up_proj"].weight.data= mask.unsqueeze(-1)*subset["mlp.up_proj"].weight.data
    #     subset["mlp.gate_proj"].weight.data= mask.unsqueeze(-1)*subset["mlp.gate_proj"].weight.data
    #     return None