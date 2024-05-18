import torch

def cal_saved_rank(sparsity_ratio, model,allocate_ratio=3):
    saevd_ratio = 1 - sparsity_ratio
    hidden_size = model.config.hidden_size
    k_att = int(saevd_ratio * hidden_size/2)
    k_max = int(k_att * 2*allocate_ratio/(1+allocate_ratio))
    if k_max >= hidden_size/2:
        k_max = int(hidden_size/2)
        k_min = int(saevd_ratio * hidden_size) - int(hidden_size/2)
    else:
        k_min = int(k_att * 2/(1+allocate_ratio))
    return k_att,k_min, k_max

def decopose(name,subset,wrapped_layers,saved_rank,method="AWSVD",return_dict=False):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param           imp: The input tensor, of shape (N, H),representing significance
    :param      saved_rank: rank_of_decomposed_matrix
    :param     return_dict: Return a dict if True, else return a tuple (L, R)
    :return:
    """
    """parameter_ratio = rank * (H + W) / (H * W)"""
    weight= subset[name].weight.data
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    weight = weight.to(dtype=torch.float32)  # Convert to float32
    if method == "AWSVD":
        imp= wrapped_layers[name].scaler_row
        imp = imp.to(dtype=torch.float32)
        imp = imp.T
        col_sums_sqrt = torch.sqrt(imp)
        col_sums_sqrt[col_sums_sqrt == 0] = 0.0001
        D = torch.diag(col_sums_sqrt)
        D_inv = torch.inverse(D)
        WD = torch.mm(weight, D)
        U, S, Vh = torch.linalg.svd(WD, full_matrices=False)
        L = U @ (torch.sqrt(torch.diag(S)[:, :saved_rank]))
        R = torch.sqrt(torch.diag(S)[:saved_rank, :]) @ Vh @ D_inv
    elif method == "SVD":
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        U_k = U[:, :saved_rank]
        s_k = torch.diag(S[:saved_rank])
        V_k = Vh[:saved_rank, :]
        L = U_k @ s_k
        R = V_k
    elif method == "AFM":
        mean = wrapped_layers[name].out_mean
        E_yy = wrapped_layers[name].out_matrix
        mean = mean.unsqueeze(1)
        E_y_y = torch.matmul(mean, mean.T)
        covariance_matrix = E_yy - E_y_y
        S, U = torch.linalg.eigh(covariance_matrix)
        sorted_indices = torch.argsort(S, descending=True)
        U = U[:, sorted_indices]
        principal_components = U[:, :saved_rank]
        R = (principal_components.T @ weight).to(dtype=torch.float16)
        L = principal_components.to(dtype=torch.float16)

    L=L.to(dtype=torch.float16)
    R=R.to(dtype=torch.float16)
    if return_dict:
        return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'saved_rank': saved_rank}
    else:
        return L, R