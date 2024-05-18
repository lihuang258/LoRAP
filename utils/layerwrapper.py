import torch
import torch.nn as nn
# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.out_mean = torch.zeros((layer.weight.data.shape[0]), device=self.dev)
        self.out_matrix = torch.zeros((layer.weight.data.shape[0], layer.weight.data.shape[0]), device=self.dev)
        self.nsamples = 0
        self.batch_num=0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.type(torch.float32)
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.out_mean *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
        out = out.type(torch.float32)
        batch = out.shape[0]

        self.out_matrix *= self.batch_num / (self.batch_num + batch)
        self.batch_num += batch
        self.out_mean += torch.mean(out, dim=0) / (self.nsamples)
        self.out_matrix += torch.matmul(out.t(), out)/self.batch_num



