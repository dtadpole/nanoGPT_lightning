import torch
import torch.nn as nn

class MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_expand * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_expand * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.c_fc.weight)
        nn.init.xavier_uniform_(self.c_proj.weight)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x):
        return MLPFunction.apply(x, self.c_fc.weight, self.c_fc.bias)
