import torch
from mhmoe import mlp_wide_triton_fwd, mlp_wide_triton_bwd2

class _FlashMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w1, w2, activation="relu"):
        o = mlp_wide_triton_fwd(x, w1, w2, activation)
        ctx.save_for_backward(x, w1, w2, o, activation)
        return o

    @staticmethod
    def backward(ctx, do):
        x, w1, w2, o, activation = ctx.saved_tensors
        dx, dw1, dw2 = mlp_wide_triton_bwd2(x, w1, w2, o, do, activation)
        return dx, dw1, dw2, None, None


class FlashMoE(torch.nn.Module):

    def __init__(self, activation="silu"):
        super().__init__()
        self.activation = activation

    def forward(self, x, w1, w2):
        return _FlashMoE.apply(x, w1, w2, self.activation)