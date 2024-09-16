import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from gpt2 import LayerNorm

@dataclass
class MyMambaConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    d_model: int = 768
    d_state: int = 16
    d_conv: int = 4
    n_expand: int = 2
    conv_bias: bool = False
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



class MyMamba(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            block = nn.ModuleList()
            block.append(LayerNorm(config.d_model, bias=config.bias))
            block.append(Mamba(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                  expand=config.n_expand, conv_bias=config.conv_bias, bias=config.bias))
            self.blocks.append(block)

    def forward(self, x, y):
        # apply embedding
        emb = self.emb(x)
        # apply blocks
        for block in self.blocks:
            orig = emb
            for m in block:
                emb = m(emb)
            emb = emb + orig

        logits = self.head(emb)  # Changed from x to emb
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.0
