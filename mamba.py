import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from gpt2 import LayerNorm, MLP

@dataclass
class MyMambaConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    d_model: int = 768
    d_ffn: int = 768 * 4
    d_state: int = 128
    d_conv: int = 4
    n_expand: int = 2
    dropout: float = 0.1
    conv_bias: bool = False
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    enable_mlp: bool = True
    use_mamba2: bool = True


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)
        self.silu    = nn.SiLU()
        self.c_proj  = nn.Linear(config.d_ffn, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MyBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        if config.use_mamba2:
            self.mamba = Mamba2(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                  expand=config.n_expand, conv_bias=config.conv_bias, bias=config.bias)
        else:
            self.mamba = Mamba(d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                  expand=config.n_expand, conv_bias=config.conv_bias, bias=config.bias)
        if config.enable_mlp:
            self.ln2 = LayerNorm(config.d_model, bias=config.bias)
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        if self.config.enable_mlp:
            x = x + self.mlp(self.ln2(x))
        return x


class MyMamba(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            self.blocks.append(MyBlock(config))

    def forward(self, x, y):
        # apply embedding
        emb = self.emb(x)
        # apply blocks
        for block in self.blocks:
            emb = block(emb)

        logits = self.head(emb)  # Changed from x to emb
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.0
