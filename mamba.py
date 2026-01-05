import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mamba_ssm import Mamba, Mamba2
from gpt2 import LayerNorm
from lightning.fabric.strategies import FSDPStrategy
from moe_layer_simple import MoE as Sigma_MoE
# from scattermoe.mlp import MLP as Scatter_MLP
from moe import FlashMoE, MoE

@dataclass
class MyMambaConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    d_model: int = 768
    d_ffn: int = 3072 # 1536
    d_state: int = 128
    d_conv: int = 4
    n_expand: int = 2
    n_head: int = 3
    n_experts: int = 48
    n_expert_capacity: int = 4
    dropout: float = 0.1
    conv_bias: bool = False
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_mamba2: bool = True
    enable_mlp: bool = True
    use_moe: bool = False
    use_flash_moe: bool = False
    use_sigma_moe: bool = False
    use_scatter_moe: bool = False


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


class Scatter_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp = Scatter_MLP(
            input_size=config.d_model,
            hidden_size=config.d_ffn,
            activation=nn.SiLU(),
            num_experts=config.n_experts,
            top_k=config.n_expert_capacity
        )
        self.choice = nn.Parameter(torch.empty(config.n_experts, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.choice, a=math.sqrt(5))

    def forward(self, x):
        # x = x.to(torch.bfloat16)
        weights = F.softmax(torch.einsum('bsd,ed->bse', x, self.choice), dim=-1) # (batch_size, n_seq_len, n_experts)
        # k = self.config.block_size * self.config.n_expert_capacity // self.config.n_experts # 1024 * 2 // 8 = 256
        k_weights, k_idxs = torch.topk(weights, self.config.n_expert_capacity) # (batch_size, n_seq_len, k)
        # P = F.one_hot(I, num_classes=self.config.block_size).to(x.dtype) # (batch_size, n_experts, k, n_seq_len)
        # x_in  = torch.einsum('beks,bsd->bekd', P, x) # (batch_size, n_experts, k, d_model)
        # x_mlp = torch.einsum('beki,edi->bekd', self.silu(torch.einsum('bekd,eod->beko', x_in, self.w1)), self.w2) # (batch_size, n_experts, k, d_model)
        # x_out = torch.einsum('beks,bek,bekd->bsd', P, G, x_mlp) # (batch_size, n_seq_len, d_model)
        # x_out = self.dropout(x_out)
        # print('Scatter MOE', x.dtype, weights.dtype, k_weights.dtype, k_idxs.dtype)
        x_out = self.mlp(x, k_weights, k_idxs)
        x_out = self.dropout(x_out)
        return x_out


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
            if config.use_scatter_moe:
                self.mlp_or_moe = Scatter_MoE(config)
            elif config.use_sigma_moe:
                self.mlp_or_moe = Sigma_MoE(
                    dmodel=config.d_model,
                    n_experts=config.n_experts,
                    expert_size=config.d_ffn//config.n_experts,
                    k=config.n_expert_capacity,
                    dropout=config.dropout,
                    selection_mode="silu")
            elif config.use_flash_moe:
                self.mlp_or_moe = FlashMoE(
                    d_model=config.d_model,
                    d_ffn=config.d_ffn,
                    n_head=config.n_head,
                    n_experts=config.n_experts,
                    n_expert_capacity=config.n_expert_capacity,
                    block_size=config.block_size,
                    dropout=config.dropout)
            elif config.use_moe:
                self.mlp_or_moe = MoE(
                    d_model=config.d_model,
                    d_ffn=config.d_ffn,
                    n_experts=config.n_experts,
                    n_expert_capacity=config.n_expert_capacity,
                    block_size=config.block_size,
                    dropout=config.dropout)
            else:
                self.mlp_or_moe = MLP(config)

    def forward(self, x):
        loss = None
        x = x + self.mamba(self.ln1(x))
        if self.config.enable_mlp:
            if self.config.use_sigma_moe:
                output, loss = self.mlp_or_moe(self.ln2(x))
                x = x + output
            else:
                x = x + self.mlp_or_moe(self.ln2(x))
        return x, loss


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
        # loss2 = torch.tensor([0.0], dtype=emb.dtype).to(emb.device)
        # apply blocks
        for block in self.blocks:
            emb, _loss = block(emb)
            # if _loss is not None:
            #     loss2 += _loss

        logits = self.head(emb)  # Changed from x to emb
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss # , loss2 / self.config.n_layer / self.config.n_experts

    def estimate_mfu(self, fabric, fwdbwd_per_iter, dt):
        return 0.0

    def model_flops_per_fwdbwd(self, fabric):
        N = self.get_num_params()
        if isinstance(fabric.strategy, FSDPStrategy):
            N = N * fabric.world_size
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        return flops_per_fwdbwd
