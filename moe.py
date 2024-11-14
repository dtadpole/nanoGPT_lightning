import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoE(nn.Module):

    def __init__(self, d_model=768, d_ffn=1536, n_experts=8, n_expert_capacity=2, block_size=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.block_size = block_size
        self.n_experts = n_experts
        self.n_expert_capacity = n_expert_capacity
        self.choice = nn.Parameter(torch.empty(n_experts, d_model))
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ffn))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ffn, d_model))
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.choice, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x):
        choice = F.softmax(torch.einsum('bsd,ed->bse', x, self.choice), dim=-1) # (batch_size, n_seq_len, n_experts)
        k = self.block_size * self.n_expert_capacity // self.n_experts # 1024 * 2 // 8 = 256
        G, I = torch.topk(torch.transpose(choice, -1, -2), k) # (batch_size, n_experts, k)
        P = F.one_hot(I, num_classes=self.block_size).to(x.dtype) # (batch_size, n_experts, k, n_seq_len)
        x_in  = torch.einsum('beks,bsd->bekd', P, x) # (batch_size, n_experts, k, d_model)
        x_mlp = torch.einsum('beki,eid->bekd', self.silu(torch.einsum('bekd,edo->beko', x_in, self.w1)), self.w2) # (batch_size, n_experts, k, d_model)
        # x_mlp = torch.einsum('beki,eid->bekd', self.silu(torch.einsum('beks,bsd,edo->beko', P, x, self.w1)), self.w2) # (batch_size, n_experts, k, d_model)
        # x_e   = torch.einsum('beks,bekd->besd', P, x_mlp) # (batch_size, n_experts, n_seq_len, d_model)
        # x_out = torch.einsum('beks,bek,besd->bsd', P, G, x_e) # (batch_size, n_seq_len, d_model)
        x_out = torch.einsum('beks,bek,bekd->bsd', P, G, x_mlp) # (batch_size, n_seq_len, d_model)
        x_out = self.dropout(x_out)
        return x_out


class FlashMoE(nn.Module):

    def __init__(self, d_model=768, d_ffn=384, n_head=6, n_experts=24, n_expert_capacity=4, block_size=1024, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        assert n_experts % n_head == 0
        assert d_model / n_head == 2 ** math.floor(math.log2(d_model / n_head))
        self.n_head = n_head
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.block_size = block_size
        self.n_experts = n_experts
        self.n_expert_capacity = n_expert_capacity
        self.choice = nn.Parameter(torch.empty(n_experts, d_model // n_head))
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model // n_head, d_ffn))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ffn, d_model // n_head))
        self.head = nn.Parameter(torch.empty(d_model, d_model))
        self.merge = nn.Parameter(torch.empty(d_model, d_model))
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.choice, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.head, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.merge, a=math.sqrt(5))

    def forward(self, x):
        x = torch.einsum('bsd,do->bso', x, self.head)
        x = x.view(x.shape[0], self.block_size * self.n_head, self.d_model // self.n_head) # (batch_size, n_seq_len * n_head, d_model // n_head)
        choice = F.softmax(torch.einsum('bsd,ed->bse', x, self.choice), dim=-1) # (batch_size, n_seq_len * n_head, n_experts)
        k = self.block_size * self.n_head * self.n_expert_capacity // self.n_experts # 1024 * 12 * 4 // 48 = 1024
        G, I = torch.topk(torch.transpose(choice, -1, -2), k) # (batch_size, n_experts, k)
        P = F.one_hot(I, num_classes=self.block_size * self.n_head).to(x.dtype) # (batch_size, n_experts, k, n_seq_len * n_head)
        x_in  = torch.einsum('beks,bsd->bekd', P, x) # (batch_size, n_experts, k, d_model // n_head)
        x_mlp = torch.einsum('beki,eid->bekd', self.silu(torch.einsum('bekd,edo->beko', x_in, self.w1)), self.w2) # (batch_size, n_experts, k, d_model // n_head)
        x_out = torch.einsum('beks,bek,bekd->bsd', P, G, x_mlp) # (batch_size, n_seq_len * n_head, d_model // n_head)
        x_out = self.dropout(x_out)
        x_out = x_out.view(x_out.shape[0], -1, self.d_model)
        x_out = torch.einsum('bsd,do->bso', x_out, self.merge)
        # x_out = self.dropout(x_out)
        return x_out


if __name__ == '__main__':
    moe = MoE()
    params = sum(p.numel() for p in moe.parameters())
    print(f'[MoE] Number of parameters: {params / 1e6:.2f}M')
    x = torch.randn(4, 1024, 768)
    print(moe(x).shape)

    fmoe = FlashMoE()
    params = sum(p.numel() for p in fmoe.parameters())
    print(f'[FlashMoE] Number of parameters: {params / 1e6:.2f}M')
    x = torch.randn(4, 1024, 768)
    print(fmoe(x).shape)
