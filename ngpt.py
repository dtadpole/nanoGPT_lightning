"""
nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Based on the Nvidia paper: "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
https://arxiv.org/abs/2410.01131

Key innovations:
1. All hidden representations are normalized to lie on a unit hypersphere
2. No LayerNorm - replaced by L2 normalization
3. Normalized updates: h = normalize(h + α * update) instead of h = h + update
4. QK normalization with learnable temperature in attention
5. Learnable scaling factors (α) for attention and MLP contributions
6. Weight matrices are normalized to unit norm
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from lightning.fabric.strategies.fsdp import FSDPStrategy


def l2_normalize(x, dim=-1, eps=1e-12):
    """L2 normalize along the specified dimension."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb

def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

class NormalizedCausalSelfAttention(nn.Module):
    """
    Normalized Causal Self-Attention for nGPT.
    
    Key differences from standard attention:
    1. Q, K, V projections use normalized weights
    2. Q and K are L2 normalized before computing attention
    3. Learnable temperature/scale for attention logits (s_qk)
    4. No dropout (nGPT paper doesn't use dropout)g
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_sqk_scaling = config.use_sqk_scaling
        
        # Q and K projections (scale not used since they get L2-normalized after)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        # V and output projections with learnable scale
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Learnable attention temperature (s_qk)
        # Q and K are L2-normalized, so this controls attention sharpness
        self.s_qk = nn.Parameter(torch.ones(self.n_embd) * math.sqrt(self.head_dim))
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        self.q_proj.weight.data.copy_(l2_normalize(self.q_proj.weight.data, 1))
        self.k_proj.weight.data.copy_(l2_normalize(self.k_proj.weight.data, 1))
        self.v_proj.weight.data.copy_(l2_normalize(self.v_proj.weight.data, 1))
        self.c_proj.weight.data.copy_(l2_normalize(self.c_proj.weight.data, 1))

    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V with normalized projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.head_dim).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        sqk = self.s_qk.view(1, 1, self.n_head, self.head_dim)

        # Normalize Q and K (key innovation of nGPT)
        q = sqk * l2_normalize(q, dim=-1)
        k = sqk * l2_normalize(k, dim=-1)
        
        # Compute attention with learnable temperature s_qk
        sqrt_head_dim = (self.head_dim) ** 0.5
        softmax_scale = sqrt_head_dim

        if self.flash:
            # Flash attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                dropout_p=0.0,
                scale=softmax_scale,
                is_causal=True
            )
            y = y.to(dtype=q.dtype)
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * softmax_scale
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        
        return y


class NormalizedMLP(nn.Module):
    """
    Normalized Gated MLP (SwiGLU) for nGPT.
    
    Uses SiLU-gated linear units with normalized weight matrices.
    Formula: hidden = (u · s_u) ⊙ σ(g · s_g)
             output = s_d · (W_down · hidden)
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.n_expand * config.n_embd
        self.n_embd = config.n_embd
        
        # Normalized linear layers
        self.w_gate = nn.Linear(config.n_embd, hidden_dim)
        self.w_up = nn.Linear(config.n_embd, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, config.n_embd)

        self.scale_factor = math.sqrt(config.n_embd)
        
        # SiLU activation for the gate
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)
        self.w_gate.weight.data.copy_(l2_normalize(self.w_gate.weight.data, 1))
        self.w_up.weight.data.copy_(l2_normalize(self.w_up.weight.data, 1))
        self.w_down.weight.data.copy_(l2_normalize(self.w_down.weight.data, 0))

    def forward(self, x):
        # Gated MLP: hidden = (u · s_u) ⊙ σ(g · s_g)
        gate = self.act(self.w_gate(x) * self.scale_factor)
        up = self.w_up(x) * self.scale_factor
        hidden = gate * up
        # Down projection
        x = self.w_down(hidden)
        return x


class NormalizedBlock(nn.Module):
    """
    Normalized Transformer Block for nGPT.
    
    Key difference from standard transformer:
    - Uses normalized update: h = normalize(h + α * update)
    - Learnable scaling factors α_attn and α_mlp
    - No LayerNorm
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = NormalizedCausalSelfAttention(config)
        self.mlp = NormalizedMLP(config)
        
        # Learnable scaling factors for residual updates (α_a and α_m in paper)
        # Initialized small to start with small updates
        # α = 1/sqrt(2*n_layer) is a good initialization
        self.scale_factor = 1.0 / math.sqrt(2 * config.n_layer)
        self.alpha_attn = nn.Parameter(torch.ones(config.n_embd) * self.scale_factor)
        self.alpha_mlp = nn.Parameter(torch.ones(config.n_embd) * self.scale_factor)

    def forward(self, x):
        # Input x should already be normalized (on unit hypersphere)
        
        # Attention update: h = normalize(h + α_a ⊙ Attn(h_a - h))
        x = l2_normalize(x + self.alpha_attn * (self.attn(x) - x), dim=-1)
        
        # MLP update: h = normalize(h + α_m ⊙ MLP(h_m - h))
        x = l2_normalize(x + self.alpha_mlp * (self.mlp(x) - x), dim=-1)
        
        return x


@dataclass
class NGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_expand: int = 4  # MLP expansion factor
    dropout: float = 0.0  # nGPT doesn't use dropout (representations are normalized)
    bias: bool = False  # nGPT doesn't use bias
    use_sqk_scaling: bool = True    


class NGPT(nn.Module):
    """
    nGPT: Normalized GPT model.
    
    All representations live on a unit hypersphere.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.normalize_weights_first_time = True

        # Normalized embeddings
        # Learnable scaling for combining token and position embeddings
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([NormalizedBlock(config, i) for i in range(config.n_layer)])

        # Output head with normalized weights
        # Note: weight tying with wte will be handled specially
        self.s_z = torch.nn.Parameter(torch.ones(config.vocab_size, dtype=torch.float32) * math.sqrt(config.n_embd))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def normalize_weights(self):
        if self.normalize_weights_first_time:
            self.normalize_weights_first_time = False
            print("Normalizing weights for the first time...")
        # Normalize embeddings and output head
        self.wte.weight.data.copy_(l2_normalize(self.wte.weight.data, 1))         # V, n_embd
        self.lm_head.weight.data.copy_(l2_normalize(self.lm_head.weight.data, 1))   # V, n_embd
        
        for layer_idx in range(0, self.config.n_layer):
            block = self.blocks[layer_idx]
            # print(f"Normalizing weights for block {layer_idx}...")
            # Attention weights
            block.attn.q_proj.weight.data.copy_(l2_normalize(block.attn.q_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.k_proj.weight.data.copy_(l2_normalize(block.attn.k_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.v_proj.weight.data.copy_(l2_normalize(block.attn.v_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.c_proj.weight.data.copy_(l2_normalize(block.attn.c_proj.weight.data, 1))             # n_embd, n_embd
            
            # MLP weights
            block.mlp.w_gate.weight.data.copy_(l2_normalize(block.mlp.w_gate.weight.data, 1))               # hidden, n_embd
            block.mlp.w_up.weight.data.copy_(l2_normalize(block.mlp.w_up.weight.data, 1))                   # hidden, n_embd
            block.mlp.w_down.weight.data.copy_(l2_normalize(block.mlp.w_down.weight.data, 0))               # n_embd, hidden

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def _init_weights(self):
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        self.wte.weight.data.copy_(l2_normalize(self.wte.weight.data, 1))
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.lm_head.weight.data.copy_(l2_normalize(self.lm_head.weight.data, 1))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Get normalized embeddings
        tok_emb = self.wte(idx)  # (b, t, n_embd), already normalized
        
        # x = l2_normalize(tok_emb, dim=-1) # not needed since the embeddings are already normalized
        x = tok_emb
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # x is already normalized from the last block
        if targets is not None:
            # Compute logits
            logits = self.lm_head(x) * self.s_z
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with proper parameter groups for nGPT.
        
        nGPT uses different learning rate scaling for different parameter types:
        - Regular parameters: base learning rate
        - Scaling parameters (alpha, s): potentially different treatment
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Separate scaling parameters from weight parameters
        # Weight matrices get weight decay, scaling parameters don't
        decay_params = []
        nodecay_params = []
        
        for name, param in param_dict.items():
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fabric, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        if isinstance(fabric.strategy, FSDPStrategy):
            N = N * fabric.world_size
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 82.5e12  # 4070 Ti Super GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    def model_flops_per_fwdbwd(self, fabric):
        N = self.get_num_params()
        if isinstance(fabric.strategy, FSDPStrategy):
            N = N * fabric.world_size
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        return flops_per_fwdbwd

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.
        
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # Crop if sequence is too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for last position and scale by temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
