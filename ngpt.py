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
    # Extract sin and cos from interleaved format (sin at even indices, cos at odd)
    sin = sinusoidal_pos[..., 0::2]  # (T, head_dim//2)
    cos = sinusoidal_pos[..., 1::2]  # (T, head_dim//2)

    # Standard RoPE: rotate pairs of dimensions
    # For each pair (x0, x1), apply rotation:
    #   x0' = x0 * cos - x1 * sin
    #   x1' = x0 * sin + x1 * cos
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    # Apply rotation
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    # Interleave back to original shape
    q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
    k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)

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
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # V and output projections with learnable scale
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Learnable attention scaling (s_qk) - per dimension as per paper/NVIDIA
        # Applied to both Q and K, creating a weighted dot product
        # Shape: (n_embd,) which can be viewed as (n_head, head_dim)
        # NVIDIA-style parameterization: store at 1/sqrt(n_embd), effective value = 1.0
        if self.use_sqk_scaling:
            self.sqk_init_value = 1.0
            self.sqk_init_scaling = 1.0 / math.sqrt(config.n_embd)
            self.s_qk = nn.Parameter(torch.ones(config.n_embd) * self.sqk_init_scaling)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self._init_weights()


    def _init_weights(self):
        # NVIDIA uses std = 1/sqrt(n_embd) for weight initialization
        std = 1.0 / math.sqrt(self.n_embd)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=std)
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
        
        # Reshape to (B, n_head, T, head_dim) for attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.head_dim).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q, k)
        # q, k remain (B, n_head, T, head_dim) after RoPE

        # Normalize Q and K (key innovation of nGPT)
        q = l2_normalize(q, dim=-1)
        k = l2_normalize(k, dim=-1)

        # s_qk scales per-dimension, applied to both Q and K (per paper/NVIDIA)
        # This creates a weighted dot product: att[i,j] = Σ_d (s_qk[d]² * q[i,d] * k[j,d])
        if self.use_sqk_scaling:
            # NVIDIA-style: multiply stored param by (init_value / init_scaling) to get effective scale
            sqk_effective = self.s_qk * (self.sqk_init_value / self.sqk_init_scaling)
            # Reshape from (n_embd,) to (1, n_head, 1, head_dim) for broadcasting
            sqk = sqk_effective.view(1, self.n_head, 1, self.head_dim)
            q = q * sqk
            k = k * sqk

        # Base scaling: sqrt(head_dim) compensates for normalized dot product having small variance
        # s_qk (init 1.0) provides additional learnable scaling on top
        softmax_scale = math.sqrt(self.head_dim)

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

        # Output projection and normalize (nGPT requires normalized outputs)
        y = self.c_proj(y)
        y = l2_normalize(y, dim=-1)

        return y


class NormalizedMLP(nn.Module):
    """
    Normalized Gated MLP (SwiGLU) for nGPT.

    Uses SiLU-gated linear units with normalized weight matrices.
    Combined projection (c_fc) for gate and up, matching NVIDIA implementation.
    Formula: uv = c_fc(x) * s_uv
             hidden = u * silu(v)
             output = w_down(hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_expand * config.n_embd
        self.n_embd = config.n_embd
        self.use_suv_scaling = config.use_suv_scaling

        # Combined projection for u and v - outputs 2 * hidden_dim
        self.c_fc = nn.Linear(config.n_embd, 2 * self.hidden_dim, bias=False)
        self.w_down = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

        # Learnable scaling factors for combined u+v (s_uv per NVIDIA)
        # Shape: (2 * hidden_dim) - first half for u, second half for v
        # NVIDIA-style parameterization: store at 1/sqrt(n_embd), effective value = 1.0
        # Then multiplied by sqrt(n_embd) in forward for base scaling
        if self.use_suv_scaling:
            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0 / math.sqrt(config.n_embd)
            self.s_uv = nn.Parameter(torch.ones(2 * self.hidden_dim) * self.suv_init_scaling)
        self.base_scale = math.sqrt(config.n_embd)

        # SiLU activation
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        # NVIDIA uses std = 1/sqrt(n_embd) for weight initialization
        std = 1.0 / math.sqrt(self.n_embd)
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=std)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=std)
        # Normalize c_fc along input dimension (each row is a unit vector)
        self.c_fc.weight.data.copy_(l2_normalize(self.c_fc.weight.data, 1))
        self.w_down.weight.data.copy_(l2_normalize(self.w_down.weight.data, 0))

    def forward(self, x):
        # Combined projection with scaling
        if self.use_suv_scaling:
            # NVIDIA-style: compute effective s_uv, then multiply by base_scale
            suv_effective = self.s_uv * (self.suv_init_value / self.suv_init_scaling)
            uv = self.c_fc(x) * (suv_effective * self.base_scale)
        else:
            uv = self.c_fc(x) * self.base_scale
        # Split into u and v (NVIDIA convention: u * silu(v))
        u, v = uv.chunk(2, dim=-1)
        # Gated MLP: hidden = u * silu(v) (matching NVIDIA)
        hidden = u * self.act(v)
        # Down projection and normalize (nGPT requires normalized outputs)
        x = self.w_down(hidden)
        x = l2_normalize(x, dim=-1)
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
        # NVIDIA-style parameterization: store at 1/sqrt(n_embd) scale for better Adam optimization
        # Effective value at init = 1/sqrt(2*n_layer) ≈ 0.204 for 12 layers
        self.alpha_init_value = 1.0 / math.sqrt(2 * config.n_layer)
        self.alpha_init_scaling = 1.0 / math.sqrt(config.n_embd)
        self.alpha_attn = nn.Parameter(torch.ones(config.n_embd) * self.alpha_init_scaling)
        self.alpha_mlp = nn.Parameter(torch.ones(config.n_embd) * self.alpha_init_scaling)

    def forward(self, x):
        # Input x should already be normalized (on unit hypersphere)

        # Compute effective alpha: stored_param * (init_value / init_scaling)
        alpha_attn = self.alpha_attn * (self.alpha_init_value / self.alpha_init_scaling)
        alpha_mlp = self.alpha_mlp * (self.alpha_init_value / self.alpha_init_scaling)

        # Attention update: h = normalize(h + α_a ⊙ Attn(h_a - h))
        x = l2_normalize(x + alpha_attn * (self.attn(x) - x), dim=-1)

        # MLP update: h = normalize(h + α_m ⊙ MLP(h_m - h))
        x = l2_normalize(x + alpha_mlp * (self.mlp(x) - x), dim=-1)

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
    use_sqk_scaling: bool = False
    use_suv_scaling: bool = False
    weight_tying: bool = True  # Share weights between wte and lm_head


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

        self.normalize_weights_print_count = 5

        # Normalized embeddings
        # Learnable scaling for combining token and position embeddings
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([NormalizedBlock(config, i) for i in range(config.n_layer)])

        # Output head with normalized weights
        # s_z is a learnable per-token scaling factor
        # NVIDIA-style parameterization: store at 1/sqrt(n_embd) scale for better Adam optimization
        # Effective value at init = (1/sqrt(n_embd)) * sqrt(n_embd) = 1.0
        self.s_z_init_scaling = 1.0 / math.sqrt(config.n_embd)
        self.s_z = nn.Parameter(torch.ones(config.vocab_size) * self.s_z_init_scaling)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output head
        if config.weight_tying:
            self.wte.weight = self.lm_head.weight

        # Initialize weights
        self._init_weights()
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    @torch.no_grad()
    def normalize_weights(self):
        if self.normalize_weights_print_count > 0:
            self.normalize_weights_print_count -= 1
            print(f"Normalizing weights... ({5 - self.normalize_weights_print_count}/5)")
        # Normalize embeddings and output head
        # When weight_tying is enabled, wte and lm_head share the same weight tensor (lm_head is master)
        self.lm_head.weight.data.copy_(l2_normalize(self.lm_head.weight.data, 1))   # V, n_embd
        if not self.config.weight_tying:
            self.wte.weight.data.copy_(l2_normalize(self.wte.weight.data, 1))         # V, n_embd
        
        for layer_idx in range(0, self.config.n_layer):
            block = self.blocks[layer_idx]
            # print(f"Normalizing weights for block {layer_idx}...")
            # Attention weights
            block.attn.q_proj.weight.data.copy_(l2_normalize(block.attn.q_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.k_proj.weight.data.copy_(l2_normalize(block.attn.k_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.v_proj.weight.data.copy_(l2_normalize(block.attn.v_proj.weight.data, 1))             # n_embd, n_embd
            block.attn.c_proj.weight.data.copy_(l2_normalize(block.attn.c_proj.weight.data, 1))             # n_embd, n_embd
            
            # MLP weights
            block.mlp.c_fc.weight.data.copy_(l2_normalize(block.mlp.c_fc.weight.data, 1))                   # 2*hidden, n_embd
            block.mlp.w_down.weight.data.copy_(l2_normalize(block.mlp.w_down.weight.data, 0))               # n_embd, hidden

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.weight_tying:
            n_params -= self.wte.weight.numel()
        return n_params

    def _init_weights(self):
        # NVIDIA uses std = 1/sqrt(n_embd) for weight initialization
        std = 1.0 / math.sqrt(self.config.n_embd)
        # When weight_tying is enabled, lm_head is the master weight (wte shares it)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        self.lm_head.weight.data.copy_(l2_normalize(self.lm_head.weight.data, 1))
        # Only initialize wte separately if not using weight tying
        if not self.config.weight_tying:
            nn.init.normal_(self.wte.weight, mean=0.0, std=std)
            self.wte.weight.data.copy_(l2_normalize(self.wte.weight.data, 1))

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
        # NVIDIA-style: multiply stored param by sqrt(n_embd) to get effective scale
        # At init: (1/sqrt(n_embd)) * sqrt(n_embd) = 1.0
        logit_scale = self.s_z * math.sqrt(self.config.n_embd)
        if targets is not None:
            # Compute logits
            logits = self.lm_head(x) * logit_scale
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute for last position
            logits = self.lm_head(x[:, [-1], :]) * logit_scale
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
