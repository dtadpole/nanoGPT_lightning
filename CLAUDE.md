# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT Lightning is a PyTorch Lightning-based training framework for GPT-style language models. It supports three architectures: GPT-2 (standard transformer), Mamba (state-space model), and nGPT (normalized transformer based on NVIDIA's hypersphere paper).

## Commands

### Training

```bash
# Basic training with default config
python main.py

# Train with specific config file
python main.py --config-file ./config/train_gpt2.py

# Train different architectures
python main.py --arch gpt2   # Standard GPT-2
python main.py --arch mamba  # Mamba state-space model
python main.py --arch ngpt   # Normalized GPT

# Multi-GPU distributed training
torchrun --standalone --nproc_per_node=8 main.py --strategy ddp

# Profile model FLOPs and memory
python main.py --profile

# Key training arguments
python main.py \
  --batch-size 12 \
  --block-size 1024 \
  --lr 3e-4 \
  --gradient-accumulation-steps 40 \
  --max-iters 100000 \
  --amp bf16-mixed \
  --compile  # Enable torch.compile
```

### Environment

```bash
source .venv/bin/activate
```

## Architecture

### Core Model Files

- **`main.py`** - Training entry point with PyTorch Lightning Fabric. Handles optimizer setup (AdamW with separate weight decay groups), data loading (memory-mapped numpy), learning rate scheduling (linear warmup + cosine decay), and distributed training.

- **`gpt2.py`** - Standard transformer with multi-head causal self-attention (Flash Attention), SiLU-activated MLPs, and LayerNorm. Includes experimental `ConvMLP` and `ShuffleMLP` variants.

- **`ngpt.py`** - Normalized transformer where all hidden states live on a unit hypersphere (L2 normalization instead of LayerNorm). Uses learnable scaling factors for residual updates, QK normalization with learnable temperature, and rotary position embeddings (RoPE).

- **`mamba.py`** - Mamba2 state-space model with optional Mixture of Experts integration.

### Mixture of Experts

- **`moe.py`** - Standard MoE and FlashMoE (head-based routing)
- **`moe_layer_ec.py`** - Expert capacity-based MoE with distributed support
- **`cvmm.py`** - Triton-optimized sparse MoE kernels

### Configuration

- **`config/train_gpt2.py`** - Overrides default args. Executed via `exec()` with `args` namespace available. Sets batch size, gradient accumulation, W&B logging, etc.

### Data

Pre-tokenized OpenWebText in `data/openwebtext/`:
- `train.bin` (~17GB, ~9B tokens as uint16)
- `val.bin` (~8.5MB validation set)

## Model Defaults

- 124M parameters (GPT-2 scale)
- vocab_size: 50,304 (padded to multiple of 64)
- block_size: 1,024 tokens
- 12 layers, 12 heads, 768 embedding dim
- 4x MLP expansion

## Key Implementation Details

- Weight tying between token embeddings and output layer
- Gradient accumulation with `fabric.no_backward_sync()` optimization
- Model FLOPs Utilization (MFU) tracking
- nGPT requires weight normalization after each optimizer step (`model.normalize_weights()`)
- Config files modify the global `args` namespace directly via `exec()`

## Dependencies

Key libraries (from CUDA 12 environment):
- `torch>=2.9`, `lightning>=2.6` - Training framework
- `mamba_ssm>=2.2` - Mamba state-space models
- `flash_attn>=2.8` - Flash Attention for efficient attention
- `triton>=3.5` - GPU kernels for sparse MoE (cvmm.py)
- `deepspeed>=0.18` - FLOPs profiler
- `wandb` - Experiment tracking

## W&B Logging

Training logs to Weights & Biases when `args.wandb_log = True` (set in config file). Logs train/val loss, learning rate, and MFU at each eval interval.
