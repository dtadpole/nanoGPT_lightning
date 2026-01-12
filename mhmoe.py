import torch

import triton
import triton.language as tl
import matplotlib
import pandas as pd
import numpy as np
from typing import Tuple
# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def d_leacky_relu(x):
    return tl.where(x >= 0, 1.0, 100.0)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def d_silu(x, o):
    sig = tl.sigmoid(x)
    return sig + o * (1 - sig)

@torch.jit.script
def torch_silu_derivative(x):
    sig = torch.sigmoid(x)
    return sig + x * sig * (1 - sig)

@triton.jit
def d_sigmoid(o):
    # sig = tl.sigmoid(x)
    return o * (1 - o)

@torch.jit.script
def torch_sigmoid_derivative(x):
  sigmoid = torch.sigmoid(x)
  return sigmoid * (1 - sigmoid)


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 128}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 128}, num_stages=2, num_warps=4),
    ],
    key=['B', 'D', 'E'],
)
@triton.jit
def mlp_wide_kernel_fwd(
    # Pointers to matrices
    x_ptr, w1_ptr, w2_ptr, o_ptr,
    # Matrix dimensions
    H, B, D: tl.constexpr, E,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the mlp
    Z = X @ W1, H = f(Z), O = H @ W2.
    - X has shape (B, D)
    - W1 has shape (D, E)
    - W2 has shape (E, D)
    - O has shape (B, D)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    batch_groups = tl.cdiv(B, BLOCK_SIZE_B)
    pid_b = pid % batch_groups
    pid_h = pid // batch_groups
    # tl.device_print('pid_H', pid_h)
    # tl.device_print('pid_B', pid_b)
    # pid_d = tl.program_id(axis=1)
    TARGET_TYPE = x_ptr.type.element_ty
    x_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(B * H, D),
        strides=(stride_xb, stride_xd),
        offsets=(pid_h * B + pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )
    w1_ptrs = tl.make_block_ptr(
        base=w1_ptr,
        shape=(D * H, E),
        strides=(stride_w1d, stride_w1e),
        offsets=(pid_h * D, 0),
        block_shape=(D, BLOCK_SIZE_E),
        order=(1, 0),
    )
    w2_ptrs = tl.make_block_ptr(
        base=w2_ptr,
        shape=(E * H, D),
        strides=(stride_w2e, stride_w2d),
        offsets=(pid_h * E, 0),
        block_shape=(BLOCK_SIZE_E, D),
        order=(1, 0),
    )
    o_ptrs = tl.make_block_ptr(
        base=o_ptr,
        shape=(B * H, D),
        strides=(stride_ob, stride_od),
        offsets=(pid_h * B + pid_b * BLOCK_SIZE_B, 0),
        block_shape=(BLOCK_SIZE_B, D),
        order=(1, 0),
    )
    x = tl.load(x_ptrs) # BLOCK_SIZE_B, D
    o = tl.zeros((BLOCK_SIZE_B, D), dtype=tl.float32)
    for e in range(0, tl.cdiv(E, BLOCK_SIZE_E)):
        # z = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_E), dtype=tl.float32)
        # loop over D
        w1 = tl.load(w1_ptrs)       # D, BLOCK_SIZE_E
        w2 = tl.load(w2_ptrs)       # BLOCK_SIZE_E, D
        z = tl.dot(x, w1, out_dtype=tl.float32)        # BLOCK_SIZE_B, BLOCK_SIZE_E
        # activation
        if ACTIVATION == "leaky_relu":
            z = leaky_relu(z).to(TARGET_TYPE)
        elif ACTIVATION == "silu":
            z = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            z = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            z = z.to(TARGET_TYPE)
        # accumulate with o
        o = tl.dot(z, w2, o, out_dtype=tl.float32)        # BLOCK_SIZE_B, D
        # advance w1 and w2
        w1_ptrs = tl.advance(w1_ptrs, (0, BLOCK_SIZE_E))
        w2_ptrs = tl.advance(w2_ptrs, (BLOCK_SIZE_E, 0))

    o = o.to(TARGET_TYPE)
    # tl.static_print('o_ptrs', o_ptrs, o)
    tl.store(o_ptrs, o)

def mlp_wide_triton_fwd(x, w1, w2, activation=""):
    # Check constraints.
    assert x.shape[0] == w1.shape[0], "Incompatible dimensions"
    assert x.shape[0] == w2.shape[0], "Incompatible dimensions"
    assert x.shape[2] == w1.shape[1], "Incompatible dimensions"
    assert w1.shape[2] == w2.shape[1], "Incompatible dimensions"
    assert x.shape[2] == w2.shape[2], "Incompatible dimensions"

    H, B, D = x.shape
    E = w1.shape[-1]

    # print(H, B, D, E)
    # print(x.shape, w1.shape, w2.shape)

    x = x.view(H * B, D)
    w1 = w1.view(D * H, E)
    w2 = w2.view(E * H, D)

    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"

    # Allocates output.
    o = torch.zeros_like(x)
    #print(x.shape, w1.shape, w2.shape, o.shape)
    # print(o.shape)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_SIZE_B']) * H,
    )
    mlp_wide_kernel_fwd[grid](
        x, w1, w2, o,
        H, B, D, E,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        o.stride(0), o.stride(1),
        ACTIVATION=activation
    )

    # print(o.shape)
    return o.reshape(H, B, D)

def mlp_torch_fwd(x, w1, w2, activation=""):
    z = torch.bmm(x, w1)
    if activation == "leaky_relu":
        z = torch.nn.functional.leaky_relu(z)
    elif activation == "silu":
        z = torch.nn.functional.silu(z)
    elif activation == "sigmoid":
        z = torch.sigmoid(z)
    o = torch.bmm(z, w2)
    return o


@triton.jit
def _mlp_wide_kernel_bwd_dx(
    dx,
    pid_h, pid_b,
    x_ptr, w1_ptr, w2_ptr, o_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    stride_dxb, stride_dxd,
    stride_dw1d, stride_dw1e,
    stride_dw2e, stride_dw2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """Kernel for computing the mlp_bwd_dx
    Z = X @ W1, H = f(Z), O = H @ W2
    - X has shape (B, D)
    - W1 has shape (D, E)
    - W2 has shape (E, D)
    - O has shape (B, D)
    - dX has shape (B, D)
    - dW1 has shape (D, E)
    - dW2 has shape (E, D)
    - dO has shape (B, D)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # pid_b = pid // H
    # pid_h = pid % H
    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    x_ptrs = x_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_xb + offs_d[None, :] * stride_xd)
    x_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

    do_ptrs = do_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_dob + offs_d[None, :] * stride_dod)
    do_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

    w1_ptrs = w1_ptr + ((pid_h * D + offs_d[:, None]) * stride_w1d + offs_e[None, :] * stride_w1e)
    w2_ptrs = w2_ptr + ((pid_h * E + offs_e[:, None]) * stride_w2e + offs_d[None, :] * stride_w2d)

    dw1_ptrs = dw1_ptr + ((pid_h * D + offs_d[:, None]) * stride_dw1d + offs_e[None, :] * stride_dw1e)
    dw2_ptrs = dw2_ptr + ((pid_h * E + offs_e[:, None]) * stride_dw2e + offs_d[None, :] * stride_dw2d)

    x = tl.load(x_ptrs, mask=x_mask, other=0.0) # BLOCK_SIZE_B, D
    do = tl.load(do_ptrs, mask=do_mask, other=0.0) # BLOCK_SIZE_B, D

    for e in range(0, tl.cdiv(E, BLOCK_SIZE_E)):

        w1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - e * BLOCK_SIZE_E)
        w2_mask = (offs_e[:, None] < E - e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

        w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0) # D, BLOCK_SIZE_E
        w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0) # BLOCK_SIZE_E, D

        z = tl.dot(x, w1, out_dtype=tl.float32)                         # BLOCK_SIZE_B, BLOCK_SIZE_E
        # activation
        if ACTIVATION == "leaky_relu":
            h = leaky_relu(z).to(TARGET_TYPE)                           # BLOCK_SIZE_B, BLOCK_SIZE_E
        elif ACTIVATION == "silu":
            h = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            h = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            h = z.to(TARGET_TYPE)                                       # BLOCK_SIZE_B, BLOCK_SIZE_E

        dh = tl.dot(do, tl.trans(w2), out_dtype=tl.float32)             # BLOCK_SIZE_B, BLOCK_SIZE_E

        if ACTIVATION == "leaky_relu":
            dz = (dh * d_leacky_relu(z)).to(TARGET_TYPE)   # BLOCK_SIZE_B, BLOCK_SIZE_E
        elif ACTIVATION == "silu":
            dz = (dh * d_silu(z, h)).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            dz = (dh * d_sigmoid(h)).to(TARGET_TYPE)
        else:
            dz = dh.to(TARGET_TYPE)

        dx += tl.dot(dz, tl.trans(w1), out_dtype=tl.float32)             # BLOCK_SIZE_B, D

        w1_ptrs += BLOCK_SIZE_E * stride_w1e
        w2_ptrs += BLOCK_SIZE_E * stride_w2e
        dw1_ptrs += BLOCK_SIZE_E * stride_dw1e
        dw2_ptrs += BLOCK_SIZE_E * stride_dw2e

    return dx

@triton.jit
def _mlp_wide_kernel_bwd_dw1w2(
    dw1, dw2,
    pid_h, pid_e,
    x_ptr, w1_ptr, w2_ptr, o_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    stride_dxb, stride_dxd,
    stride_dw1d, stride_dw1e,
    stride_dw2e, stride_dw2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """Kernel for computing the mlp_bwd_dw1w2
    Z = X @ W1, H = f(Z), O = H @ W2
    - X has shape (B, D)
    - W1 has shape (D, E)
    - W2 has shape (E, D)
    - O has shape (B, D)
    - dX has shape (B, D)
    - dW1 has shape (D, E)
    - dW2 has shape (E, D)
    - dO has shape (B, D)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # pid_b = pid // H
    # pid_h = pid % H
    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    x_ptrs = x_ptr + ((pid_h * B + offs_b[:, None]) * stride_xb + offs_d[None, :] * stride_xd)

    do_ptrs = do_ptr + ((pid_h * B + offs_b[:, None]) * stride_dob + offs_d[None, :] * stride_dod)
    do_mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)

    w1_ptrs = w1_ptr + ((pid_h * D + offs_d[:, None]) * stride_w1d + (pid_e * BLOCK_SIZE_E + offs_e[None, :]) * stride_w1e)
    w1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - pid_e * BLOCK_SIZE_E)
    w2_ptrs = w2_ptr + ((pid_h * E + pid_e * BLOCK_SIZE_E + offs_e[:, None]) * stride_w2e + offs_d[None, :] * stride_w2d)
    w2_mask = (offs_e[:, None] < E - pid_e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

    w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)                      # D, BLOCK_SIZE_E
    w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)                      # BLOCK_SIZE_E, D
    do = tl.load(do_ptrs, mask=do_mask, other=0.0)                      # BLOCK_SIZE_B, D

    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):

        x_mask = (offs_b[:, None] < B - b * BLOCK_SIZE_B) & (offs_d[None, :] < D)
        do_mask = (offs_b[:, None] < B - b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)                     # BLOCK_SIZE_B, D
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)                  # BLOCK_SIZE_B, D

        z = tl.dot(x, w1, out_dtype=tl.float32)                         # BLOCK_SIZE_B, BLOCK_SIZE_E
        # activation
        if ACTIVATION == "leaky_relu":
            h = leaky_relu(z).to(TARGET_TYPE)                           # BLOCK_SIZE_B, BLOCK_SIZE_E
        elif ACTIVATION == "silu":
            h = silu(z).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            h = tl.sigmoid(z).to(TARGET_TYPE)
        else:
            h = z.to(TARGET_TYPE)                                       # BLOCK_SIZE_B, BLOCK_SIZE_E

        dh = tl.dot(do, tl.trans(w2), out_dtype=tl.float32)             # BLOCK_SIZE_B, BLOCK_SIZE_E

        dw2 += tl.dot(tl.trans(h), do, out_dtype=tl.float32)            # BLOCK_SIZE_E, D
        # tl.store(dw2_ptrs, dw2, mask=dw2_mask, eviction_policy="evict_last")

        if ACTIVATION == "leaky_relu":
            dz = (dh * d_leacky_relu(z)).to(TARGET_TYPE)   # BLOCK_SIZE_B, BLOCK_SIZE_E
        elif ACTIVATION == "silu":
            dz = (dh * d_silu(z, h)).to(TARGET_TYPE)
        elif ACTIVATION == "sigmoid":
            dz = (dh * d_sigmoid(h)).to(TARGET_TYPE)
        else:
            dz = dh.to(TARGET_TYPE)

        dw1 += tl.dot(tl.trans(x), dz, out_dtype=tl.float32)             # D, BLOCK_SIZE_E

        x_ptrs += BLOCK_SIZE_B * stride_xb
        do_ptrs += BLOCK_SIZE_B * stride_dob

    return dw1, dw2


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 16}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 64}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 64, 'GROUP_SIZE_B': 1}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_E': 128, 'GROUP_SIZE_B': 1}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_E': 128}, num_stages=2, num_warps=4),
    ],
    key=['H', 'B', 'D', 'E'],
)
@triton.jit
def mlp_wide_kernel_bwd2(
    x_ptr, w1_ptr, w2_ptr, o_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
    H, B, D: tl.constexpr, E,
    stride_xb, stride_xd,
    stride_w1d, stride_w1e,
    stride_w2e, stride_w2d,
    stride_ob, stride_od,
    stride_dxb, stride_dxd,
    stride_dw1d, stride_dw1e,
    stride_dw2e, stride_dw2d,
    stride_dob, stride_dod,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_E: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """Kernel for computing the mlp
    Z = X @ W1, H = f(Z), O = H @ W2
    - X has shape (B, D)
    - W1 has shape (D, E)
    - W2 has shape (E, D)
    - O has shape (B, D)
    - dX has shape (B, D)
    - dW1 has shape (D, E)
    - dW2 has shape (E, D)
    - dO has shape (B, D)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    pid_x_w = 0

    # batch_groups = tl.cdiv(B, BLOCK_SIZE_B)
    batch_groups_e = tl.cdiv(E, BLOCK_SIZE_E)
    batch_groups_b = tl.cdiv(B, BLOCK_SIZE_B)
    idx = pid % (batch_groups_e + batch_groups_b)
    pid_h = pid // (batch_groups_e + batch_groups_b)
    # pid_b = pid // H
    # pid_h = pid % H
    TARGET_TYPE = x_ptr.type.element_ty

    offs_b = tl.arange(0, BLOCK_SIZE_B)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_SIZE_E)

    if idx >= batch_groups_e:

        pid_b = idx - batch_groups_e

        dx_ptrs = dx_ptr + ((pid_h * B + pid_b * BLOCK_SIZE_B + offs_b[:, None]) * stride_dxb + offs_d[None, :] * stride_dxd)
        dx_mask = (offs_b[:, None] < B - pid_b * BLOCK_SIZE_B) & (offs_d[None, :] < D)

        dx = tl.zeros((BLOCK_SIZE_B, D), dtype=tl.float32)  # BLOCK_SIZE_B, D
        dx = _mlp_wide_kernel_bwd_dx(
            dx,
            pid_h, pid_b,
            x_ptr, w1_ptr, w2_ptr, o_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
            H, B, D, E,
            stride_xb, stride_xd,
            stride_w1d, stride_w1e,
            stride_w2e, stride_w2d,
            stride_ob, stride_od,
            stride_dxb, stride_dxd,
            stride_dw1d, stride_dw1e,
            stride_dw2e, stride_dw2d,
            stride_dob, stride_dod,
            BLOCK_SIZE_B, BLOCK_SIZE_E,
            ACTIVATION
        )

        tl.store(dx_ptrs, dx.to(TARGET_TYPE), mask=dx_mask)

    else:

        pid_e = idx

        dw1_ptrs = dw1_ptr + ((pid_h * D + offs_d[:, None]) * stride_dw1d + (pid_e * BLOCK_SIZE_E + offs_e[None, :]) * stride_dw1e)
        dw1_mask = (offs_d[:, None] < D) & (offs_e[None, :] < E - pid_e * BLOCK_SIZE_E)
        dw2_ptrs = dw2_ptr + ((pid_h * E + pid_e * BLOCK_SIZE_E + offs_e[:, None]) * stride_dw2e + offs_d[None, :] * stride_dw2d)
        dw2_mask = (offs_e[:, None] < E - pid_e * BLOCK_SIZE_E) & (offs_d[None, :] < D)

        dw1 = tl.zeros((D, BLOCK_SIZE_E), dtype=tl.float32)                 # D, BLOCK_SIZE_E
        dw2 = tl.zeros((BLOCK_SIZE_E, D), dtype=tl.float32)                 # BLOCK_SIZE_E, D

        dw1, dw2 = _mlp_wide_kernel_bwd_dw1w2(
            dw1, dw2,
            pid_h, pid_e,
            x_ptr, w1_ptr, w2_ptr, o_ptr, dx_ptr, dw1_ptr, dw2_ptr, do_ptr,
            H, B, D, E,
            stride_xb, stride_xd,
            stride_w1d, stride_w1e,
            stride_w2e, stride_w2d,
            stride_ob, stride_od,
            stride_dxb, stride_dxd,
            stride_dw1d, stride_dw1e,
            stride_dw2e, stride_dw2d,
            stride_dob, stride_dod,
            BLOCK_SIZE_B, BLOCK_SIZE_E,
            ACTIVATION
        )

        tl.store(dw1_ptrs, dw1.to(TARGET_TYPE), mask=dw1_mask)
        tl.store(dw2_ptrs, dw2.to(TARGET_TYPE), mask=dw2_mask)

def mlp_wide_triton_bwd2(x, w1, w2, o, do, activation=""):
    # Check constraints.
    assert x.shape[-1] == w1.shape[-2], "Incompatible dimensions"
    assert w1.shape[-1] == w2.shape[-2], "Incompatible dimensions"
    assert x.shape[-1] == w2.shape[-1], "Incompatible dimensions"
    assert x.shape == o.shape, "Incompatible dimensions"
    assert x.shape == do.shape, "Incompatible dimensions"

    H, B, D = x.shape
    E = w1.shape[-1]

    x = x.view(H * B, D)
    w1 = w1.view(D * H, E)
    w2 = w2.view(E * H, D)
    o = o.view(H * B, D)
    do = do.view(H * B, D)

    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w1.is_contiguous(), "Matrix W1 must be contiguous"
    assert w2.is_contiguous(), "Matrix W2 must be contiguous"
    assert o.is_contiguous(), "Matrix O must be contiguous"
    assert do.is_contiguous(), "Matrix dO must be contiguous"


    # Allocates output.
    dx = torch.zeros_like(x)
    dw1 = torch.zeros_like(w1)
    dw2 = torch.zeros_like(w2)
    # print(dx.shape, dw1.shape, dw2.shape, do.shape)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        # triton.cdiv(B, META['BLOCK_SIZE_B']) * H,
        (triton.cdiv(B, META['BLOCK_SIZE_B']) + triton.cdiv(E, META['BLOCK_SIZE_E'])) * H,
    )
    mlp_wide_kernel_bwd2[grid](
        x, w1, w2, o, dx, dw1, dw2, do,
        H, B, D, E,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        o.stride(0), o.stride(1),
        dx.stride(0), dx.stride(1),
        dw1.stride(0), dw1.stride(1),
        dw2.stride(0), dw2.stride(1),
        do.stride(0), do.stride(1),
        ACTIVATION=activation
    )

    # print(dx.shape, dw1.shape, dw2.shape)
    return dx.view(H, B, D), dw1.view(H, D, E), dw2.view(H, E, D)


def mlp_torch_bwd2(x, w1, w2, o, do, activation="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if activation == "leaky_relu":
        y_ref = torch.nn.functional.leaky_relu(x @ w1, negative_slope=0.01) @ w2
    elif activation == "silu":
        y_ref = torch.nn.functional.silu(x @ w1) @ w2
    elif activation == "sigmoid":
        y_ref = torch.nn.functional.sigmoid(x @ w1) @ w2
    else:
        y_ref = x @ w1 @ w2
    y_ref.backward(do, retain_graph=True)
    # print(x.grad.shape, w1.grad.shape, w2.grad.shape)
    return x.grad, w1.grad, w2.grad


def unit_test_fwd():
    # torch.manual_seed(115)
    dtype = torch.bfloat16
    B = 1024
    D = 16
    E = 768
    H = 12
    x = torch.randn((H, B, D), device='cuda', dtype=dtype) / np.sqrt(D)
    w1 = torch.randn((H, D, E), device='cuda', dtype=dtype) / np.sqrt(E)
    w2 = torch.randn((H, E, D), device='cuda', dtype=dtype) / np.sqrt(D)

    activation = "silu"

    triton_output = mlp_wide_triton_fwd(x, w1, w2, activation=activation)
    torch_output = mlp_torch_fwd(x, w1, w2, activation=activation)

    print(f"triton_output={triton_output.shape, triton_output}")
    print(f"torch_output={torch_output.shape, torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    diff = np.abs(triton_output.to(torch.float32).cpu().numpy() - torch_output.to(torch.float32).cpu().numpy())
    print("max diff:",np.max(diff))
    print("mean diff:",np.mean(diff))


def unit_test_bwd2():
    torch.manual_seed(12932)
    DTYPE = torch.bfloat16
    B = 1024 * 4
    D = 64
    E = 768
    H = E // D

    x = torch.randn((H, B, D), device='cuda', dtype=DTYPE, requires_grad=True)
    w1 = torch.randn((H, D, E), device='cuda', dtype=DTYPE, requires_grad=True)
    w2 = torch.randn((H, E, D), device='cuda', dtype=DTYPE, requires_grad=True)
    o = torch.randn((H, B, D), device='cuda', dtype=DTYPE)
    do = 0.05 * torch.randn((H, B, D), device='cuda', dtype=DTYPE) / np.sqrt(D)

    activation = "silu"

    triton_output = mlp_wide_triton_bwd2(x, w1, w2, o, do, activation=activation)
    # triton_output = mlp_torch_bwd_raw(x, w1, w2, o, do, activation=activation)
    torch_output = mlp_torch_bwd2(x, w1, w2, o, do, activation=activation)

    print(f"triton_output={triton_output[1].shape, triton_output[1]}")
    print(f"torch_output={torch_output[1].shape, torch_output[1]}")

    eplison = 3e-2
    if torch.allclose(triton_output[1], torch_output[1], atol=eplison, rtol=eplison):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    dx_diff = np.abs(triton_output[0].to(torch.float32).detach().cpu().numpy() - torch_output[0].to(torch.float32).detach().cpu().numpy())
    dx_rel_diff = dx_diff / np.abs(triton_output[0].to(torch.float32).detach().cpu().numpy() + torch_output[0].to(torch.float32).detach().cpu().numpy() + eplison)
    print(f"[dx: {triton_output[0].shape}, {torch_output[0].shape}] max diff: {np.max(dx_diff):.2e}, mean diff: {np.mean(dx_diff):.2e}, rel max diff: {np.max(dx_rel_diff)*100:.2f}%, rel mean diff: {np.mean(dx_rel_diff)*100:.2f}%")

    dw1_diff = np.abs(triton_output[1].to(torch.float32).detach().cpu().numpy() - torch_output[1].to(torch.float32).detach().cpu().numpy())
    dw1_rel_diff = dw1_diff / np.abs(triton_output[1].to(torch.float32).detach().cpu().numpy() + torch_output[1].to(torch.float32).detach().cpu().numpy() + eplison)
    print(f"[dw1: {triton_output[1].shape}, {torch_output[1].shape}] max diff: {np.max(dw1_diff):.2e}, mean diff: {np.mean(dw1_diff):.2e}, rel max diff: {np.max(dw1_rel_diff)*100:.2f}%, rel mean diff: {np.mean(dw1_rel_diff)*100:.2f}%")

    dw2_diff = np.abs(triton_output[2].to(torch.float32).detach().cpu().numpy() - torch_output[2].to(torch.float32).detach().cpu().numpy())
    dw2_rel_diff = dw2_diff / np.abs(triton_output[2].to(torch.float32).detach().cpu().numpy() + torch_output[2].to(torch.float32).detach().cpu().numpy() + eplison)
    print(f"[dw2: {triton_output[2].shape}, {torch_output[2].shape}] max diff: {np.max(dw2_diff):.2e}, mean diff: {np.mean(dw2_diff):.2e}, rel max diff: {np.max(dw2_rel_diff)*100:.2f}%, rel mean diff: {np.mean(dw2_rel_diff)*100:.2f}%")


if __name__ == '__main__':

    unit_test_fwd()
    unit_test_bwd2()

    # DTYPE = torch.bfloat16
    # HEAD = 12
    # B = 1024 * HEAD
    # E =768
    # D = E // HEAD
    # x = torch.randn((HEAD, B, D), device='cuda', dtype=DTYPE) / np.sqrt(D)
    # w1 = torch.randn((HEAD, D, E), device='cuda', dtype=DTYPE) / np.sqrt(E)
    # w2 = torch.randn((HEAD, E, D), device='cuda', dtype=DTYPE) / np.sqrt(D)

    # mlp_torch(x, w1, w2, activation="leaky_relu")

    # o = mlp_wide_triton_fwd(x, w1, w2, activation="leaky_relu")
    # print(o.shape)
