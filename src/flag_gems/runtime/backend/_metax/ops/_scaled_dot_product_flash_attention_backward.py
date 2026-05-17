import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt
exp2 = tl_extra_shim.exp2

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, Q_CTX, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    off_m = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < Q_CTX

    off_hz = tle.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(
        O + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * D_HEAD * Q_CTX + off_m[:, None] * D_HEAD + off_n[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * Q_CTX + off_m, delta, mask=mask)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    key,
    value,
    sm_scale,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    Q_CTX,
    KV_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        offs_m_mask = offs_m < Q_CTX

        qT_ptrs = (
            Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
        )
        do_ptrs = (
            DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        )

        qT = tl.load(
            qT_ptrs, mask=offs_m_mask[None, :], other=0.0
        )

        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))

        qkT = tl.dot(key, qT)
        m = tl.broadcast_to(m[None, :], (BLOCK_N1, BLOCK_M1))
        m = tl.where(offs_n_mask[:, None], m, float("inf"))
        pT = exp2(qkT - m)

        mask = (offs_m < Q_CTX)[None, :] & (offs_n < KV_CTX)[
            :, None
        ]
        if MASK:
            mask &= offs_m[None, :] >= offs_n[:, None]
        pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)

        dv += tl.dot(pT, do.to(tl.float32))
        Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)

        dpT = tl.dot(value, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        qT = tl.where(offs_m_mask[None, :], qT, 0.0)
        dsT = tl.where(
            offs_m_mask[None, :] & offs_n_mask[:, None], dsT, 0.0
        )
        dk += tl.dot(
            dsT, tl.trans(qT)
        )
        curr_m += step_m
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq,
    query,
    K,
    V,
    do,
    m,
    D,
    stride_tok,
    stride_d,
    H,
    Q_CTX,
    KV_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        offs_n_mask = offs_n < KV_CTX

        kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
        vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

        kT = tl.load(kT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        qk = tl.dot(query, kT)
        p = exp2(qk - m)
        mask = (offs_m < Q_CTX)[:, None] & (offs_n < KV_CTX)[None, :]
        if MASK:
            mask &= offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = tl.where(mask, ds, 0.0).to(kT.dtype)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
    return dq


@libentry()
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    kv_stride_z,
    kv_stride_h,
    H,
    Q_CTX,
    KV_CTX,
    kv_head_num,
    GROUP_HEAD: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    tl.device_assert(Q_CTX % BLOCK_M1 == 0, "Q_CTX must be a multiple of BLOCK_M1.")

    LN2: tl.constexpr = 0.6931471824645996

    bhid = tle.program_id(2)
    off_chz = (bhid * Q_CTX).to(tl.int64)
    batch_id = bhid // H
    q_head_id = bhid % H
    kv_head_id = q_head_id // GROUP_HEAD
    adj = (stride_h * q_head_id + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * kv_head_id + kv_stride_z * batch_id).to(tl.int64)

    pid = tle.program_id(0)

    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, BLOCK_DMODEL)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < KV_CTX

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    key = tl.load(
        K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_n_mask[:, None],
        other=0.0,
    )
    value = tl.load(
        V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_n_mask[:, None],
        other=0.0,
    )

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        key,
        value,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        Q_CTX,
        KV_CTX,
        MASK_BLOCK_M1,
        BLOCK_N1,
        BLOCK_DMODEL,
        start_n,
        start_m,
        num_steps,
        MASK=True,
    )

    start_m += num_steps * MASK_BLOCK_M1
    remaining_m = Q_CTX - start_m
    num_steps = (remaining_m + BLOCK_M1 - 1) // BLOCK_M1

    if num_steps > 0 and start_m < Q_CTX:
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,
            Q,
            key,
            value,
            sm_scale,
            DO,
            M,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M1,
            BLOCK_N1,
            BLOCK_DMODEL,
            start_n,
            start_m,
            num_steps,
            MASK=False,
        )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv, mask=offs_n_mask[:, None])

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk, mask=offs_n_mask[:, None])

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    start_m = pid * BLOCK_M2
    end_n = min(start_m + BLOCK_M2, KV_CTX)
    num_steps = (end_n - start_n + MASK_BLOCK_N2 - 1) // MASK_BLOCK_N2

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < Q_CTX

    query = tl.load(
        Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_m_mask[:, None],
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d,
        mask=offs_m_mask[:, None],
        other=0.0,
    )

    m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))
    m = m[:, None]

    if num_steps > 0:
        dq = _attn_bwd_dq(
            dq,
            query,
            K,
            V,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M2,
            MASK_BLOCK_N2,
            BLOCK_DMODEL,
            start_m,
            start_n,
            num_steps,
            MASK=True,
        )

    stage2_end_n = start_n
    stage2_num_steps = (stage2_end_n + BLOCK_N2 - 1) // BLOCK_N2

    if stage2_num_steps > 0:
        dq = _attn_bwd_dq(
            dq,
            query,
            K,
            V,
            do,
            m,
            D,
            stride_tok,
            stride_d,
            H,
            Q_CTX,
            KV_CTX,
            BLOCK_M2,
            BLOCK_N2,
            BLOCK_DMODEL,
            start_m,
            stage2_end_n - stage2_num_steps * BLOCK_N2,
            stage2_num_steps,
            MASK=False,
        )
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2

    tl.store(dq_ptrs, dq, mask=offs_m_mask[:, None])


def scaled_dot_product_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q=None,
    cum_seq_k=None,
    max_q=None,
    max_k=None,
    dropout_p=0.0,
    is_causal=False,
    philox_seed=None,
    philox_offset=None,
    attn_mask=None,
    scale=None,
    enable_gqa=False,
):
    """Metax specialized implementation of flash attention backward.

    Handles both regular SDPA backward and flash attention backward (aten operator).
    Currently varlen sequences and dropout are not supported.
    """
    logger.debug("METAX GEMS SCALED DOT PRODUCT ATTENTION BACKWARD")

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    assert grad_out.is_contiguous()
    assert (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and out.is_contiguous()
    )
    assert query.stride() == out.stride() == grad_out.stride()
    assert key.stride() == value.stride()

    BLOCK_DMODEL = HEAD_DIM_K
    BATCH, Q_HEAD, Q_CTX = query.shape[:3]
    _, KV_HEAD, KV_CTX = key.shape[:3]
    group_head = Q_HEAD // KV_HEAD

    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.0 / math.log(2)

    arg_k = key * (sm_scale * RCP_LN2)
    PRE_BLOCK = 256

    pre_grid = (triton.cdiv(Q_CTX, PRE_BLOCK), BATCH * Q_HEAD)

    delta = torch.empty_like(logsumexp)

    dq = torch.empty_like(query).contiguous()
    dk = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_K),
        device=key.device,
        dtype=key.dtype,
        memory_format=torch.contiguous_format,
    )
    dv = torch.empty(
        (BATCH, Q_HEAD, KV_CTX, HEAD_DIM_V),
        device=value.device,
        dtype=value.dtype,
        memory_format=torch.contiguous_format,
    )

    with torch_device_fn.device(query.device):
        _attn_bwd_preprocess[pre_grid](
            out,
            grad_out,
            delta,
            BATCH,
            Q_HEAD,
            Q_CTX,
            BLOCK_M=PRE_BLOCK,
            D_HEAD=BLOCK_DMODEL,
        )

    max_block_n1 = 128
    grid = (triton.cdiv(Q_CTX, max_block_n1), 1, BATCH * Q_HEAD)

    with torch_device_fn.device(query.device):
        _attn_bwd[grid](
            query,
            arg_k,
            value,
            sm_scale,
            grad_out,
            dq,
            dk,
            dv,
            logsumexp,
            delta,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            Q_HEAD,
            Q_CTX,
            KV_CTX,
            KV_HEAD,
            GROUP_HEAD=group_head,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )

    if group_head > 1:
        dk = dk.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_K)
        dv = dv.reshape(BATCH, Q_HEAD // group_head, group_head, KV_CTX, HEAD_DIM_V)
        dk = dk.sum(dim=2)
        dv = dv.sum(dim=2)

    return dq, dk, dv