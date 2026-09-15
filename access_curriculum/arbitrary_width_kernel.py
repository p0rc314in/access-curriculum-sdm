# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from the Sparse Delta Memory backward kernel; see THIRD_PARTY.md.
"""Masked-lane form of the released fused backward elementwise kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_fused_bwd_elementwise_kernel(
    base_ret_ptr,
    base_dv_dw_ptr,
    kv_ptr,
    cumul_k_ptr,
    post_decay_ptr,
    d_kv_A_row_ptr,
    d_kv_A_col_ptr,
    d_kv_from_QK_ptr,
    base_go_msq_ptr,
    qv_ptr,
    cumul_q_ptr,
    d_qv_from_QK_ptr,
    grad_kv_ptr,
    d_lck_total_ptr,
    d_rev_lck_ptr,
    d_g_from_pd_ptr,
    grad_qv_ptr,
    d_lcq_total_ptr,
    NCxC: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= NCxC:
        return

    write_offsets = tl.arange(0, BLOCK_W)
    read_offsets = tl.arange(0, BLOCK_R)
    valid_write = write_offsets < W
    valid_read = read_offsets < R

    base_ret = tl.load(
        base_ret_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    base_dv_dw = tl.load(
        base_dv_dw_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    kv = tl.load(kv_ptr + token * W + write_offsets, mask=valid_write, other=0.0)
    cumul_k = tl.load(
        cumul_k_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    post_decay = tl.load(
        post_decay_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    d_a_row = tl.load(
        d_kv_A_row_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    d_a_col = tl.load(
        d_kv_A_col_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )
    d_kv_qk = tl.load(
        d_kv_from_QK_ptr + token * W + write_offsets,
        mask=valid_write,
        other=0.0,
    )

    go_msq = tl.load(
        base_go_msq_ptr + token * R + read_offsets,
        mask=valid_read,
        other=0.0,
    )
    qv = tl.load(qv_ptr + token * R + read_offsets, mask=valid_read, other=0.0)
    cumul_q = tl.load(
        cumul_q_ptr + token * R + read_offsets,
        mask=valid_read,
        other=0.0,
    )
    d_qv_qk = tl.load(
        d_qv_from_QK_ptr + token * R + read_offsets,
        mask=valid_read,
        other=0.0,
    )

    d_kv_ret = base_ret * cumul_k
    d_lck_ret = base_ret * kv * cumul_k
    d_lck_a = kv * d_a_row - kv * d_a_col
    d_kv_a = d_a_row + d_a_col
    d_lck_qk = -kv * d_kv_qk
    d_kv_wv = base_dv_dw * post_decay
    d_pd_val = kv * base_dv_dw
    d_rev_lck = d_pd_val * post_decay
    d_g_pd = -d_pd_val * post_decay

    grad_kv = d_kv_a + d_kv_qk + d_kv_ret + d_kv_wv
    d_lck_total = d_lck_a + d_lck_qk + d_lck_ret

    d_lcq_qk = qv * d_qv_qk
    d_lcq_inter = go_msq * qv * cumul_q
    grad_qv = d_qv_qk + go_msq * cumul_q
    d_lcq_total = d_lcq_qk + d_lcq_inter

    tl.store(
        grad_kv_ptr + token * W + write_offsets,
        grad_kv,
        mask=valid_write,
    )
    tl.store(
        d_lck_total_ptr + token * W + write_offsets,
        d_lck_total,
        mask=valid_write,
    )
    tl.store(
        d_rev_lck_ptr + token * W + write_offsets,
        d_rev_lck,
        mask=valid_write,
    )
    tl.store(
        d_g_from_pd_ptr + token * W + write_offsets,
        d_g_pd,
        mask=valid_write,
    )
    tl.store(
        grad_qv_ptr + token * R + read_offsets,
        grad_qv,
        mask=valid_read,
    )
    tl.store(
        d_lcq_total_ptr + token * R + read_offsets,
        d_lcq_total,
        mask=valid_read,
    )


def masked_fused_bwd_elementwise(
    base_ret_all: torch.Tensor,
    base_dv_dw_all: torch.Tensor,
    wy_k_val_tm: torch.Tensor,
    wy_cumul_k_tm: torch.Tensor,
    wy_post_decay_tm: torch.Tensor,
    d_kv_A_row: torch.Tensor,
    d_kv_A_col: torch.Tensor,
    d_kv_from_QK: torch.Tensor,
    base_go_msq_all: torch.Tensor,
    wy_q_val_tm: torch.Tensor,
    wy_cumul_q_tm: torch.Tensor,
    d_qv_from_QK: torch.Tensor,
    grad_kv_tm: torch.Tensor,
    d_lck_total_all: torch.Tensor,
    d_rev_lck_all: torch.Tensor,
    d_g_from_pd_all: torch.Tensor,
    grad_qv_tm: torch.Tensor,
    d_lcq_total_all: torch.Tensor,
    *,
    num_writes: int,
    num_reads: int,
    block_writes: int,
    block_reads: int,
) -> None:
    tokens = base_ret_all.shape[0] * base_ret_all.shape[1]
    _masked_fused_bwd_elementwise_kernel[(tokens,)](
        base_ret_all,
        base_dv_dw_all,
        wy_k_val_tm,
        wy_cumul_k_tm,
        wy_post_decay_tm,
        d_kv_A_row,
        d_kv_A_col,
        d_kv_from_QK,
        base_go_msq_all,
        wy_q_val_tm,
        wy_cumul_q_tm,
        d_qv_from_QK,
        grad_kv_tm,
        d_lck_total_all,
        d_rev_lck_all,
        d_g_from_pd_all,
        grad_qv_tm,
        d_lcq_total_all,
        NCxC=tokens,
        W=num_writes,
        R=num_reads,
        BLOCK_W=block_writes,
        BLOCK_R=block_reads,
        num_warps=4,
    )


__all__ = ["masked_fused_bwd_elementwise"]
