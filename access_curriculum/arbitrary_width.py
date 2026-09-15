"""Opt-in dispatch for non-power-of-two sparse access widths.

The released WY recurrence already consumes tensors whose last dimensions are
the actual write and read counts. One fused backward elementwise kernel used
those counts directly as ``tl.arange`` extents, which Triton only accepts when
they are powers of two. This dispatch pads only that kernel's lane blocks and
masks the padding; it does not add selected routes or change recurrence math.

The Triton implementation is imported only when installed on the accelerator,
so source validation and CPU-only campaign tooling do not require Triton.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("access width must be positive")
    return 1 << (value - 1).bit_length()


def install_arbitrary_access_width_backward() -> None:
    """Dispatch non-power-of-two widths through the masked kernel only."""

    from lingua.sparse_delta_memory import memory_ops

    from .arbitrary_width_kernel import masked_fused_bwd_elementwise

    if getattr(memory_ops, "_topk_arbitrary_width_installed", False):
        return
    accepted: Callable[..., Any] = memory_ops.fused_bwd_elementwise

    def dispatch(
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
        num_writes: int,
        num_reads: int,
    ) -> None:
        if _power_of_two(num_writes) and _power_of_two(num_reads):
            accepted(
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
                num_writes,
                num_reads,
            )
            return

        masked_fused_bwd_elementwise(
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
            num_writes=num_writes,
            num_reads=num_reads,
            block_writes=_next_power_of_two(num_writes),
            block_reads=_next_power_of_two(num_reads),
        )

    memory_ops.fused_bwd_elementwise = dispatch
    memory_ops._topk_arbitrary_width_installed = True
    memory_ops._topk_arbitrary_width_accepted = accepted


__all__ = ["install_arbitrary_access_width_backward"]
