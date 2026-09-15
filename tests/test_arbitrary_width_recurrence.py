from __future__ import annotations

import importlib
import unittest

import torch

from access_curriculum.arbitrary_width_sort import (
    install_arbitrary_access_width_sort,
)


def _dense_oracle(
    row_idx: torch.Tensor,
    row_val: torch.Tensor,
    row_lc: torch.Tensor,
    col_idx: torch.Tensor,
    col_val: torch.Tensor,
    col_lc: torch.Tensor,
    *,
    causal_mode: int,
) -> torch.Tensor:
    matches = row_idx[:, :, None, :, None] == col_idx[:, None, :, None, :]
    contributions = (
        row_val[:, :, None, :, None]
        * col_val[:, None, :, None, :]
        * torch.exp(row_lc[:, :, None, :, None] - col_lc[:, None, :, None, :])
    )
    output = (matches * contributions).sum(dim=(-2, -1))
    chunk_length = row_idx.shape[1]
    row_time = torch.arange(chunk_length, device=row_idx.device)[:, None]
    col_time = torch.arange(chunk_length, device=row_idx.device)[None, :]
    causal = row_time > col_time if causal_mode == 1 else row_time >= col_time
    return output * causal


def _sort_routes(
    sorter,
    indices: torch.Tensor,
    values: torch.Tensor,
    log_cumulative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = indices.shape
    sorted_indices, permutation = sorter(indices.reshape(-1, shape[-1]))
    permutation = permutation.reshape(shape)
    return (
        sorted_indices.reshape(shape),
        values.gather(-1, permutation),
        log_cumulative.gather(-1, permutation),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA accelerator")
class ArbitraryWidthRecurrenceCudaParityTest(unittest.TestCase):
    def test_non_power_of_two_forward_and_backward_match_dense_oracle(self) -> None:
        sort_module = importlib.import_module(
            "lingua.sparse_delta_memory.triton_argsort"
        )
        memory_ops = importlib.import_module(
            "lingua.sparse_delta_memory.memory_ops"
        )
        install_arbitrary_access_width_sort()
        generator = torch.Generator(device="cuda").manual_seed(20260831)
        chunks = 2
        chunk_length = 5
        logical_capacity = 257

        for width in (12, 20, 24):
            base = torch.arange(width, device="cuda", dtype=torch.int64)
            row_indices = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            (base * 7 + chunk * 11 + time * 3) % logical_capacity
                            for time in range(chunk_length)
                        )
                    )
                    for chunk in range(chunks)
                )
            )
            col_indices = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            (base * 7 + chunk * 11 + time * 5) % logical_capacity
                            for time in range(chunk_length)
                        )
                    )
                    for chunk in range(chunks)
                )
            )
            row_values = torch.randn(
                chunks,
                chunk_length,
                width,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            col_values = torch.randn(
                chunks,
                chunk_length,
                width,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            row_lc = 0.05 * torch.randn(
                chunks,
                chunk_length,
                width,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            col_lc = 0.05 * torch.randn(
                chunks,
                chunk_length,
                width,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )

            row_indices, row_values, row_lc = _sort_routes(
                sort_module.triton_argsort,
                row_indices,
                row_values,
                row_lc,
            )
            col_indices, col_values, col_lc = _sort_routes(
                sort_module.triton_argsort,
                col_indices,
                col_values,
                col_lc,
            )
            self.assertTrue(bool(torch.all(row_indices[..., 1:] >= row_indices[..., :-1])))
            self.assertTrue(bool(torch.all(col_indices[..., 1:] >= col_indices[..., :-1])))

            for causal_mode in (1, 2):
                expected_row = row_values.detach().clone().requires_grad_(True)
                expected_col = col_values.detach().clone().requires_grad_(True)
                expected = _dense_oracle(
                    row_indices,
                    expected_row,
                    row_lc,
                    col_indices,
                    expected_col,
                    col_lc,
                    causal_mode=causal_mode,
                )
                observed = memory_ops.sparse_inner_product_gated(
                    row_indices,
                    row_values,
                    row_lc,
                    col_indices,
                    col_values,
                    col_lc,
                    include_diag=causal_mode == 2,
                )
                torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)

                upstream = torch.randn(
                    observed.shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                (expected * upstream).sum().backward()
                observed_row_grad, observed_col_grad = (
                    memory_ops._sparse_ip_gated_bwd_vals(
                        row_indices,
                        row_values,
                        row_lc,
                        col_indices,
                        col_values,
                        col_lc,
                        upstream,
                        chunk_length,
                        causal_mode,
                    )
                )
                torch.testing.assert_close(
                    observed_row_grad,
                    expected_row.grad,
                    rtol=5e-5,
                    atol=5e-5,
                )
                torch.testing.assert_close(
                    observed_col_grad,
                    expected_col.grad,
                    rtol=5e-5,
                    atol=5e-5,
                )


if __name__ == "__main__":
    unittest.main()
