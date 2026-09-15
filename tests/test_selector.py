from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from access_curriculum.selector import (
    exact_streaming_p2_scores,
    install_exact_p2_selectors,
    model_access,
    set_model_access,
)


def dense_oracle(scores: torch.Tensor, selected: int) -> tuple[torch.Tensor, torch.Tensor]:
    banks, time, _, codebook_size = scores.shape
    dense = (
        scores[:, :, 0].float().unsqueeze(-1)
        + scores[:, :, 1].float().unsqueeze(-2)
    ).reshape(banks, time, -1)
    rows_values = []
    rows_ids = []
    for bank in range(banks):
        time_values = []
        time_ids = []
        for position in range(time):
            ranked = sorted(
                range(codebook_size**2),
                key=lambda index: (-float(dense[bank, position, index]), index),
            )[:selected]
            ranked.sort()
            ids = torch.tensor(ranked, dtype=torch.int64, device=scores.device)
            time_ids.append(ids)
            time_values.append(torch.gather(dense[bank, position], 0, ids))
        rows_ids.append(torch.stack(time_ids))
        rows_values.append(torch.stack(time_values))
    return torch.stack(rows_values).to(scores.dtype), torch.stack(rows_ids)


class SelectorTest(unittest.TestCase):
    def test_k_at_or_below_c_delegates_exactly(self) -> None:
        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._last_selector_backend = "released-p2"
                self.calls = 0

            def _select_product_keys_for_head(self, scores, selected):
                self.calls += 1
                return scores[..., :selected], torch.arange(selected).expand(
                    *scores.shape[:-1], selected
                )

        mixer = Mixer()
        model = nn.Module()
        with patch(
            "access_curriculum.selector._iter_sdm_mixers",
            return_value=iter((mixer,)),
        ):
            install_exact_p2_selectors(
                model,
                codebook_size=32,
                scan_block=8,
                maximum_candidate_elements=8_388_608,
            )
        scores = torch.randn(1, 2, 64)
        expected_values = scores[..., :16]
        values, ids = mixer._select_product_keys_for_head(scores, 16)
        self.assertEqual(mixer.calls, 1)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(ids, torch.arange(16).expand(1, 2, 16))
        self.assertEqual(mixer._paper_write_selector_trace.backend, "released-p2")

    def test_matches_dense_oracle_for_every_production_width_above_c(self) -> None:
        generator = torch.Generator().manual_seed(20260828)
        scores = torch.randn(2, 3, 2, 32, generator=generator)
        for selected in (64, 128, 256, 512):
            values, ids, trace = exact_streaming_p2_scores(
                scores,
                selected=selected,
                scan_block=8,
                maximum_candidate_elements=8_388_608,
            )
            expected_values, expected_ids = dense_oracle(scores, selected)
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
            self.assertLess(trace.maximum_candidate_axis, 32**2)
            self.assertEqual(trace.logical_capacity_tensor_elements, 0)

    def test_ties_prefer_low_logical_ids(self) -> None:
        scores = torch.zeros(1, 1, 2, 32)
        _, ids, _ = exact_streaming_p2_scores(
            scores,
            selected=64,
            scan_block=8,
            maximum_candidate_elements=8_388_608,
        )
        torch.testing.assert_close(ids[0, 0], torch.arange(64), rtol=0, atol=0)

    def test_backward_reaches_only_selected_factor_entries(self) -> None:
        scores = torch.randn(1, 1, 2, 32, requires_grad=True)
        values, ids, _ = exact_streaming_p2_scores(
            scores,
            selected=64,
            scan_block=8,
            maximum_candidate_elements=8_388_608,
        )
        values.sum().backward()
        self.assertIsNotNone(scores.grad)
        expected = torch.zeros_like(scores, dtype=torch.bool)
        expected[0, 0, 0, torch.div(ids[0, 0], 32, rounding_mode="floor")] = True
        expected[0, 0, 1, torch.remainder(ids[0, 0], 32)] = True
        self.assertTrue(torch.equal(scores.grad != 0, expected))
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_rejects_k_at_or_below_c(self) -> None:
        with self.assertRaisesRegex(ValueError, "C < K"):
            exact_streaming_p2_scores(
                torch.randn(1, 1, 2, 32),
                selected=32,
                scan_block=8,
                maximum_candidate_elements=8_388_608,
            )

    def test_model_access_keeps_read_and_write_counts_separate(self) -> None:
        class Args:
            num_reads = 8
            num_writes = 8

        class Mixer:
            args = Args()
            slots_per_head = 1_024

        mixer = Mixer()
        with patch(
            "access_curriculum.selector._iter_sdm_mixers",
            side_effect=(iter((mixer,)), iter((mixer,))),
        ):
            set_model_access(nn.Module(), reads=128, writes=32)
            self.assertEqual(model_access(nn.Module()), (128, 32))


if __name__ == "__main__":
    unittest.main()
