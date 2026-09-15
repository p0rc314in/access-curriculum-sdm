from __future__ import annotations

import unittest
from unittest import mock

import torch

from access_curriculum.arbitrary_width_sort import (
    _sort_selected_route_ids,
    install_arbitrary_access_width_sort,
)


class ArbitraryWidthSortDispatchTest(unittest.TestCase):
    def test_power_of_two_widths_delegate_exactly(self) -> None:
        sentinel_values = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        sentinel_perm = torch.tensor([[2, 1, 0]], dtype=torch.int64)
        accepted = mock.Mock(return_value=(sentinel_values, sentinel_perm))
        keys = torch.arange(16, dtype=torch.int64).reshape(1, 16)
        observed = _sort_selected_route_ids(keys, accepted)

        accepted.assert_called_once_with(keys)
        self.assertIs(observed[0], sentinel_values)
        self.assertIs(observed[1], sentinel_perm)

    def test_non_power_of_two_widths_are_stably_sorted(self) -> None:
        accepted = mock.Mock(side_effect=AssertionError("accepted path called"))

        for width in (12, 20, 24):
            keys = torch.tensor(
                [[(index * 7) % 11 for index in range(width)]],
                dtype=torch.int64,
            )
            values, permutation = _sort_selected_route_ids(keys, accepted)
            expected_values, expected_permutation = torch.sort(
                keys, dim=-1, stable=True
            )
            torch.testing.assert_close(values, expected_values)
            torch.testing.assert_close(permutation, expected_permutation)
        accepted.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA accelerator")
class ArbitraryWidthSortCudaParityTest(unittest.TestCase):
    def test_all_campaign_widths_match_stable_torch_oracle(self) -> None:
        import importlib

        module = importlib.import_module("lingua.sparse_delta_memory.triton_argsort")
        install_arbitrary_access_width_sort()
        generator = torch.Generator(device="cuda").manual_seed(20260831)

        for width in (8, 12, 16, 20, 24, 32):
            keys = torch.randint(
                0,
                257,
                (257, width),
                generator=generator,
                device="cuda",
                dtype=torch.int64,
            )
            expected_values, expected_permutation = torch.sort(
                keys, dim=-1, stable=True
            )
            values, permutation = module.triton_argsort(keys)
            torch.testing.assert_close(values, expected_values)
            torch.testing.assert_close(permutation, expected_permutation)


if __name__ == "__main__":
    unittest.main()
