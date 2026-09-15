"""Correct selected-route ordering for non-power-of-two access widths.

The accepted selector returns the exact top-K additive product keys and then
sorts the selected logical IDs for the sparse recurrence.  Its Triton bitonic
argsort is exact when K is a power of two, but its padded lanes do not
participate correctly for other widths.  This opt-in dispatch leaves the
accepted power-of-two path untouched and sorts only the already-selected K
routes for arbitrary widths.  It never materializes logical capacity.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _sort_selected_route_ids(
    keys: torch.Tensor,
    accepted: Callable[..., Any],
):
    if keys.ndim != 2 or keys.dtype != torch.int64:
        raise ValueError("selected logical IDs must be a two-dimensional int64 tensor")
    width = int(keys.shape[-1])
    if _power_of_two(width):
        return accepted(keys)
    return torch.sort(keys, dim=-1, stable=True)


def install_arbitrary_access_width_sort() -> None:
    """Install a stable exact sort for selected widths that are not powers of two."""

    module = importlib.import_module("lingua.sparse_delta_memory.triton_argsort")
    if getattr(module, "_paper_arbitrary_width_sort_installed", False):
        return

    accepted: Callable[..., Any] = module.triton_argsort

    def dispatch(keys: torch.Tensor):
        return _sort_selected_route_ids(keys, accepted)

    module.triton_argsort = dispatch
    module._paper_arbitrary_width_sort_installed = True
    module._paper_arbitrary_width_sort_accepted = accepted


__all__ = ["install_arbitrary_access_width_sort"]
