"""Exact bounded P2 selection for read or write widths above C.

The accepted selector remains the only path when K <= C. For K > C this
module scans the additive C-by-C score matrix in factor-1 blocks, retaining a
bounded exact frontier. It never constructs a tensor with an N-wide axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import types
from typing import Callable, Iterator

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SelectorTrace:
    selected: int
    factors: int
    codebook_size: int
    backend: str
    maximum_candidate_scratch_elements: int
    maximum_candidate_axis: int
    selected_route_tensor_elements: int
    logical_capacity_tensor_elements: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _iter_sdm_mixers(model: nn.Module) -> Iterator[nn.Module]:
    from reproduction.model import iter_sdm_mixers

    return iter_sdm_mixers(model)


def _ordered_float32_bits(values: torch.Tensor) -> torch.Tensor:
    """Map float32 values to monotonically ordered unsigned integers."""

    if values.dtype != torch.float32 or not values.is_contiguous():
        raise ValueError("ordered-bit conversion requires contiguous float32")
    values = torch.where(values == 0, torch.zeros_like(values), values)
    bits = values.view(torch.int32).to(torch.int64).bitwise_and(0xFFFF_FFFF)
    negative = bits.bitwise_right_shift(31).bool()
    return torch.where(
        negative,
        bits.bitwise_not().bitwise_and(0xFFFF_FFFF),
        bits.bitwise_xor(0x8000_0000),
    )


def _exact_positions(
    scores: torch.Tensor,
    logical_ids: torch.Tensor,
    keep: int,
    *,
    logical_capacity: int,
) -> torch.Tensor:
    """Select score-descending, logical-ID-ascending positions exactly."""

    if scores.ndim != 2 or scores.dtype != torch.float32:
        raise ValueError("candidate scores must be a float32 matrix")
    if logical_ids.shape != scores.shape or logical_ids.dtype != torch.int64:
        raise ValueError("candidate IDs must be an aligned int64 matrix")
    if keep <= 0 or keep > scores.shape[-1]:
        raise ValueError("top-K lies outside the candidate frontier")
    ordered = _ordered_float32_bits(scores.contiguous())
    stride = logical_capacity + 1
    keys = ordered * stride + (logical_capacity - logical_ids)
    return torch.topk(keys, k=keep, dim=-1, largest=True, sorted=False).indices


def exact_streaming_p2_scores(
    factor_scores: torch.Tensor,
    *,
    selected: int,
    scan_block: int,
    maximum_candidate_elements: int,
) -> tuple[torch.Tensor, torch.Tensor, SelectorTrace]:
    """Return exact additive P2 top-K routes using a bounded pair scan.

    Candidate comparisons are detached. Once the exact IDs are known, their
    scores are recomputed from only the selected live factor logits, preserving
    sparse backward through the selected entries.
    """

    if factor_scores.ndim != 4:
        raise ValueError("factor scores must be [B,T,2,C]")
    banks, time, factors, codebook_size = factor_scores.shape
    if factors != 2:
        raise ValueError("the streaming pair selector is P2-only")
    logical_capacity = codebook_size**2
    if not codebook_size < selected <= logical_capacity:
        raise ValueError("streaming selection requires C < K <= C²")
    if not 1 <= scan_block < codebook_size:
        raise ValueError("scan block must be positive and smaller than C")

    largest_axis = selected + scan_block * codebook_size
    if maximum_candidate_elements < largest_axis:
        raise ValueError("candidate budget cannot hold one streaming frontier")
    token_chunk = max(1, maximum_candidate_elements // largest_axis)
    flattened = factor_scores.reshape(-1, factors, codebook_size)
    output_values: list[torch.Tensor] = []
    output_ids: list[torch.Tensor] = []
    largest_elements = 0
    observed_axis = 0

    for token_start in range(0, flattened.shape[0], token_chunk):
        token_stop = min(token_start + token_chunk, flattened.shape[0])
        chunk = flattened[token_start:token_stop]
        token_count = token_stop - token_start
        first = chunk[:, 0]
        second = chunk[:, 1]
        frontier_scores: torch.Tensor | None = None
        frontier_ids: torch.Tensor | None = None

        for first_start in range(0, codebook_size, scan_block):
            first_stop = min(first_start + scan_block, codebook_size)
            block_scores = (
                first[:, first_start:first_stop].detach().float().unsqueeze(-1)
                + second.detach().float().unsqueeze(-2)
            ).reshape(token_count, -1)
            block_ids = torch.arange(
                first_start * codebook_size,
                first_stop * codebook_size,
                device=chunk.device,
                dtype=torch.int64,
            ).view(1, -1).expand(token_count, -1)
            if frontier_scores is not None:
                block_scores = torch.cat((frontier_scores, block_scores), dim=-1)
                block_ids = torch.cat((frontier_ids, block_ids), dim=-1)
            observed_axis = max(observed_axis, block_scores.shape[-1])
            largest_elements = max(largest_elements, block_scores.numel())
            keep = min(selected, block_scores.shape[-1])
            chosen = _exact_positions(
                block_scores,
                block_ids,
                keep,
                logical_capacity=logical_capacity,
            )
            frontier_scores = torch.gather(block_scores, -1, chosen)
            frontier_ids = torch.gather(block_ids, -1, chosen)
            id_order = frontier_ids.argsort(dim=-1, stable=True)
            frontier_scores = torch.gather(frontier_scores, -1, id_order)
            frontier_ids = torch.gather(frontier_ids, -1, id_order)

        if frontier_ids is None or frontier_ids.shape[-1] != selected:
            raise AssertionError("streaming pair scan did not resolve K routes")
        first_ids = torch.div(
            frontier_ids, codebook_size, rounding_mode="floor"
        )
        second_ids = torch.remainder(frontier_ids, codebook_size)
        selected_values = (
            torch.gather(first, -1, first_ids).float()
            + torch.gather(second, -1, second_ids).float()
        )
        output_values.append(selected_values)
        output_ids.append(frontier_ids)

    values = torch.cat(output_values, dim=0).reshape(banks, time, selected)
    ids = torch.cat(output_ids, dim=0).reshape(banks, time, selected)
    trace = SelectorTrace(
        selected=selected,
        factors=2,
        codebook_size=codebook_size,
        backend="p2-streaming-exact-pair-kgtc",
        maximum_candidate_scratch_elements=largest_elements,
        maximum_candidate_axis=observed_axis,
        selected_route_tensor_elements=values.numel() + ids.numel(),
    )
    return values.to(factor_scores.dtype), ids, trace


def install_exact_p2_selectors(
    model: nn.Module,
    *,
    codebook_size: int,
    scan_block: int,
    maximum_candidate_elements: int,
) -> None:
    """Install the K>C seam while preserving accepted K<=C behavior exactly."""

    found = False
    for mixer in _iter_sdm_mixers(model):
        found = True
        accepted: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]] = (
            mixer._select_product_keys_for_head
        )
        mixer._paper_accepted_selector = accepted
        mixer._paper_selector_trace = None
        mixer._paper_write_selector_trace = None
        mixer._paper_read_selector_trace = None
        mixer._paper_next_selector_side = "write"
        # Compatibility names consumed by the provenance-pinned bounded
        # diagnostics imported from the earlier curriculum implementation.
        mixer._topk_selector_trace = None
        mixer._topk_write_selector_trace = None
        mixer._topk_read_selector_trace = None

        def wrapper(
            bound_mixer: nn.Module,
            scores: torch.Tensor,
            selected: int,
            *,
            _accepted=accepted,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            side = str(bound_mixer._paper_next_selector_side)
            if side not in {"write", "read"}:
                raise RuntimeError("selector side state is invalid")
            if selected <= codebook_size:
                result = _accepted(scores, selected)
                trace = SelectorTrace(
                    selected=selected,
                    factors=2,
                    codebook_size=codebook_size,
                    backend=str(bound_mixer._last_selector_backend),
                    maximum_candidate_scratch_elements=0,
                    maximum_candidate_axis=0,
                    selected_route_tensor_elements=sum(row.numel() for row in result),
                )
                bound_mixer._paper_selector_trace = trace
                setattr(bound_mixer, f"_paper_{side}_selector_trace", trace)
                bound_mixer._topk_selector_trace = trace
                setattr(bound_mixer, f"_topk_{side}_selector_trace", trace)
                bound_mixer._paper_next_selector_side = (
                    "read" if side == "write" else "write"
                )
                return result

            values, indices, trace = exact_streaming_p2_scores(
                scores.reshape(*scores.shape[:-1], 2, codebook_size),
                selected=selected,
                scan_block=scan_block,
                maximum_candidate_elements=maximum_candidate_elements,
            )
            bound_mixer._last_selector_backend = trace.backend
            bound_mixer._paper_selector_trace = trace
            setattr(bound_mixer, f"_paper_{side}_selector_trace", trace)
            bound_mixer._topk_selector_trace = trace
            setattr(bound_mixer, f"_topk_{side}_selector_trace", trace)
            bound_mixer._paper_next_selector_side = (
                "read" if side == "write" else "write"
            )
            return values, indices

        mixer._select_product_keys_for_head = types.MethodType(wrapper, mixer)
    if not found:
        raise RuntimeError("P2 selector installation found no SDM layers")


def set_model_access(model: nn.Module, *, reads: int, writes: int) -> None:
    if reads <= 0 or writes <= 0:
        raise ValueError("read and write access counts must be positive")
    found = False
    for mixer in _iter_sdm_mixers(model):
        found = True
        if reads > int(mixer.slots_per_head) or writes > int(mixer.slots_per_head):
            raise ValueError("access width exceeds logical capacity")
        mixer.args.num_reads = int(reads)
        mixer.args.num_writes = int(writes)
    if not found:
        raise RuntimeError("access scheduling found no SDM layers")


def model_access(model: nn.Module) -> tuple[int, int]:
    pairs = {
        (int(mixer.args.num_reads), int(mixer.args.num_writes))
        for mixer in _iter_sdm_mixers(model)
    }
    if len(pairs) != 1:
        raise RuntimeError("SDM layers disagree on sparse access")
    return pairs.pop()


__all__ = [
    "SelectorTrace",
    "exact_streaming_p2_scores",
    "install_exact_p2_selectors",
    "model_access",
    "set_model_access",
]
