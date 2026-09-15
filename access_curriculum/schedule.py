"""The recorded eight-cell access experiment, with inclusive step boundaries."""
from __future__ import annotations

WIKI_STEPS = 21_603
RECALL_STEPS = 30_000
CELLS = {
    "A": (32, 32, "direct"), "B": (32, 32, "staged"),
    "C": (64, 64, "direct"), "D": (64, 64, "staged"),
    "E": (16, 64, "direct"), "F": (16, 64, "staged"),
    "G": (32, 128, "direct"), "H": (32, 128, "staged"),
}
ALIASES = {"wiki_forward": "H", "recall_forward": "D", "balanced": "F"}
POLICIES = ("native_k8", "fixed_k8", "wiki_forward", "recall_forward", "balanced")
HEADLINE_ARMS = ("dense_attention", *(arm for policy in POLICIES for arm in (policy, "hybrid_" + policy)))
# Reuse recorded identities so study/topology/full runs share completed outputs.
STUDY_ARMS = ("fixed_k8", "wiki_forward", "recall_forward", "G", "C")
FACTORIAL_ARMS = tuple(CELLS)


def stages(arm: str, benchmark: str) -> tuple[tuple[int, int, int, int], ...]:
    """Return (first step, last step, reads, writes) for every training stage."""
    if benchmark not in ("wikitext", "recall"):
        raise ValueError("benchmark must be wikitext or recall")
    total = WIKI_STEPS if benchmark == "wikitext" else RECALL_STEPS
    arm = arm.removeprefix("hybrid_")
    if arm == "dense_attention":
        return ()
    if arm in ("native_k8", "fixed_k8"):
        return ((1, total, 8, 8),)
    reads, writes, contraction = CELLS[ALIASES.get(arm, arm)]
    boundaries = (1000, 2000, 3000, 4000) if benchmark == "wikitext" else (1389, 2777, 4166, 5555)
    rows = [(1, boundaries[0], reads, writes)]
    if contraction == "staged":
        rows.extend((left + 1, right, k, k) for left, right, k in zip(boundaries, boundaries[1:], (24, 16, 12)))
    rows.append((rows[-1][1] + 1, total, 8, 8))
    return tuple(rows)


def access_at_step(arm: str, step: int, benchmark: str) -> tuple[int, int]:
    for first, last, reads, writes in stages(arm, benchmark):
        if first <= step <= last:
            return reads, writes
    raise ValueError("step is outside the SDM schedule")


def install(model) -> None:
    from .arbitrary_width import install_arbitrary_access_width_backward
    from .arbitrary_width_sort import install_arbitrary_access_width_sort
    from .selector import install_exact_p2_selectors
    install_arbitrary_access_width_backward()
    install_arbitrary_access_width_sort()
    install_exact_p2_selectors(model, codebook_size=32, scan_block=8, maximum_candidate_elements=8_388_608)


def apply(model, arm: str, step: int, benchmark: str) -> tuple[int, int] | None:
    if arm == "dense_attention":
        return None
    from .selector import set_model_access
    reads, writes = access_at_step(arm, step, benchmark)
    set_model_access(model, reads=reads, writes=writes)
    return reads, writes
