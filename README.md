# Access Curricula for Sparse Delta Memory

## Intuition

Retrieval in Sparse Delta Memory (SDM) depends on a rendezvous between an
earlier write and a later read: the read must reach a row containing the
information it needs. Residualized routing gives the two roles a common map,
and an access curriculum broadens their opportunities to meet while that
map is being learned.

If R read rows and W write rows are chosen independently and uniformly from
N memory rows, each read has a W/N chance of reaching a written row. The
expected number of intersections is therefore:

```text
ρ = R × W / N
```

When broader access lets a read reach useful information from an earlier
write, the task loss can reinforce that connection. Gradually narrowing
access then gives the model time to adapt those connections to a tighter
budget.

## Construction

We tested eight curricula, varying opening read/write widths and contraction
schedules in an eight-layer SDM with residualized routing (N≈1k).
The [appendix](APPENDIX.md) contains results for all eight; we focus on the
best-performing schedule below:

| Stage | WikiText steps | Recall steps | Reads | Writes |
|---|---:|---:|---:|---:|
| Opening | 1–1,000 | 1–1,389 | 32 | 128 |
| Contraction 1 | 1,001–2,000 | 1,390–2,777 | 24 | 24 |
| Contraction 2 | 2,001–3,000 | 2,778–4,166 | 16 | 16 |
| Contraction 3 | 3,001–4,000 | 4,167–5,555 | 12 | 12 |
| Final access | 4,001–21,603 | 5,556–30,000 | 8 | 8 |

## Result

The comparison uses WikiText-103 and Adaptive Recall, with about 15M parameters
and 354M training tokens for the WikiText runs.

The native and residualized SDM controls use eight reads and eight writes
throughout training.

| Model | WikiText test NLL ↓ | Recall exact-set accuracy ↑ |
|---|---:|---:|
| Dense attention | 4.3625 | **58.70%** |
| Native SDM | 4.4821 | 44.86% |
| Residualized SDM | 4.3800 | 55.38% |
| Residualized SDM, curriculum | **4.3334** | 57.60% |

The curriculum brings residualized SDM's WikiText NLL below dense attention
while narrowing the remaining gap in exact recall.

![Side-by-side comparisons of dense attention, native SDM, residualized SDM, and residualized SDM with curriculum. WikiText test-NLL dots show 4.3625, 4.4821, 4.3800, and 4.3334 on a focused axis from 4.32 to 4.50. Adaptive Recall bars show overall exact-set accuracy of 58.70%, 44.86%, 55.38%, and 57.60%. Attention uses hatching, native SDM an outline, residualized SDM light gray, and curriculum black.](figures/curriculum-gain.png)

## Why it matters

The gains in language NLL and exact recall persist after access narrows,
improving on router residualization with no increase in the inference budget.
This is consistent with broader early access helping the router connect
later reads to useful information stored by earlier writes.

## Reproduction

```bash
uv sync --frozen
uv run --no-sync ./reproduce.sh prepare
# Prepare CUDA binaries as described in REPRODUCING.md.
uv run --no-sync ./reproduce.sh study --gpus 0,1,2,3
```

This runs the curriculum/control comparison and supporting ablations on both
tasks.
[REPRODUCING.md](REPRODUCING.md) gives runtime, cost, and the supporting suites;
[results](data/results.json) and [provenance](provenance.json) give the measurements
and their sources.
