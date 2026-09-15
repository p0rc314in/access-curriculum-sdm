# Exact experiment definition

## Model and comparison

SDM layers: width 128, one memory head, value width 128, two routing
factors, C=32 entries per factor, N=C²=1,024 logical rows per layer. The
initial value of row (i,j) is the sum of two learned factor vectors. Residualized read and
write projections share a learned base with separate residuals, initialized
to the mean of the original role-keyed projections with zero residuals.
Native controls use independent read and write projections. Each SDM layer
retains its own recurrent memory. All stacks have eight layers; B7A1 replaces
the final SDM layer with four-head attention. B8 uses SDM in all eight layers.

Every sparse arm ends at R=W=8. Per-side access density is 8/1,024=0.78125%;
R×W/N is 0.0625, the expected overlap count for independent uniform route sets.
Broader training access
increases training work; deployment geometry and parameter count are constant.

Dense attention uses eight attention layers and four heads. The common width,
depth, feed-forward shell, input stream, seed, and number of updates are fixed.
Curricula retain the parameter count of their residualized fixed-access control.

| Stack / router | WikiText parameters | Recall parameters | Learned initial memory |
|---|---:|---:|---:|
| A8 | 14,965,120 | 2,138,496 | 0 |
| B8 / native | 15,038,880 | 2,212,256 | 65,536 |
| B7A1 / native | 15,029,660 | 2,203,036 | 57,344 |
| B8 / residualized | 15,104,928 | 2,278,304 | 65,536 |
| B7A1 / residualized | 15,087,452 | 2,260,828 | 57,344 |

WikiText uses untied input/output tables of 6,432,896 parameters each; Recall
uses a 14,592-parameter semantic adapter and a 24,576-parameter output head.
The remainder after subtracting those interfaces and learned initial memory
is the sequence processor. Rotary positions add no trainable parameters.
All counts are unique trainable parameters; recurrent request state is separate.

Common SDM parameters are role-key initialized on CPU before the model is
converted to BF16. The [initialization guard](reproduction/initialization.py)
checks every parameter and buffer at that precision before training; the
reproduction specifications retain the historical FP32 common-parameter hashes.

## Access schedules

The supporting eight-cell B8 factorial crosses ρ=R×W/N in {1,4}, a=W/R in {1,4}, and
direct versus staged contraction. All stages use inclusive optimizer indices.

| Cell | Initial R/W | ρ | a | Contraction |
|---|---:|---:|---:|---|
| A | 32/32 | 1 | 1 | Direct |
| B | 32/32 | 1 | 1 | Staged |
| C | 64/64 | 4 | 1 | Direct |
| D | 64/64 | 4 | 1 | Staged |
| E | 16/64 | 1 | 4 | Direct |
| F | 16/64 | 1 | 4 | Staged |
| G | 32/128 | 4 | 4 | Direct |
| H | 32/128 | 4 | 4 | Staged |

| Stage | WikiText steps | Recall steps | Direct access | Staged access |
|---|---|---|---|---|
| Opening | 1–1,000 | 1–1,389 | Initial R/W | Initial R/W |
| 2 | 1,001–2,000 | 1,390–2,777 | 8/8 | 24/24 |
| 3 | 2,001–3,000 | 2,778–4,166 | 8/8 | 16/16 |
| 4 | 3,001–4,000 | 4,167–5,555 | 8/8 | 12/12 |
| Terminal | 4,001–21,603 | 5,556–30,000 | 8/8 | 8/8 |

Selection at K≤C delegates to the pinned SDM selector. K>C uses an exact
streamed search over additive pair scores, retaining a bounded candidate set
with logical-ID tie breaking. Selected IDs at non-power-of-two K are stably
sorted before recurrence, and backward lane padding is masked.

## Training and evaluation

All arms use model seed zero. Tables report individual runs on the following
accelerators; the per-run software versions are in [provenance](provenance.json).

| Models | WikiText GPU | Recall GPU |
|---|---|---|
| Dense attention | A40 | A40 |
| B8 native, fixed access | RTX 4090 | RTX PRO 4000 Blackwell |
| B8 residualized, fixed access | RTX 4090 | RTX 4090 |
| B8 curricula A–F and H | RTX PRO 4500 Blackwell | RTX PRO 4000 Blackwell |
| B8 curriculum G | RTX 4090 | RTX PRO 4000 Blackwell |
| All B7A1 models | RTX PRO 4500 Blackwell | RTX PRO 4000 Blackwell |

WikiText protocol: `wikitext103-gpt2-causal-t2048-coverage-v1`.
`Salesforce/wikitext`, raw-v1 revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`; GPT-2 tokenizer, 50,257 tokens,
untied embedding/output weights, context 2,048. Exactly 117,980,449 training
targets per pass, three passes, 353,941,347 target presentations and 21,603
updates. Batch eight records; SDM accumulates eight microbatches of one,
attention uses one batch of eight. Stream seed 20260818.
Full validation/test score 247,416/283,426 targets once with stride 512.

WikiText AdamW: learning rate 3e-4, 540-step linear warmup, cosine decay to
3e-5; betas (0.9,0.95), epsilon 1e-8, weight decay 0.01, clip norm 1.0.
BF16 model/gradients with FP32 master parameters and FP32 moments.

Recall protocol: `adaptive-recall-seed102337-v1`. Thirty conditions cover
pointer chains, span retrieval, and overwrite-then-retrieve tasks. Each example
has 16 queries; training uses 30,000 full batches of 32, totaling 156,160,000
input tokens, with stream seed 102337 and a semantic embedding of entity,
slot, role, and hop fields. Each evaluation
split has 2,048 examples per condition: 61,440 examples and 983,040 queries,
evaluated in batches of eight. Query loss averages over queries; exact-set
accuracy averages the indicator that all 16 answers are correct.

Recall AdamW: learning rate 3e-4, 100-step warmup then constant; betas
(0.9,0.95), epsilon 1e-8, weight decay 0.01, clip norm 1.0. Every arm uses
the released optimizer with BF16 parameters and moments and a single full
training batch of 32.

## Opening access and contraction work together

A 64/64 opening has the same nominal overlap load as 32/128:
R×W/N = 4 under independent uniform routing. A four-cell comparison crosses
these two openings with either a direct jump to 8/8 or the staged contraction.

With a direct jump, 32/128 has lower WikiText NLL but worse exact recall than
64/64. Staging raises 32/128 exact recall by **1.65 percentage points** and
further lowers WikiText NLL, leaving it ahead on both metrics. Staging 64/64
improves language loss while exact recall changes from 57.34% to 57.21%.

## Factorial results

Each effect averages the terminal metric difference across the other two
factor axes in these recorded runs. Lower WikiText NLL and higher Recall exact-set accuracy are
improvements.

| Factor change | WikiText NLL | Recall exact-set accuracy (pp) |
|---|---:|---:|
| Staged minus direct | -0.003294 | +1.109 |
| ρ=4 minus ρ=1 | -0.001862 | +0.797 |
| a=4 minus a=1 | -0.007665 | -0.264 |

Staging improves WikiText NLL in all four pairs and exact recall in three.
Cell H, the staged 32/128 curriculum, has the lowest WikiText NLL and highest
Recall exact-set accuracy.

| Cell | Opening R/W | Contraction | WikiText NLL | Recall exact-set accuracy |
|---|---:|---|---:|---:|
| A | 32/32 | Direct | 4.34532 | 55.66% |
| B | 32/32 | Staged | 4.34488 | 56.84% |
| C | 64/64 | Direct | 4.34888 | 57.34% |
| D | 64/64 | Staged | 4.34443 | 57.21% |
| E | 16/64 | Direct | 4.34283 | 55.34% |
| F | 16/64 | Staged | 4.33888 | 57.08% |
| G | 32/128 | Direct | 4.33774 | 55.96% |
| H | 32/128 | Staged | 4.33341 | 57.60% |

## Complete topology comparison

The 22 endpoints are eleven models on each of the two tasks: five SDM policies
in B8 and B7A1, plus dense A8. All values below use the full test split.

| Policy | Stack | WikiText NLL | Recall exact-set accuracy |
|---|---|---:|---:|
| Dense attention | 8 attention | 4.36248 | 58.70% |
| Native, fixed 8/8 | 8 SDM | 4.48206 | 44.86% |
| Native, fixed 8/8 | 7 SDM + attention | 4.45031 | 56.28% |
| Residualized, fixed 8/8 | 8 SDM | 4.37999 | 55.38% |
| Residualized, fixed 8/8 | 7 SDM + attention | 4.36357 | 55.46% |
| 32/128, staged | 8 SDM | 4.33341 | 57.60% |
| 32/128, staged | 7 SDM + attention | 4.33243 | 57.20% |
| 64/64, staged | 8 SDM | 4.34443 | 57.21% |
| 64/64, staged | 7 SDM + attention | 4.34041 | 57.07% |
| 16/64, staged | 8 SDM | 4.33888 | 57.08% |
| 16/64, staged | 7 SDM + attention | 4.32788 | 57.16% |

## Recall breakdown

Exact-set accuracy below is aggregated from correct examples. Each condition
contains 2,048 examples: pointer chasing has 16 conditions, span recall has
eight, and overwrite recall has six. The overall score weights these families
by their example counts. All values below use the full test split.

For one-hop retrieval, fixed access reaches
99.62% exact-set accuracy and the curriculum reaches 99.90% across 8,192
examples. Every recorded model scores zero exact sets on the 2-, 4-, and
8-hop conditions in both evaluation splits. Because the four hop depths have
equal weight, the full pointer-family accuracy is one quarter of its one-hop
accuracy; the overall suite score includes all four depths.

| Policy | Stack | Pointer chasing | Span recall | Overwrite recall |
|---|---|---:|---:|---:|
| Dense attention | 8 attention | 24.85% | 98.63% | 95.74% |
| Native, fixed 8/8 | 8 SDM | 24.82% | 56.34% | 82.98% |
| Native, fixed 8/8 | 7 SDM + attention | 22.37% | 94.41% | 95.88% |
| Residualized, fixed 8/8 | 8 SDM | 24.91% | 83.70% | 98.91% |
| Residualized, fixed 8/8 | 7 SDM + attention | 24.94% | 83.91% | 98.90% |
| 32/128, staged | 8 SDM | 24.98% | 91.63% | 99.24% |
| 32/128, staged | 7 SDM + attention | 24.95% | 89.90% | 99.58% |
| 64/64, staged | 8 SDM | 24.96% | 90.97% | 98.17% |
| 64/64, staged | 7 SDM + attention | 24.98% | 89.44% | 99.49% |
| 16/64, staged | 8 SDM | 24.95% | 89.63% | 99.38% |
| 16/64, staged | 7 SDM + attention | 24.99% | 90.48% | 98.49% |

[Complete breakdowns](data/recall-breakdown.json) retain all 30 conditions and
family totals on validation and test, keyed to the source identities in
[provenance](provenance.json).
