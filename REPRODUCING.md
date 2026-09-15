# Reproducing the experiments

`reproduce.sh` prepares the inputs, trains the models, and evaluates them.
`verify-results` checks the recorded tables and figures.

## Requirements and expected cost

Use Linux x86-64, Python 3.12 or 3.13, PyTorch 2.11 with CUDA 12.8, and
BF16-capable NVIDIA GPUs. RTX 4090 / RTX PRO 4000–4500 Blackwell-class devices
were used for the sparse runs. Allow 24 GB device memory and at least 40 GB
local disk for prepared streams, binary builds, and retained checkpoints.

The recorded sparse runs took approximately 6–9 GPU-hours for WikiText and
6–7 for Recall. At their $0.57–$0.72 hourly rates, the suites have
these approximate compute requirements, plus preparation and storage:

| Suite | Models × tasks | GPU-hours | Compute cost | Wall time on four GPUs |
|---|---:|---:|---:|---:|
| Study: fixed control + C/D/G/H | 5 × 2 | 60–80 | $35–$60 | 15–22 hours |
| Supporting topology comparison | 11 × 2 | 120–165 | $70–$120 | 30–45 hours |
| Full factorial + topology | 16 × 2 | 180–245 | $105–$180 | 45–65 hours |

Independent jobs run concurrently, one per listed GPU, with remaining jobs
queued in successive waves. The four-GPU estimates above include those waves;
the longest arm determines suite time only when every job has its own GPU.
Use matching hardware within each new comparison. Early broad access costs more than
terminal K8 and may use more memory.

## Prepare before allocating experiment GPUs

```bash
git clone https://github.com/p0rc314in/access-curriculum-sdm.git
cd access-curriculum-sdm
uv sync --frozen
uv run --no-sync ./reproduce.sh prepare
```

WikiText is downloaded from its pinned public revision, serialized/tokenized
twice, and checked against the recorded manifest and complete input inventory.
Recall is generated twice from its frozen public code and seed. Every array
must match the manifest. Training loads these prepared arrays.
Both commands can revalidate an existing complete input directory.

Prepare the two SDM CUDA binaries on a Linux CUDA development host with the
same locked environment and target accelerator architecture, before starting
training jobs:

```bash
uv run --no-sync ./reproduce.sh build-extensions
```

This stage requires the CUDA toolkit and a visible compatible GPU. It writes
`runs/extensions/manifest.json` plus two `.so` files. Transfer that directory
and `runs/data/` to the training machine, retaining their hashes. Binaries are
specific to the Python/PyTorch/CUDA environment and compiled GPU architecture;
build separately for a different target. Training requires these binaries;
missing extensions stop the run. Triton kernels specialize during first use.

## Run the comparison

```bash
# Fixed residualized B8 plus 32/128 and 64/64, each direct and staged.
uv run --no-sync ./reproduce.sh study --gpus 0,1,2,3

# Supporting 22-endpoint topology comparison.
uv run --no-sync ./reproduce.sh topology --gpus 0,1,2,3

# Complete eight-cell factorial and topology; reuse verified finished jobs.
uv run --no-sync ./reproduce.sh full --gpus 0,1,2,3

# Preview the primary study's jobs and configurations.
uv run --no-sync ./reproduce.sh study --dry-run

# Restrict to one task, or choose independent input/output directories.
uv run --no-sync ./reproduce.sh full --benchmark recall --gpus 0,1 \
  --data runs/data --output runs/experiments --extensions runs/extensions
```

The study reuses the recorded arm IDs: `fixed_k8`, `wiki_forward` (cell H,
32/128 staged), `recall_forward` (D, 64/64 staged), `G` (32/128 direct), and
`C` (64/64 direct). The topology comparison also includes `balanced` (F,
16/64 staged). `headline` remains an alias for `topology`. Output identities
are shared across suites, so moving from `study` to `full` reuses finished arms.

`all` performs input preparation followed by the full training/evaluation suite;
the CUDA binaries must already exist. The default training output is
`runs/experiments/<wikitext|recall>/<arm>/`. Each arm retains its resolved
configuration, metrics, training log, recovery state, and terminal model
checkpoint with SHA-256. Recall also retains per-condition predictions and
checks inference after checkpoint reload. The scheduler skips only verified
terminal results, resumes verified recovery checkpoints, and lets independent
jobs finish if another arm fails. Rerun the same command after resolving the
failed arm's log. Copy terminal checkpoints and metrics to durable storage
before removing an experiment machine.

## Expected measurements and checks

The primary metrics are WikiText test NLL and Recall exact-set accuracy,
approximately 4.33–4.48 and 45–59%, respectively. Broad diagnostic
acceptance ranges are 4.28–4.53 WikiText NLL and 40–65% Recall exact-set accuracy.
The checker retains every measurement and flags values outside these ranges.
Report the measured values and arm ordering from each completed reproduction.
The implementation fixes precision, optimizer, batching, and initialization.
Repeated multi-chunk BF16 prefill can produce different logits with unchanged
SDM weights, including in the released native runtime. WikiText recovery
restores the saved model and optimizer state. Every Recall arm uses a 100-step
warmup followed by a constant learning rate of 3e-4.
The initialization guard checks every parameter and buffer after BF16 conversion,
including routing projections, memory priors, and rotary positions. Historical
FP32 common-parameter fingerprints remain in the source for provenance.

```bash
uv run --no-sync ./reproduce.sh check
# Check the complete supporting suite, including retained Recall predictions.
uv run --no-sync ./reproduce.sh check --suite full --data runs/data
# Match a task/output restriction used for training.
uv run --no-sync ./reproduce.sh check runs/experiments --suite full --benchmark recall
uv run --no-sync ./reproduce.sh verify-results
uv run --no-sync ./reproduce.sh test
```

The result checker requires every arm in the selected suite and benchmark.
It verifies terminal checkpoint identity and hashes, all Recall prediction
files and condition totals, and recomputes query and exact-set accuracy from
the prepared labels. The default selection is `study` on both tasks.

The CPU tests cover exact bounded selection, tie handling, sparse score
gradients, schedule boundaries, retry behavior, and output verification.
CUDA tests check selected-ID sorting and sparse inner-product forward/backward
against dense oracles. The lane-masked backward dispatch is checked at K12/K24,
including unequal widths, against the released kernel with zero-padded lanes
in FP32 and BF16. Power-of-two widths are checked through the original dispatch.
CUDA-dependent checks are explicitly skipped on CPU hosts. Dense attention
Recall completed all 30,000 updates and both full evaluation splits from this
reproduction code on one A40 in about 25 minutes. The terminal checkpoint
passed exact inference after reload; saved predictions reproduce the reported
query and exact-set accuracies.
