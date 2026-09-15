"""Verify the BF16 initialization actually consumed by training.

The reference parameters reproduce the recorded FP32 common hashes on Linux
AMD EPYC with the locked PyTorch 2.11 CUDA 12.8 build. Intel initialization
differs in low FP32 bits that disappear in BF16. The guard covers all named
parameters and buffers, including frozen routes, priors and nonpersistent RoPE.
Historical FP32 fingerprints remain in spec.py and recall_spec.py.
"""
from __future__ import annotations

import hashlib
import json

import torch


BF16_STATE_SHA256 = {
    ('recall', 'BBBBBBBB', 'unmodified'): '19dd1b623d584e08cb5f3493cf1a92b1444e8971a69ec3b8a676e11fc50808c1',
    ('recall', 'BBBBBBBB', 'residualized_read_write'): '8743ae0bdea0cac3baff0501ef8901d5c86f3829163a347ebb31234f2600d033',
    ('recall', 'BBBBBBBA', 'unmodified'): 'ef9862dbb1f1e7561e930e1abe05021b9f5947c0d6f93e996bcf5c73ae5ef014',
    ('recall', 'BBBBBBBA', 'residualized_read_write'): 'b28aa20309509b940c7ef9762efe4efb88d06980218bb796ea2122107c3c558a',
    ('wikitext', 'BBBBBBBB', 'unmodified'): 'aa75b5fe1ecaf7974ec6b8efd5739eacf5357500252559c75bb40f069c806122',
    ('wikitext', 'BBBBBBBB', 'residualized_read_write'): 'aeb6b7f4965e92d54f3f81792849da84780b12483cc4e1b8433351545d22633f',
    ('wikitext', 'BBBBBBBA', 'unmodified'): '03fd0a734ecacf2b82533cbf0faa01f0d29d7bd629648ed183a673589c1ed229',
    ('wikitext', 'BBBBBBBA', 'residualized_read_write'): 'f18115b4c103fb31990f6ca6df7208b319603d5b839777331d1a3eb7ed6aa398',
}


def initialization_sha256(model: torch.nn.Module) -> str:
    """Hash names, shapes, dtypes and bytes without modifying model or RNG."""
    digest = hashlib.sha256()
    groups = (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    )
    for kind, tensors in groups:
        for name, tensor in sorted(tensors):
            value = tensor.detach().cpu().contiguous()
            header = [kind, name, list(value.shape), str(value.dtype)]
            digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\0")
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_initialization(model: torch.nn.Module, *, task: str, layout: str, router: str) -> str:
    expected = BF16_STATE_SHA256[(task, layout, router)]
    observed = initialization_sha256(model)
    if observed != expected:
        raise ValueError(
            f"{task} {layout} {router} BF16 initialization identity changed: "
            f"{observed}; expected {expected}"
        )
    return observed
