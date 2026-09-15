# Third-party sources

The vendored runtime derives from the official
[Sparse Delta Memory implementation](https://github.com/facebookresearch/sparse-delta-memory),
revision `183e7df809131b80ad4393741029d0f20fc3640b`, and its accepted
capacity-independent selected-row adapter. `third_party/runtime/manifest.json`
records the exact lineage and file/tree hashes. The source is included here;
development repositories are provenance, not retrieval dependencies.

Sparse Delta Memory code retains CC BY-NC 4.0 licensing. Meta Lingua-derived
code retains BSD 3-Clause licensing. Both license texts are included under
`third_party/runtime/`. The repository MIT license covers original prose,
curriculum, experiment, and integration code. The lane-masked backward kernel
in `access_curriculum/arbitrary_width_kernel.py` is adapted from SDM and
retains its CC BY-NC 4.0 terms.

WikiText-103 is retrieved from `Salesforce/wikitext` at revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`. Its dataset card records
CC BY-SA and GFDL licensing. Corpus files and model weights are not
redistributed. Adaptive Recall inputs are deterministically generated from
the source included in this repository.
