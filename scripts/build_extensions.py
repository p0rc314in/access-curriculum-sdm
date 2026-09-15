"""Prepare checksum-addressed CUDA binaries on a build host, before experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

NAMES = ("sparse_ip_sorted", "warp_cooperative_gather")


def verify(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    import torch
    if manifest["torch"] != torch.__version__ or manifest["cuda"] != torch.version.cuda:
        raise ValueError("extension build and runtime versions differ")
    for name in NAMES:
        path = root / name / f"{name}.so"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"extension hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/extensions"))
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        print(json.dumps(verify(args.output)))
        return
    if os.environ.get("SDM2_CUDA_EXTENSION_DIR"):
        raise ValueError("unset SDM2_CUDA_EXTENSION_DIR when building")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("use a Linux CUDA development host for binary preparation")
    from lingua.sparse_delta_memory.cuda.sparse_ip_cuda import _get_cuda_module as sparse
    from lingua.sparse_delta_memory.cuda.warp_cooperative_gather_cuda import _get_cuda_module as gather
    modules = (sparse(), gather())
    hashes = {}
    for name, module in zip(NAMES, modules):
        dest = args.output / name / f"{name}.so"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module.__file__, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    record = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "architecture_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
              "build_gpu": torch.cuda.get_device_name(), "sha256": hashes}
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    verify(args.output)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
