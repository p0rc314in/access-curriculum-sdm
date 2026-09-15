"""Maintain the compact result/figure inventory; never substitute for training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    paths = ("data/results.json", "data/recall-breakdown.json", "figures/curriculum-gain.png", "figures/social-preview.png", "provenance.json", "PROJECT_IDENTITY.json", "reproduction/extraction.json", "reproduction/initialization.py")
    if args.write:
        payload = {"schema_version": 1, "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}
        (ROOT / "data/artifact-manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    from scripts.verify_results import validate
    validate()
    print("Result and figure inventory verified.")


if __name__ == "__main__":
    main()
