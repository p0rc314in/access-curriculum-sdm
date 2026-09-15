"""Validate fresh terminal results and checkpoints; report observed differences."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from reproduction.spec import EXPECTED_MANIFEST_SHA256, TOTAL_STEPS
from reproduction.recall_spec import MANIFEST_SHA256, STEPS

ROOT = Path(__file__).resolve().parents[1]


def verified_file(root: Path, record: dict) -> Path:
    filename = (root / record["path"]).resolve()
    if root.resolve() not in filename.parents or not filename.is_file():
        raise ValueError("required output is absent or outside its arm directory")
    with filename.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"output hash changed: {filename.name}")
    return filename


def check_recall_predictions(path: Path, result: dict, data) -> None:
    import numpy as np
    from reproduction.recall_generate import CONDITIONS, OUTPUT_CLASSES, QUERIES
    from reproduction.recall_spec import EVAL_EXAMPLES
    if data is None:
        raise ValueError("Recall verification requires the prepared labels")
    expected = {c["id"]: c for c in CONDITIONS}
    for split in ("validation", "test"):
        rows = result[split]["by_condition"]
        if len(rows) != len(expected) or {r["condition_id"] for r in rows} != set(expected):
            raise ValueError("Recall condition coverage changed")
        for row in rows:
            condition = expected[row["condition_id"]]
            if row["family"] != condition["family"] or row["examples"] != EVAL_EXAMPLES:
                raise ValueError("Recall condition identity or example count changed")
            prediction = np.load(verified_file(path, row["predictions"]), allow_pickle=False)
            if (prediction.shape != (EVAL_EXAMPLES, QUERIES) or prediction.dtype != np.uint8
                    or np.any(prediction >= OUTPUT_CLASSES)):
                raise ValueError("Recall prediction shape, dtype, or class changed")
            _, labels = data.evaluation(split, condition["id"])
            if labels.shape != prediction.shape:
                raise ValueError("Recall label shape changed")
            correct = prediction == labels
            if int(correct.sum()) != row["correct_queries"] or int(correct.all(-1).sum()) != row["exact_sets"]:
                raise ValueError("saved Recall predictions do not reproduce reported accuracy")
            if not math.isfinite(row["loss_sum"]) or row["loss_sum"] < 0:
                raise ValueError("invalid Recall condition loss")

        def aggregate(selected):
            examples = sum(r["examples"] for r in selected)
            queries = examples * QUERIES
            return {"examples": examples, "queries": queries,
                    "query_loss": sum(r["loss_sum"] for r in selected) / queries,
                    "query_accuracy": sum(r["correct_queries"] for r in selected) / queries,
                    "exact_set_accuracy": sum(r["exact_sets"] for r in selected) / examples}

        summaries = [(result[split]["aggregate"], aggregate(rows))]
        families = {c["family"] for c in CONDITIONS}
        if set(result[split]["by_family"]) != families:
            raise ValueError("Recall family coverage changed")
        summaries.extend((result[split]["by_family"][f], aggregate([r for r in rows if r["family"] == f]))
                         for f in families)
        for observed, computed in summaries:
            if set(observed) != set(computed) or any(
                    not math.isclose(observed[k], v, rel_tol=0, abs_tol=1e-12) for k, v in computed.items()):
                raise ValueError("Recall aggregate differs from retained condition evidence")


def check_arm(path: Path, benchmark: str, arm: str, *, recall_data=None):
    if benchmark not in ("wikitext", "recall"):
        raise ValueError("unknown benchmark")
    row = json.loads((path / "result.json").read_text())
    if row.get("status") != "terminal_verified" or row.get("arm") != arm:
        raise ValueError("result is not the requested completed arm")
    expected_manifest = EXPECTED_MANIFEST_SHA256 if benchmark == "wikitext" else MANIFEST_SHA256
    if row.get("manifest_sha256") != expected_manifest:
        raise ValueError("result input identity differs")
    cfg = row["config"]
    if (cfg.get("reads"), cfg.get("writes")) != (8, 8):
        raise ValueError("terminal access changed")
    # Compare the complete resolved identity, including inclusive curriculum boundaries.
    if benchmark == "wikitext":
        from dataclasses import asdict
        from reproduction.train_wikitext import resolved_config
        expected = asdict(resolved_config(arm))
    else:
        from reproduction.recall_spec import config
        expected = config(arm)
    if cfg != json.loads(json.dumps(expected)):
        raise ValueError("resolved experiment configuration changed")
    if benchmark == "wikitext":
        if cfg["total_steps"] != TOTAL_STEPS or not row["coverage"]["complete"]:
            raise ValueError("incomplete WikiText training coverage")
        for split, targets in (("validation", 247416), ("test", 283426)):
            if row["terminal_" + split]["scored_targets"] != targets:
                raise ValueError("incomplete WikiText evaluation coverage")
        metrics = {"nll": row["terminal_test"]["nll"]}
        band = (4.28, 4.53)
    else:
        if row["steps"] != STEPS or not row["checkpoint_roundtrip_inference_passed"]:
            raise ValueError("incomplete Recall training or checkpoint verification")
        for split in ("validation", "test"):
            aggregate = row[split]["aggregate"]
            if aggregate["examples"] != 61440 or aggregate["queries"] != 983040:
                raise ValueError("incomplete Recall evaluation coverage")
        check_recall_predictions(path, row, recall_data)
        metrics = row["test"]["aggregate"]
        band = (0.40, 0.65)
    filename = verified_file(path, row["terminal_checkpoint"])
    import torch
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    schema = ("access-curriculum-terminal-model-v1" if benchmark == "wikitext"
              else "access-curriculum-recall-checkpoint-v1")
    if (payload.get("schema") != schema or payload.get("manifest_sha256") != expected_manifest
            or json.loads(json.dumps(payload.get("config"))) != cfg):
        raise ValueError("terminal checkpoint schema or experiment identity changed")
    state = payload.get("model")
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("terminal checkpoint model state is absent or malformed")
    if benchmark == "wikitext" and payload.get("arm") != arm:
        raise ValueError("terminal checkpoint arm changed")
    if benchmark == "recall" and (payload.get("step") != STEPS or payload.get("mode") != "experiment"):
        raise ValueError("Recall checkpoint is not a terminal experiment")
    metric = "nll" if benchmark == "wikitext" else "exact_set_accuracy"
    value = metrics[metric]
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite terminal metric")
    # Bands are expectations, not a mechanism claim or a reason to discard a run.
    return {"arm": arm, "benchmark": benchmark, "metrics": metrics,
            "primary_metric": metric, "expected_metric_range": band,
            "within_expected_range": band[0] <= value <= band[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "runs/experiments")
    parser.add_argument("--suite", choices=("study", "topology", "headline", "full"), default="study")
    parser.add_argument("--benchmark", choices=("wikitext", "recall", "both"), default="both")
    parser.add_argument("--data", type=Path, default=ROOT / "runs/data")
    args = parser.parse_args()
    from scripts.run_experiments import selected_arms
    benchmarks = ("wikitext", "recall") if args.benchmark == "both" else (args.benchmark,)
    jobs = [(b, a) for b in benchmarks for a in selected_arms(args.suite)]
    missing = [f"{b}/{a}" for b, a in jobs if not (args.output / b / a / "result.json").is_file()]
    if missing:
        raise SystemExit("incomplete selected suite; missing results: " + ", ".join(missing))
    recall_data = None
    if "recall" in benchmarks:
        from scripts.prepare_recall import check
        recall_data = check(args.data / "recall/manifest.json")
    results = [check_arm(args.output / b / a, b, a, recall_data=recall_data) for b, a in jobs]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
