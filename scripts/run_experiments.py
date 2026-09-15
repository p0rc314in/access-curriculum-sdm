"""Run independent recorded arms concurrently, one process per CUDA device."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from access_curriculum.schedule import HEADLINE_ARMS, STUDY_ARMS, ALIASES, CELLS
from scripts.check_reproduction import check_arm

ROOT = Path(__file__).resolve().parents[1]


def selected_arms(suite: str):
    if suite == "study":
        return STUDY_ARMS
    return HEADLINE_ARMS if suite in ("headline", "topology") else (*HEADLINE_ARMS, *(c for c in CELLS if c not in ALIASES.values()))


def command(benchmark, arm, manifest, output):
    cmd = [sys.executable, "-m", f"reproduction.train_{benchmark}", "--arm", arm,
           "--manifest", str(manifest), "--output", str(output), "--reads", "8", "--writes", "8"]
    latest = output / "LATEST_RECOVERY.json"
    if benchmark == "wikitext" and latest.is_file():
        row = json.loads(latest.read_text())
        path = (output / row["path"]).resolve()
        if output.resolve() not in path.parents:
            raise ValueError("recovery checkpoint escapes its arm directory")
        cmd.extend(["--resume", str(path), "--resume-sha256", row["sha256"]])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitext", "recall", "both"), default="both")
    parser.add_argument("--suite", choices=("study", "topology", "headline", "full"), default="study")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    parser.add_argument("--data", type=Path, default=ROOT / "runs/data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiments")
    parser.add_argument("--extensions", type=Path, default=ROOT / "runs/extensions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError("--gpus requires distinct comma-separated device indices")
    benchmarks = ("wikitext", "recall") if args.benchmark == "both" else (args.benchmark,)
    output, data, extensions = args.output.resolve(), args.data.resolve(), args.extensions.resolve()
    if output == data or data in output.parents:
        raise ValueError("results must be outside immutable inputs")
    jobs = [(bench, arm) for bench in benchmarks for arm in selected_arms(args.suite)]
    if args.dry_run:
        for bench, arm in jobs:
            print(json.dumps({"benchmark": bench, "arm": arm, "evidence_tier": "small_wikitext_plus_recall",
                              "command": command(bench, arm, data / bench / "manifest.json", output / bench / arm)}))
        return
    from scripts.verify_vendor import validate_vendor
    from scripts.build_extensions import verify
    validate_vendor(ROOT / "third_party/runtime")
    verify(extensions)
    recall_data = None
    for bench in benchmarks:
        if bench == "wikitext":
            subprocess.run([sys.executable, "-m", "scripts.check_prepared_data", str(data / bench / "manifest.json")], check=True)
        else:
            from scripts.prepare_recall import check
            recall_data = check(data / bench / "manifest.json")
    running, failures = {}, []
    try:
        while jobs or running:
            free = [gpu for gpu in gpus if gpu not in {row[0] for row in running.values()}]
            while jobs and free:
                bench, arm = jobs.pop(0)
                dest = output / bench / arm
                log = None
                try:
                    if (dest / "result.json").exists():
                        check_arm(dest, bench, arm, recall_data=recall_data)
                        continue
                    gpu = free[0]
                    dest.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, SDM2_CUDA_EXTENSION_DIR=str(extensions),
                               SDM_SOURCE_ROOT=str(ROOT / "third_party/runtime"),
                               PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "third_party/runtime"))))
                    cmd = command(bench, arm, data / bench / "manifest.json", dest)
                    log = (dest / "train.log").open("ab")
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                except Exception as error:
                    if log is not None:
                        log.close()
                    failures.append(f"{bench}/{arm}: {error}")
                    print(json.dumps({"marker": "FAILED", "benchmark": bench, "arm": arm,
                                      "error": str(error)}), flush=True)
                    continue
                free.pop(0)
                running[proc] = (gpu, bench, arm, log, dest)
                print(json.dumps({"marker": "LAUNCHED", "benchmark": bench, "arm": arm, "gpu": gpu}), flush=True)
            for proc, (gpu, bench, arm, log, dest) in list(running.items()):
                status = proc.poll()
                if status is None:
                    continue
                log.close()
                del running[proc]
                try:
                    if status:
                        raise RuntimeError(f"exit {status}")
                    check_arm(dest, bench, arm, recall_data=recall_data)
                except Exception as error:
                    failures.append(f"{bench}/{arm}: {error}")
                print(json.dumps({"marker": "FINISHED", "benchmark": bench, "arm": arm, "exit": status}), flush=True)
            if running:
                time.sleep(1)
    finally:
        for proc, (_, _, _, log, _) in running.items():
            proc.terminate()
            proc.wait()
            log.close()
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
