#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/third_party/runtime"
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  prepare)
    if [[ -f runs/data/wikitext/manifest.json ]]; then
      python -m scripts.check_prepared_data runs/data/wikitext/manifest.json
    else
      python -m scripts.prepare_canonical_wikitext103 --output runs/data/wikitext
    fi
    python -m scripts.prepare_recall --output runs/data/recall
    ;;
  build-extensions) python -m scripts.build_extensions "$@" ;;
  study|topology|headline) python -m scripts.run_experiments --suite "$action" "$@" ;;
  full) python -m scripts.run_experiments --suite full "$@" ;;
  all)
    ./reproduce.sh prepare
    ./reproduce.sh full "$@"
    ;;
  check) python -m scripts.check_reproduction "$@" ;;
  verify-results) python -m scripts.verify_results ;;
  test) python -m unittest discover -s tests -v ;;
  *)
    echo 'Usage: ./reproduce.sh {prepare|build-extensions|study|topology|full|all|check|verify-results|test}'
    echo 'Read REPRODUCING.md for hardware, binary preparation, runtime, cost, and metric ranges.'
    ;;
esac
