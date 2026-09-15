"""Check recorded arithmetic and provenance; this does not train a model."""
import hashlib
import json
import math
from pathlib import Path

from access_curriculum.schedule import ALIASES, CELLS, HEADLINE_ARMS, STUDY_ARMS, POLICIES

ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT):
    data = json.loads((root / "data/results.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    rows = data["rows"]
    assert data["primary_metrics"] == {"wikitext": "nll", "recall": "exact_set_accuracy"}
    assert data["headline_arms"] == list(HEADLINE_ARMS)
    assert data["primary_arms"] == list(STUDY_ARMS)
    assert (data["terminal_reads"], data["terminal_writes"], data["logical_rows_per_layer"]) == (8, 8, 1024)
    sources = {r["id"]: r for r in provenance["source_results"]}
    assert len(sources) == 32
    assert data["headline_endpoint_count"] == 2 * len(HEADLINE_ARMS) == 22
    for arm in (*CELLS, *HEADLINE_ARMS):
        for benchmark in ("wikitext", "recall"):
            row = rows[arm][benchmark]
            source = sources[row["source_id"]]
            assert len(source["source_sha256"]) == 64
            for split in ("validation", "test"):
                values = row[split]
                assert all(math.isfinite(v) for v in values.values())
                if benchmark == "wikitext":
                    assert values["scored_targets"] == (247416 if split == "validation" else 283426)
                else:
                    assert values["queries"] == 16 * values["examples"] == 983040
                    assert 0 <= values["exact_set_accuracy"] <= values["query_accuracy"] <= 1
    for alias, cell in ALIASES.items():
        assert rows[alias] == rows[cell]
    for benchmark, metric in (("wikitext", "nll"), ("recall", "query_loss")):
        losses = {c: rows[c][benchmark]["test"][metric] for c in CELLS}
        comparisons = {
            "staged_minus_direct": ("BDFH", "ACEG"),
            "rho4_minus_rho1": ("CDGH", "ABEF"),
            "asymmetry4_minus_asymmetry1": ("EFGH", "ABCD"),
        }
        key = "wikitext_test_nll" if benchmark == "wikitext" else "recall_test_loss"
        for effect, (plus, minus) in comparisons.items():
            delta = (sum(losses[c] for c in plus) - sum(losses[c] for c in minus)) / 4
            assert math.isclose(delta, data["average_main_effects"][effect][key], abs_tol=1e-12)
        assert all(losses[b] < losses[a] for a, b in zip("ACEG", "BDFH"))
        if benchmark == "wikitext":
            for arm in ALIASES:
                assert rows[arm][benchmark]["test"][metric] < min(rows[c][benchmark]["test"][metric] for c in ("dense_attention", "fixed_k8", "hybrid_fixed_k8"))
    for effect, (plus, minus) in comparisons.items():
        delta = 25 * (sum(rows[c]["recall"]["test"]["exact_set_accuracy"] for c in plus)
                      - sum(rows[c]["recall"]["test"]["exact_set_accuracy"] for c in minus))
        assert math.isclose(delta, data["average_main_effects"][effect]["recall_test_exact_set_accuracy_pp"], abs_tol=1e-12)
    for arm in ALIASES:
        assert rows[arm]["recall"]["test"]["exact_set_accuracy"] > max(rows[c]["recall"]["test"]["exact_set_accuracy"] for c in ("fixed_k8", "hybrid_fixed_k8"))
    for split in ("validation", "test"):
        assert min(CELLS, key=lambda c: rows[c]["wikitext"][split]["nll"]) == "H"
        assert max(CELLS, key=lambda c: rows[c]["recall"][split]["exact_set_accuracy"]) == "H"
        assert rows["H"]["wikitext"][split]["nll"] < rows["fixed_k8"]["wikitext"][split]["nll"]
        assert rows["H"]["recall"][split]["exact_set_accuracy"] > rows["fixed_k8"]["recall"][split]["exact_set_accuracy"]
        assert rows["G"]["recall"][split]["exact_set_accuracy"] < rows["C"]["recall"][split]["exact_set_accuracy"]
        assert rows["D"]["recall"][split]["exact_set_accuracy"] < rows["C"]["recall"][split]["exact_set_accuracy"]
    assert rows["dense_attention"]["recall"]["test"]["exact_set_accuracy"] > max(rows[a]["recall"]["test"]["exact_set_accuracy"] for a in ALIASES)
    for policy in POLICIES:
        base, hybrid = rows[policy], rows["hybrid_" + policy]
        delta = hybrid["recall"]["test"]["exact_set_accuracy"] - base["recall"]["test"]["exact_set_accuracy"]
        assert delta < 0 if policy in ("wiki_forward", "recall_forward") else delta > 0
        assert hybrid["wikitext"]["test"]["nll"] < base["wikitext"]["test"]["nll"]
    breakdown = json.loads((root / "data/recall-breakdown.json").read_text())["sources"]
    assert set(breakdown) == {sid for sid in sources if sid.startswith("recall:")}
    for row in rows.values():
        for split in ("validation", "test"):
            detail = breakdown[row["recall"]["source_id"]][split]
            conditions = detail["by_condition"]
            assert len({c["condition_id"] for c in conditions}) == 30
            assert all(c["examples"] == 2048 and 0 <= c["exact_sets"] <= c["examples"] for c in conditions)
            for c in conditions:
                assert c["exact_set_accuracy"] == c["exact_sets"] / c["examples"]
            total = sum(c["exact_sets"] for c in conditions) / 61440
            assert math.isclose(total, row["recall"][split]["exact_set_accuracy"], abs_tol=1e-12)
            for family, expected_count in (("pointer_chase", 16), ("span_recall", 8), ("overwrite_recall", 6)):
                selected = [c for c in conditions if c["family"] == family]
                assert len(selected) == expected_count
                family_row = detail["by_family"][family]
                assert family_row["examples"] == sum(c["examples"] for c in selected)
                assert family_row["exact_sets"] == sum(c["exact_sets"] for c in selected)
                assert family_row["exact_set_accuracy"] == family_row["exact_sets"] / family_row["examples"]
    inventory = json.loads((root / "data/artifact-manifest.json").read_text())
    for name, digest in inventory["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    return data


if __name__ == "__main__":
    validate()
    print("Recorded results, factorial effects, and artifact hashes verified; no experiment executed.")
