"""Re-judge complex-task answers with an independent judge model.

Coverage is deterministic and unaffected; only the quality score is re-computed,
using a different model than the one under test to remove self-preference bias.
Reads/overwrites results/complex/*.json and rewrites summary.json.
Usage: python scripts/rejudge_complex.py --judge-url http://127.0.0.1:8081/v1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from quantigence.llm import LlamaClient  # noqa: E402
from eval.complex_eval import judge_quality  # noqa: E402
CONDS = ["zeroshot", "single_agent", "quantigence"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", default="http://127.0.0.1:8081/v1")
    args = ap.parse_args()
    judge = LlamaClient(base_url=args.judge_url)
    queries = {q["id"]: q for q in json.loads((CODE / "bench/complex.json").read_text())}

    recs = []
    for f in sorted(glob.glob(str(CODE / "results/complex/*.json"))):
        if f.endswith("summary.json"):
            continue
        d = json.loads(Path(f).read_text())
        q = queries[d["id"]]
        d["quality"] = round(judge_quality(judge, q["query"], d.get("answer", "")), 4)
        Path(f).write_text(json.dumps(d, indent=2))
        recs.append(d)
        print(f"[{d['condition']}/{d['id']}] coverage={d['coverage']:.0%} "
              f"quality(14B)={d['quality']:.2f}")

    summary = {}
    for c in CONDS:
        rs = [r for r in recs if r["condition"] == c]
        if not rs:
            continue
        n = len(rs)
        summary[c] = {
            "n": n,
            "mean_coverage": round(sum(r["coverage"] for r in rs) / n, 4),
            "mean_quality": round(sum(r["quality"] for r in rs) / n, 4),
            "avg_latency_s": round(sum(r.get("elapsed_s", 0) for r in rs) / n, 2),
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in rs) / n, 2),
        }
    (CODE / "results/complex/summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== RE-JUDGED SUMMARY (14B judge) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
