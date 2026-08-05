"""Complex-task evaluation: the fair test for multi-agent orchestration.

Unlike the atomic benchmark, each query here is multi-faceted and requires
several domains at once (crypto, threat, standards, risk). We score two things:

  coverage - fraction of a per-query rubric of required sub-points that the
             answer correctly addresses (deterministic, via the atomic scorer).
  quality  - an LLM judge's 1-5 rating of comprehensiveness and correctness,
             normalized to [0, 1].

This is where decomposition should help: a single agent must cover every facet in
one pass, while the multi-agent system assigns facets to specialists and
synthesizes. We compare zero-shot, single-agent, and Quantigence.

Usage: python -m eval.complex_eval --conditions single_agent,quantigence \\
           --base-url http://127.0.0.1:8080/v1 --judge-url http://127.0.0.1:8081/v1 \\
           --out results/complex
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from quantigence.corpus import load as load_corpus
from quantigence.llm import LlamaClient
from quantigence.registry import Defenses, Registry
from quantigence import orchestrator as O
from eval import score as S

BENCH = Path(__file__).resolve().parents[1] / "bench" / "complex.json"

QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "comprehensiveness": {"type": "integer", "minimum": 1, "maximum": 5},
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["comprehensiveness", "correctness"],
    "additionalProperties": False,
}


def coverage(answer: str, rubric: list[dict]) -> tuple[float, list[str]]:
    hit = []
    for pt in rubric:
        if S.check_answer(answer, pt["check"]):
            hit.append(pt["point"])
    return len(hit) / len(rubric), hit


def judge_quality(judge: LlamaClient, query: str, answer: str) -> float:
    msgs = [
        {"role": "system", "content": "You are a strict technical reviewer of "
         "post-quantum security analyses. Rate the answer, not its length."},
        {"role": "user", "content":
            f"QUESTION:\n{query}\n\nANSWER:\n{answer[:3500]}\n\nRate 1-5 for "
            "comprehensiveness (are all parts of the question addressed?) and "
            "correctness (are the technical claims accurate?)."},
    ]
    try:
        v = judge.complete_json(msgs, QUALITY_SCHEMA)
        return (v["comprehensiveness"] + v["correctness"]) / 10.0
    except Exception:
        return 0.0


def run_condition(client, judge, corpus, condition, q) -> dict:
    reg = Registry(corpus, Defenses())
    t0 = time.time()
    try:
        if condition == "zeroshot":
            res = O.run_zeroshot(client, q["query"])
        elif condition == "single_agent":
            res = O.run_single_agent(client, q["query"], reg)
        else:
            res = O.run_quantigence(client, q["query"], reg)
        answer, tool_calls, llm_calls = res.answer, res.tool_calls, res.llm_calls
    except Exception as e:
        return {"id": q["id"], "condition": condition, "error": repr(e),
                "coverage": 0.0, "quality": 0.0, "answer": ""}
    cov, hits = coverage(answer, q["rubric"])
    qual = judge_quality(judge, q["query"], answer)
    return {"id": q["id"], "condition": condition, "coverage": round(cov, 4),
            "hits": hits, "n_rubric": len(q["rubric"]), "quality": round(qual, 4),
            "tool_calls": tool_calls, "llm_calls": llm_calls,
            "elapsed_s": round(time.time() - t0, 2), "answer": answer}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="zeroshot,single_agent,quantigence")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--out", default="results/complex")
    args = ap.parse_args()

    queries = json.loads(BENCH.read_text())
    client = LlamaClient(base_url=args.base_url)
    judge = LlamaClient(base_url=args.judge_url or args.base_url)
    corpus = load_corpus()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    records = []
    for condition in args.conditions.split(","):
        for q in queries:
            fp = out / f"{condition}__{q['id']}.json"
            if fp.exists():
                rec = json.loads(fp.read_text())
            else:
                rec = run_condition(client, judge, corpus, condition, q)
                fp.write_text(json.dumps(rec, indent=2))
            records.append(rec)
            print(f"[{condition}/{q['id']}] coverage={rec['coverage']:.0%} "
                  f"quality={rec['quality']:.2f} ({rec.get('elapsed_s', 0)}s)")

    summary = {}
    for c in args.conditions.split(","):
        rs = [r for r in records if r["condition"] == c]
        n = len(rs)
        summary[c] = {
            "n": n,
            "mean_coverage": round(sum(r["coverage"] for r in rs) / n, 4),
            "mean_quality": round(sum(r["quality"] for r in rs) / n, 4),
            "avg_latency_s": round(sum(r.get("elapsed_s", 0) for r in rs) / n, 2),
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in rs) / n, 2),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== COMPLEX-TASK SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
