"""Benchmark harness: run conditions x queries, score, and aggregate.

Resumable: each (condition, query) result is written to its own JSON file, so a
crashed or interrupted run continues instead of restarting. A tool/model failure
on one query is recorded as a failure (correct=False, error set) rather than
aborting the run.

Usage:
  python -m eval.harness --conditions zeroshot,single_agent,quantigence \\
      --model-tag qwen3.5-9b --out results/main [--limit N] [--categories risk,std]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from quantigence.corpus import load as load_corpus
from quantigence.llm import LlamaClient
from quantigence.registry import Defenses, Registry
from quantigence import orchestrator as O
from eval import score as S

BENCH = Path(__file__).resolve().parents[1] / "bench" / "queries.json"


def peak_vram_mb(gpu: int = 0) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def run_one(client, corpus, condition: str, q: dict, defenses: Defenses) -> dict:
    reg = Registry(corpus, defenses)
    fn = O.CONDITIONS[condition]
    try:
        if condition == "zeroshot":
            res = fn(client, q["query"])
        else:
            res = fn(client, q["query"], reg)
        error = None
    except Exception as e:
        return {"id": q["id"], "category": q["category"], "condition": condition,
                "correct": False, "error": repr(e), "answer": ""}

    correct = S.check_answer(res.answer, q["check"])
    cites = S.citation_stats(res.answer)
    return {
        "id": q["id"], "category": q["category"], "condition": condition,
        "correct": bool(correct), "error": None,
        "citation_total": cites["total"], "citation_valid": cites["valid"],
        "n_sources": len(set(res.sources)), "tool_calls": res.tool_calls,
        "llm_calls": res.llm_calls, "elapsed_s": round(res.elapsed_s, 2),
        "vram_mb": peak_vram_mb(),
        "answer": res.answer, "plan": res.plan,
    }


def aggregate(records: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for r in records:
        c = r["condition"]
        b = by.setdefault(c, {"n": 0, "correct": 0, "cite_total": 0, "cite_valid": 0,
                              "elapsed": 0.0, "tool_calls": 0, "vram": []})
        b["n"] += 1
        b["correct"] += int(r["correct"])
        b["cite_total"] += r.get("citation_total", 0)
        b["cite_valid"] += r.get("citation_valid", 0)
        b["elapsed"] += r.get("elapsed_s", 0.0)
        b["tool_calls"] += r.get("tool_calls", 0)
        if r.get("vram_mb", -1) > 0:
            b["vram"].append(r["vram_mb"])
    summary = {}
    for c, b in by.items():
        summary[c] = {
            "n": b["n"],
            "accuracy": round(b["correct"] / b["n"], 4) if b["n"] else 0,
            "citation_validity": round(b["cite_valid"] / b["cite_total"], 4)
                                 if b["cite_total"] else None,
            "citations_stated": b["cite_total"],
            "avg_latency_s": round(b["elapsed"] / b["n"], 2) if b["n"] else 0,
            "avg_tool_calls": round(b["tool_calls"] / b["n"], 2) if b["n"] else 0,
            "peak_vram_mb": max(b["vram"]) if b["vram"] else None,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="zeroshot,single_agent,quantigence")
    ap.add_argument("--model-tag", default="qwen3.5-9b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--categories", default="")
    ap.add_argument("--defenses", default="", help="comma list: sanitize,source_hierarchy,consensus")
    args = ap.parse_args()

    queries = json.loads(BENCH.read_text())
    if args.categories:
        cats = set(args.categories.split(","))
        queries = [q for q in queries if q["category"] in cats]
    if args.limit:
        queries = queries[:args.limit]

    dset = set(args.defenses.split(",")) if args.defenses else set()
    defenses = Defenses("sanitize" in dset, "source_hierarchy" in dset, "consensus" in dset)

    out = Path(args.out) / args.model_tag
    out.mkdir(parents=True, exist_ok=True)
    client = LlamaClient(base_url=args.base_url)
    corpus = load_corpus()

    records: list[dict] = []
    conditions = args.conditions.split(",")
    total = len(conditions) * len(queries)
    done = 0
    t0 = time.time()
    for condition in conditions:
        for q in queries:
            done += 1
            fpath = out / f"{condition}__{q['id']}.json"
            if fpath.exists():
                records.append(json.loads(fpath.read_text()))
                print(f"[{done}/{total}] cached {condition}/{q['id']}")
                continue
            rec = run_one(client, corpus, condition, q, defenses)
            fpath.write_text(json.dumps(rec, indent=2))
            records.append(rec)
            flag = "ok" if rec["correct"] else ("ERR" if rec["error"] else "x")
            print(f"[{done}/{total}] {condition}/{q['id']} {flag} "
                  f"({rec.get('elapsed_s', 0)}s)  [{time.time()-t0:.0f}s elapsed]")

    summary = aggregate(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
