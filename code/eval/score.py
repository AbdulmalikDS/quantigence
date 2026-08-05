"""Deterministic scoring for benchmark answers, plus citation validity.

Accuracy is machine-checkable per query via one of four check types:
  keywords_all  - every substring must appear
  keywords_any  - each synonym group must be satisfied (AND of ORs)
  regex         - pattern must match
  qars          - the computed score must appear (numeric proximity), robust to
                  contamination from the input parameters echoed in the answer

Citation validity extracts arXiv ids and CVE ids the model *states* and checks
each against the live registries, yielding a hallucination rate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantigence.qars import Asset, qars, urgency_ratio  # noqa: E402
from quantigence import tools  # noqa: E402

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_CVE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


def _expected_qars(params: dict) -> tuple[float, bool]:
    a = Asset(name="q", **params)
    return qars(a), urgency_ratio(a) > 1.0


def _qars_match(answer: str, params: dict, tol: float = 0.02) -> bool:
    expected, _ = _expected_qars(params)
    inputs = {float(v) for v in params.values()}
    nums = [float(n) for n in _NUM.findall(answer)]
    for n in nums:
        if abs(n - expected) <= tol and all(abs(n - i) > 0.005 for i in inputs):
            return True
    return False


def check_answer(answer: str, check: dict) -> bool:
    a = answer.lower()
    kind = check.get("type")
    if kind == "keywords_all":
        return all(k.lower() in a for k in check["must"])
    if kind == "keywords_any":
        return all(any(syn.lower() in a for syn in group) for group in check["groups"])
    if kind == "regex":
        return re.search(check["pattern"], answer, re.I) is not None
    if kind == "qars":
        return _qars_match(answer, check["params"])
    raise ValueError(f"unknown check type: {kind}")


# --- citation validity ------------------------------------------------------

_cache: dict[str, bool] = {}


def _valid_cve(cid: str) -> bool:
    cid = cid.upper()
    if cid not in _cache:
        _cache[cid] = tools.cve_exists(cid)
    return _cache[cid]


def _valid_arxiv(aid: str) -> bool:
    if aid not in _cache:
        _cache[aid] = tools.arxiv_exists(aid)
    return _cache[aid]


def citation_stats(answer: str) -> dict:
    """Return {'total': n, 'valid': m} over CVE/arXiv ids stated in the answer."""
    cves = set(_CVE.findall(answer))
    arxivs = {a for a in _ARXIV.findall(answer)}
    total = len(cves) + len(arxivs)
    valid = sum(_valid_cve(c) for c in cves) + sum(_valid_arxiv(a) for a in arxivs)
    return {"total": total, "valid": valid,
            "cves": sorted(cves), "arxiv": sorted(arxivs)}


if __name__ == "__main__":
    assert check_answer("It affects OpenSSL and ML-KEM",
                        {"type": "keywords_all", "must": ["openssl", "ml-kem"]})
    assert check_answer("The answer is FIPS 205 / SLH-DSA",
                        {"type": "keywords_any", "groups": [["fips 205", "205"], ["slh-dsa"]]})
    # QARS: X=3,Y=25,Z=10,S=1,E=0.5 -> 0.9; must not be fooled by echoed inputs.
    p = {"migration_time": 3, "shelf_life": 25, "collapse_time": 10,
         "sensitivity": 1.0, "exploitability": 0.5}
    assert check_answer("Given X=3, Y=25, Z=10, the QARS is 0.90 (violation).",
                        {"type": "qars", "params": p})
    assert not check_answer("I could not compute it, but X=3 and Z=10.",
                            {"type": "qars", "params": p})
    print("score self-check passed")
