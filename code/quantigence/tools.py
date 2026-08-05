"""External knowledge tools: arXiv search and NVD CVE lookup.

Each tool returns plain dicts (JSON-serializable) so results drop straight into
the shared memory store. Network calls time out and retry once; a failure is
raised loudly rather than returned as an empty result, so the eval harness can
record it as a failure instead of silently scoring a blank answer.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_UA = {"User-Agent": "Quantigence/0.2 (research; mailto:aalquwayfili@ncai.gov.sa)"}


class ToolError(RuntimeError):
    """Raised when a tool call fails after retrying."""


def _get(url: str, params: dict[str, Any], timeout: float = 30.0, retries: int = 1):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_UA, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))  # NVD asks for backoff; be polite.
    raise ToolError(f"GET {url} failed after {retries + 1} attempts: {last}")


def arxiv_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search arXiv. Returns a list of {arxiv_id, title, authors, summary,
    published, pdf_url}."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = _get(ARXIV_API, params)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", ns):
        raw_id = entry.findtext("a:id", default="", namespaces=ns)
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        pdf_url = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        out.append({
            "arxiv_id": arxiv_id,
            "title": " ".join((entry.findtext("a:title", "", ns)).split()),
            "authors": [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
            "summary": " ".join((entry.findtext("a:summary", "", ns)).split()),
            "published": entry.findtext("a:published", "", ns),
            "pdf_url": pdf_url,
        })
    return out


def arxiv_exists(arxiv_id: str) -> bool:
    """True iff an arXiv id resolves to a real paper. Used for citation checking."""
    bare = arxiv_id.split("v")[0].strip()
    resp = _get(ARXIV_API, {"id_list": bare, "max_results": 1})
    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        # arXiv returns a placeholder entry with no real id for bad lookups.
        if entry.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1].split("v")[0] == bare:
            return True
    return False


def _cvss(metrics: dict[str, Any]) -> tuple[float | None, str]:
    """Pull the best available CVSS base score/severity from an NVD metrics block."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            data = metrics[key][0]["cvssData"]
            score = data.get("baseScore")
            sev = data.get("baseSeverity") or metrics[key][0].get("baseSeverity", "")
            return score, sev
    return None, ""


def nvd_search(keyword: str | None = None, cve_id: str | None = None,
               max_results: int = 5) -> list[dict[str, Any]]:
    """Look up CVEs by keyword or exact id. Returns {cve_id, description,
    cvss_score, severity, published}."""
    if cve_id:
        params: dict[str, Any] = {"cveId": cve_id.upper()}
    elif keyword:
        params = {"keywordSearch": keyword, "resultsPerPage": max_results}
    else:
        raise ValueError("nvd_search needs keyword or cve_id")
    data = _get(NVD_API, params, timeout=40.0).json()
    out: list[dict[str, Any]] = []
    for item in data.get("vulnerabilities", []):
        cve = item["cve"]
        desc = next((d["value"] for d in cve.get("descriptions", [])
                     if d.get("lang") == "en"), "")
        score, sev = _cvss(cve.get("metrics", {}))
        out.append({
            "cve_id": cve.get("id", ""),
            "description": desc,
            "cvss_score": score,
            "severity": sev,
            "published": cve.get("published", ""),
        })
    return out


def cve_exists(cve_id: str) -> bool:
    """True iff a CVE id is real. Used for citation checking."""
    try:
        return len(nvd_search(cve_id=cve_id)) > 0
    except ToolError:
        return False


if __name__ == "__main__":
    # Live self-check against both APIs.
    papers = arxiv_search("post-quantum cryptography lattice", max_results=3)
    assert papers and all(p["arxiv_id"] and p["title"] for p in papers), papers
    print(f"arxiv_search -> {len(papers)} hits; first: {papers[0]['arxiv_id']} "
          f"{papers[0]['title'][:60]}")
    assert arxiv_exists("2402.07867"), "PoisonedRAG arXiv id should resolve"
    assert not arxiv_exists("2402.99999"), "bogus arXiv id should not resolve"

    time.sleep(6)  # NVD rate limit without a key: ~5 req / 30s.
    cves = nvd_search(keyword="OpenSSL", max_results=3)
    assert cves and cves[0]["cve_id"].startswith("CVE-"), cves
    print(f"nvd_search  -> {len(cves)} hits; first: {cves[0]['cve_id']} "
          f"sev={cves[0]['severity']}")
    print("tools self-check passed")
