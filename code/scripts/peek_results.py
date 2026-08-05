"""Quick partial-results aggregate for sanity-checking a run in progress."""
import glob
import json
from collections import defaultdict

acc = defaultdict(lambda: [0, 0])
cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
cite = defaultdict(lambda: [0, 0])
lat = defaultdict(list)
for f in glob.glob("results/main/qwen3.5-9b/*.json"):
    if f.endswith("summary.json"):
        continue
    r = json.load(open(f))
    c = r["condition"]
    acc[c][0] += int(r["correct"]); acc[c][1] += 1
    cat[c][r["category"]][0] += int(r["correct"]); cat[c][r["category"]][1] += 1
    cite[c][0] += r.get("citation_valid", 0); cite[c][1] += r.get("citation_total", 0)
    lat[c].append(r.get("elapsed_s", 0))

for c in ["zeroshot", "single_agent", "quantigence"]:
    if acc[c][1] == 0:
        continue
    a = acc[c][0] / acc[c][1]
    cv = (cite[c][0] / cite[c][1]) if cite[c][1] else None
    avlat = sum(lat[c]) / len(lat[c]) if lat[c] else 0
    cvs = "n/a" if cv is None else f"{cv:.0%}"
    print(f"{c:14s} n={acc[c][1]:2d} acc={a:.0%} cite_valid={cvs} "
          f"({cite[c][0]}/{cite[c][1]}) avg_lat={avlat:.0f}s")
    for k in ["standards", "algorithm", "vulnerability", "risk"]:
        v = cat[c][k]
        if v[1]:
            print(f"    {k:14s} {v[0]}/{v[1]}")
