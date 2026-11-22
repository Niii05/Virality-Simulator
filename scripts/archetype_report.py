import json, sys
from collections import defaultdict

for n in sys.argv[1:]:
    r = json.load(open(f"tmp/result{n}.json", encoding="utf-8"))
    g = defaultdict(list)
    for w in ("wave1", "wave2"):
        for p in r.get(w) or []:
            g[p["persona_id"].rsplit("_", 2)[0]].append(p["score"])
    print(n, {k: round(sum(v) / len(v), 2) for k, v in sorted(g.items())})
    print("  count", {k: len(v) for k, v in sorted(g.items())})
