import json, glob, os, csv
from collections import defaultdict
W = {"stop_scrolling": 0.35, "share_likelihood": 0.25, "watch_to_end": 0.25, "comment_likelihood": 0.15}
def mix(dm, wave, mode):
    d = (dm.get(wave) or {}).get(mode)
    return sum(W[k] * d[k] for k in W if k in d) if d else None
rows, arch = [], {}
for f in sorted(glob.glob("tmp/result*.json")):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if not r.get("complete") or r.get("core_resonance") is None:
        continue
    dm = r.get("dimension_means") or {}
    on, off = mix(dm, "wave1", "sound_on"), mix(dm, "wave1", "sound_off")
    name = os.path.basename(f)[:-5]
    rows.append({"result": name, "niche": r.get("niche_category"), "core": round(r["core_resonance"], 2),
                 "crossover": round(r.get("crossover_score") or 0, 2), "tier": r.get("reach_tier"),
                 "w1_sound_on": None if on is None else round(on, 2),
                 "w1_sound_off": None if off is None else round(off, 2),
                 "sound_gap": None if on is None or off is None else round(on - off, 2)})
    g = defaultdict(list)
    for w in ("wave1", "wave2"):
        for p in r.get(w) or []:
            g[p["persona_id"].rsplit("_", 2)[0]].append(p["score"])
    arch[name] = {k: round(sum(v) / len(v), 2) for k, v in g.items()}
cols = ["result", "niche", "core", "crossover", "tier", "w1_sound_on", "w1_sound_off", "sound_gap"]
print(" | ".join(cols))
for x in rows:
    print(" | ".join(str(x[c]) for c in cols))
with open("tmp/analysis_summary.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=cols)
    wr.writeheader()
    wr.writerows(rows)
print("archetype means per result:")
for k, v in arch.items():
    print(k, v)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [x["result"] for x in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - 0.2 for i in range(len(rows))], [x["core"] for x in rows], 0.4, label="Core resonance")
    ax.bar([i + 0.2 for i in range(len(rows))], [x["crossover"] for x in rows], 0.4, label="Crossover")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score out of 10")
    ax.set_title("Core vs crossover per reel")
    ax.legend()
    fig.tight_layout()
    fig.savefig("tmp/analysis_core_vs_crossover.png", dpi=200)
    gaps = [(x["result"], x["sound_gap"]) for x in rows if x["sound_gap"] is not None]
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.bar([a for a, _ in gaps], [b for _, b in gaps])
    ax2.set_ylabel("Wave 1 score: sound on minus sound off")
    ax2.set_title("How much sound matters per reel")
    plt.setp(ax2.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    fig2.tight_layout()
    fig2.savefig("tmp/analysis_sound_gap.png", dpi=200)
    print("saved charts in tmp/")
except Exception as e:
    print("charts skipped:", e)
