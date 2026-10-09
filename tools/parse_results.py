# -*- coding: utf-8 -*-
"""Collects PARCS outputs (out_*.txt) into a Markdown table (+ speedup.png).
   python parse_results.py out_25k_w1.txt out_25k_w2.txt out_25k_w4.txt ...
Rows are grouped by number of documents, sorted by workers."""
import sys

rows = []
for p in sys.argv[1:]:
    d = {"file": p}
    for l in open(p, errors="ignore"):
        if l.startswith("LABELS") or l.startswith("cluster "):
            continue
        if ":" in l:
            k, v = l.split(":", 1)
            d[k.strip()] = v.strip()
    rows.append(d)
rows.sort(key=lambda d: (int(d["docs"]), int(d["workers"])))

f = lambda d, k: float(d[k]) if k in d else float("nan")
print("| Docs | K | Iter | Workers | T_prep, s | T_serial, s | T_parallel, s | Speedup | Efficiency | End-to-end speedup | 1st iter, s | Next iter, s | Match |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in rows:
    e2e = (f(d, "t_prep") + f(d, "t_serial")) / (f(d, "t_prep") + f(d, "t_parallel"))
    print("| %s | %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |" % (
        d["docs"], d["clusters"], d["iterations"], d["workers"], f(d, "t_prep"), f(d, "t_serial"),
        f(d, "t_parallel"), f(d, "speedup"), f(d, "efficiency"), e2e,
        f(d, "par_iter_first"), f(d, "par_iter_avg_rest"), d.get("serial_equals_parallel", "?")))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for n in sorted(set(int(d["docs"]) for d in rows)):
        sub = [d for d in rows if int(d["docs"]) == n]
        plt.plot([int(d["workers"]) for d in sub], [f(d, "speedup") for d in sub], "o-", label="%d docs" % n)
    mw = max(int(d["workers"]) for d in rows)
    plt.plot([1, mw], [1, mw], "k--", label="ideal")
    plt.xlabel("Workers"); plt.ylabel("Speedup (T_serial / T_parallel)"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("speedup.png", dpi=150); print("\nsaved speedup.png")
except ImportError:
    pass
