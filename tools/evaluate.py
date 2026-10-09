# -*- coding: utf-8 -*-
"""Якість кластеризації: purity і NMI (чистий Python).
   python evaluate.py output.txt docs.txt.labels"""
import math, sys
from collections import Counter

out = open(sys.argv[1]).read()
pred = [int(x) for x in [l for l in out.splitlines() if l.startswith("LABELS:")][0][7:].split()]
true = [int(x) for x in open(sys.argv[2]).read().split()]
n = len(pred); assert n == len(true), (n, len(true))
cont = Counter(zip(pred, true)); cp = Counter(pred); ct = Counter(true)
purity = sum(max(v for (p, t), v in cont.items() if p == c) for c in cp) / float(n)
mi = sum(v / float(n) * math.log(v * n / float(cp[p] * ct[t])) for (p, t), v in cont.items())
hp = -sum(v / float(n) * math.log(v / float(n)) for v in cp.values())
ht = -sum(v / float(n) * math.log(v / float(n)) for v in ct.values())
print("docs=%d purity=%.4f NMI=%.4f" % (n, purity, mi / math.sqrt(hp * ht) if hp * ht else 0))
