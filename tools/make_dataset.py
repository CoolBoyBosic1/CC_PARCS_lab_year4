# -*- coding: utf-8 -*-
"""Builds the input file for PARCS: one document per line.
Optional header (first line) tunes the job: e.g. "#PARCS iters=10 tol=0 k=20".

  pip install scikit-learn      (only for the 20ng modes)
  python make_dataset.py 20ng      18000 docs_18k.txt            # real 20 Newsgroups (max ~18k docs)
  python make_dataset.py 20ng-aug  25000 docs_25k.txt            # bigger corpus built from 20 Newsgroups
  python make_dataset.py 20ng-aug  50000 docs_50k.txt
  python make_dataset.py synthetic 25000 docs_25k.txt            # no internet / no sklearn needed

Every command also writes <file>.labels (true class per line) for evaluate.py.
Default header: "#PARCS iters=10 tol=0" (fixed number of iterations -> comparable runs).
Use a different one with the 4th argument, e.g.:  ... docs.txt "#PARCS iters=15 tol=0 k=30"
"""
import random, re, sys

DEFAULT_HEADER = "#PARCS iters=10 tol=0"


def dump(docs, labels, path, header):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header + "\n")
        for d in docs:
            f.write(re.sub(r"\s+", " ", d).strip() + "\n")
    with open(path + ".labels", "w") as f:
        f.write("\n".join(str(x) for x in labels))
    print("written %d docs to %s (%d classes)" % (len(docs), path, len(set(labels))))


def load_20ng():
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    pairs = [(t, int(l)) for t, l in zip(d.data, d.target) if len(t.split()) > 20]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def augment(docs, labels, n, words=150, seed=1):
    """New documents: random words of two documents of the same class (class label is kept)."""
    rnd = random.Random(seed)
    by_class = {}
    for t, l in zip(docs, labels):
        by_class.setdefault(l, []).append(t.split())
    classes = sorted(by_class)
    out, lab = [], []
    for _ in range(n):
        c = rnd.choice(classes)
        a, b = rnd.choice(by_class[c]), rnd.choice(by_class[c])
        w = [x for x in a if rnd.random() < 0.6] + [x for x in b if rnd.random() < 0.6]
        rnd.shuffle(w)
        out.append(" ".join(w[:words]))
        lab.append(c)
    return out, lab


def synthetic_base(n=4000, topics=20):
    rnd = random.Random(7)
    L = lambda x: chr(97 + x // 26) + chr(97 + x % 26)
    pools = [["top" + L(t) + L(i) for i in range(80)] for t in range(topics)]
    common = ["com" + L(i) for i in range(300)]
    docs, labels = [], []
    for _ in range(n):
        t = rnd.randrange(topics)
        docs.append(" ".join(rnd.choice(pools[t]) if rnd.random() < 0.45 else rnd.choice(common) for _ in range(120)))
        labels.append(t)
    return docs, labels


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    path = sys.argv[3] if len(sys.argv) > 3 else "docs.txt"
    header = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_HEADER
    if kind == "20ng":
        d, l = load_20ng()
        idx = list(range(len(d))); random.Random(1).shuffle(idx); idx = idx[:n]
        dump([d[i] for i in idx], [l[i] for i in idx], path, header)
    elif kind == "20ng-aug":
        d, l = load_20ng()
        dump(*augment(d, l, n), path=path, header=header)
    else:
        d, l = synthetic_base()
        dump(*augment(d, l, n, words=150), path=path, header=header)
