# -*- coding: utf-8 -*-
"""Локальний запуск без хмари: імітує воркерів PARCS.
   python local_test.py [docs.txt] [workers]"""
import random
import sys
import time
from kmeans_solver import Solver


class Promise(object):
    def __init__(self, v):
        self.value = v


try:
    import serpent
    def wire(x):                      # imitate the Pyro4 serializer
        return serpent.loads(serpent.dumps(x))
except ImportError:
    def wire(x):
        return x


class FakeWorker(object):
    def assign_chunk(self, chunk, key, cent_str, k, V):
        a = wire([chunk, key, cent_str, k, V])
        return Promise(wire(Solver.assign_chunk(*a)))


def make_synthetic(path, n=3000, topics=8, seed=1):
    rnd = random.Random(seed)
    L = lambda x: chr(97 + x // 26) + chr(97 + x % 26)
    pools = [["top" + L(t) + L(i) for i in range(40)] for t in range(topics)]
    common = ["com" + L(i) for i in range(60)]
    labels = []
    with open(path, "w") as f:
        for _ in range(n):
            t = rnd.randrange(topics)
            words = [rnd.choice(pools[t]) if rnd.random() < 0.6 else rnd.choice(common)
                     for _ in range(80)]
            f.write(" ".join(words) + "\n")
            labels.append(t)
    with open(path + ".labels", "w") as f:
        f.write("\n".join(str(x) for x in labels))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic.txt"
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if len(sys.argv) <= 1:
        make_synthetic(path)
    t0 = time.time()
    s = Solver([FakeWorker() for _ in range(W)], path, "out_local.txt")
    s.solve()
    print("total %.2f s" % (time.time() - t0))
    print(open("out_local.txt").read().split("LABELS")[0])
