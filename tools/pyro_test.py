# Real Pyro4 test: workers are separate processes with a Pyro4 daemon (like PARCS nodes).
import multiprocessing as mp, sys, time
import Pyro4
from concurrent.futures import ThreadPoolExecutor
import kmeans_solver as ks

def worker_proc(q):
    d = Pyro4.Daemon(host="127.0.0.1")
    uri = d.register(ks.Solver())          # same as node.py: Solver() instance exposed
    q.put(str(uri)); d.requestLoop()

class Call(object):
    pool = ThreadPoolExecutor(16)
class Proxy(object):
    def __init__(self, uri): self.uri = uri
    def assign_chunk(self, *a):
        def run():
            with Pyro4.Proxy(self.uri) as p:
                return p.assign_chunk(*a)
        f = Call.pool.submit(run)
        class P(object):
            @property
            def value(s): return f.result()
        return P()

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    q = mp.Queue(); ps = []
    for _ in range(W):
        p = mp.Process(target=worker_proc, args=(q,), daemon=True); p.start(); ps.append(p)
    uris = [q.get() for _ in range(W)]
    s = ks.Solver([Proxy(u) for u in uris], "docs.txt", "out_pyro.txt")
    s.solve()
    print(open("out_pyro.txt").read().split("cluster 0")[0])
