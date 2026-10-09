# -*- coding: utf-8 -*-
"""
PARCS-python: parallel text clustering (TF-IDF + spherical K-Means).

Scheme (Map-Reduce on every iteration):
  master: tokenization -> TF-IDF -> initial centroids
  loop:   MAP    - a worker gets its chunk of documents (sent once, cached on the
                   worker) + the current centroids, returns labels, counts,
                   inertia and partial per-cluster sums
          REDUCE - master adds the partial sums, normalizes them, gets new centroids
Compatible with Python 2.7 and 3.x, standard library only. ASCII only.

All bulk data is passed between nodes as flat strings (fast to (de)serialize with
Pyro4/serpent); floats are written with repr(), so the round trip is exact and the
serial and parallel runs give identical results.
"""
import io
import math
import random
import re
import time

try:
    from Pyro4 import expose
except ImportError:          # local run without Pyro4
    def expose(f):
        return f

K = 20               # number of clusters
MAX_ITER = 15        # max iterations
MAX_FEATURES = 1000  # vocabulary size
TOL = 1e-3           # relative inertia change for stopping
SEED = 42
RUN_SERIAL = True    # also run the serial version on master for comparison

STOP = set(("the and for are but not you all can had her was one our out has have this "
            "that with from they will would there their what about which when your said "
            "them these some into than then its also been were more other only").split())


class Solver:
    _cache = {}   # decoded chunks cached on the worker

    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        print("Inited")

    # ------------------------------------------------------------------ master
    def solve(self):
        global K, MAX_ITER, TOL, MAX_FEATURES, RUN_SERIAL
        print("Job Started")
        W = len(self.workers)
        print("Workers %d" % W)

        docs, cfg = self.read_input()
        # optional first line of the input file: "#PARCS k=20 iters=10 tol=0 features=1000 serial=1"
        if "k" in cfg:
            K = int(cfg["k"])
        if "iters" in cfg:
            MAX_ITER = int(cfg["iters"])
        if "tol" in cfg:
            TOL = float(cfg["tol"])
        if "features" in cfg:
            MAX_FEATURES = int(cfg["features"])
        if "serial" in cfg:
            RUN_SERIAL = bool(int(cfg["serial"]))
        print("Config: K=%d iters=%d tol=%g features=%d serial=%s" % (K, MAX_ITER, TOL, MAX_FEATURES, RUN_SERIAL))

        t0 = time.time()
        vecs, vocab = Solver.build_tfidf(docs, MAX_FEATURES)
        t_prep = time.time() - t0
        n, V = len(vecs), len(vocab)
        init = Solver.init_centroids(vecs, K, V, SEED)
        print("Docs %d, vocab %d, preprocessing %.3f s" % (n, V, t_prep))

        # ---- parallel version (chunk encoding and the first transfer are included in the time)
        workers = self.workers
        token = "%d_%d" % (int(time.time() * 1000), random.randrange(10 ** 9))
        stats = {"miss": 0}
        t0 = time.time()
        step = (n + W - 1) // W
        chunk_str = [Solver.encode_docs(vecs[i * step:(i + 1) * step]) for i in range(W)]
        sent = [False] * W
        nocache = [False] * W

        def par_parts(centroids):
            cstr = Solver.encode_floats(centroids)   # encoded once, shared by all workers
            promises = []
            for w in range(W):
                payload = chunk_str[w] if (nocache[w] or not sent[w]) else None
                promises.append(workers[w].assign_chunk(payload, token + "_%d" % w, cstr, K, V))
            res = [p.value for p in promises]
            for w in range(W):
                if res[w] == "MISS":              # worker lost its cache -> resend and stop relying on it
                    stats["miss"] += 1
                    nocache[w] = True
                    res[w] = workers[w].assign_chunk(chunk_str[w], token + "_%d" % w, cstr, K, V).value
                sent[w] = True
            return res

        lab_p, inr_p, it_p, cent_p, cnt_p, times_p = Solver.kmeans_loop(par_parts, init, V)
        t_par = time.time() - t0
        print("Parallel done: %.3f s, %d iterations" % (t_par, it_p))

        # ---- serial version (for correctness check and speedup); same code path, one chunk
        t_ser = None
        match = None
        times_s = []
        if RUN_SERIAL:
            t0 = time.time()
            whole = Solver.encode_docs(vecs)

            def ser_parts(centroids):
                cstr = Solver.encode_floats(centroids)
                return [Solver.assign_chunk(whole, "serial_" + token, cstr, K, V)]

            lab_s, inr_s, it_s, _, _, times_s = Solver.kmeans_loop(ser_parts, init, V)
            t_ser = time.time() - t0
            Solver._cache.pop("serial_" + token, None)
            same = sum(1 for a, b in zip(lab_p, lab_s) if a == b)
            match = (same == n and it_s == it_p and abs(inr_s - inr_p) <= 1e-6 * max(1.0, inr_s))
            print("Serial done: %.3f s, labels equal: %s" % (t_ser, match))

        self.write_output(n, V, W, it_p, t_prep, t_par, t_ser, inr_p, match, cnt_p, cent_p, vocab,
                          lab_p, times_p, times_s, stats["miss"], sum(len(c) for c in chunk_str))

    # ------------------------------------------------------------------ map (worker)
    @staticmethod
    @expose
    def assign_chunk(chunk, key, cent_str, k, V):
        """chunk: documents as a flat string (see encode_docs), or None if the chunk is
        already cached on the worker under key (otherwise returns "MISS").
        cent_str: k*V centroid values as a flat string.
        Returns [labels, counts, inertia, sums_string]."""
        if chunk is None:
            docs = Solver._cache.get(key)
            if docs is None:
                return "MISS"
        else:
            docs = Solver.decode_docs(chunk)
            Solver._cache[key] = docs
        flat = [float(x) for x in cent_str.split()]
        centroids = [flat[j * V:(j + 1) * V] for j in range(k)]
        cnorm = [sum(x * x for x in c) for c in centroids]
        sums = [[0.0] * V for _ in range(k)]
        counts = [0] * k
        labels = []
        inertia = 0.0
        for doc in docs:
            best, bd = 0, 1e18
            for j in range(k):
                c = centroids[j]
                dot = 0.0
                for i, v in doc:
                    dot += v * c[i]
                d = 1.0 - 2.0 * dot + cnorm[j]      # ||x|| = 1
                if d < bd:
                    bd, best = d, j
            labels.append(best)
            counts[best] += 1
            inertia += bd
            s = sums[best]
            for i, v in doc:
                s[i] += v
        return [labels, counts, inertia, Solver.encode_floats(sums)]

    # ------------------------------------------------------------------ reduce (master)
    @staticmethod
    def reduce_parts(parts, old, k, V):
        sums = [[0.0] * V for _ in range(k)]
        counts = [0] * k
        inertia = 0.0
        labels = []
        for lab, cnt, inr, sstr in parts:
            labels.extend(lab)
            inertia += inr
            flat = [float(x) for x in sstr.split()]
            for j in range(k):
                counts[j] += cnt[j]
                row = sums[j]
                base = j * V
                for i in range(V):
                    row[i] += flat[base + i]
        new = []
        for j in range(k):
            if counts[j] == 0:
                new.append(old[j])
                continue
            norm = math.sqrt(sum(x * x for x in sums[j])) or 1.0
            new.append([x / norm for x in sums[j]])
        return new, counts, inertia, labels

    @staticmethod
    def kmeans_loop(compute_parts, init, V):
        centroids, prev = init, None
        labels, counts, inertia, it = [], [], 0.0, 0
        times = []
        for it in range(1, MAX_ITER + 1):
            t = time.time()
            parts = compute_parts(centroids)
            centroids, counts, inertia, labels = Solver.reduce_parts(parts, centroids, K, V)
            times.append(time.time() - t)
            if TOL > 0 and prev is not None and abs(prev - inertia) <= TOL * prev:
                break
            prev = inertia
        return labels, inertia, it, centroids, counts, times

    # ------------------------------------------------------------------ transport helpers
    @staticmethod
    def encode_floats(rows):
        return " ".join(" ".join(repr(x) for x in row) for row in rows)

    @staticmethod
    def encode_docs(docs):
        return "\n".join(" ".join("%d:%s" % (i, repr(v)) for i, v in d) for d in docs)

    @staticmethod
    def decode_docs(text):
        out = []
        for line in text.split("\n"):
            d = []
            for tok in line.split():
                a, b = tok.split(":")
                d.append((int(a), float(b)))
            out.append(d)
        return out

    # ------------------------------------------------------------------ NLP
    @staticmethod
    def tokenize(text):
        return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP]

    @staticmethod
    def build_tfidf(docs, max_features):
        n = len(docs)
        toks = [Solver.tokenize(d) for d in docs]
        df = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        cand = [(c, w) for w, c in df.items() if c >= 2 and c <= 0.5 * n]
        cand.sort(key=lambda x: (-x[0], x[1]))
        vocab = [w for _, w in cand[:max_features]]
        index = dict((w, i) for i, w in enumerate(vocab))
        idf = [math.log((1.0 + n) / (1.0 + df[w])) + 1.0 for w in vocab]
        vecs = []
        for t in toks:
            tf = {}
            for w in t:
                i = index.get(w)
                if i is not None:
                    tf[i] = tf.get(i, 0) + 1
            vec = sorted([i, (1.0 + math.log(c)) * idf[i]] for i, c in tf.items())
            norm = math.sqrt(sum(v * v for _, v in vec)) or 1.0
            # values are rounded to 6 significant digits: smaller transfer, identical in serial and parallel
            vecs.append([(i, float("%.6g" % (v / norm))) for i, v in vec])
        return vecs, vocab

    @staticmethod
    def init_centroids(vecs, k, V, seed):
        rnd = random.Random(seed)
        cents = []
        for idx in rnd.sample(range(len(vecs)), k):
            row = [0.0] * V
            for i, v in vecs[idx]:
                row[i] = v
            cents.append(row)
        return cents

    # ------------------------------------------------------------------ I/O
    def read_input(self):
        cfg = {}
        docs = []
        with io.open(self.input_file_name, 'r', encoding='utf-8', errors='ignore') as f:
            for n, line in enumerate(f):
                line = line.strip()
                if n == 0 and line.startswith("#PARCS"):
                    for tok in line.split()[1:]:
                        if "=" in tok:
                            a, b = tok.split("=", 1)
                            cfg[a] = b
                    continue
                if line:
                    docs.append(line)
        return docs, cfg

    def write_output(self, n, V, W, iters, t_prep, t_par, t_ser, inertia, match, counts, cents, vocab,
                     labels, times_p, times_s, misses, sent_chars):
        f = open(self.output_file_name, 'w')
        f.write("docs: %d\nvocab: %d\nclusters: %d\nworkers: %d\niterations: %d\n" % (n, V, K, W, iters))
        f.write("t_prep: %.4f\nt_parallel: %.4f\n" % (t_prep, t_par))
        if t_ser is not None:
            f.write("t_serial: %.4f\nspeedup: %.3f\nefficiency: %.3f\nserial_equals_parallel: %s\n"
                    % (t_ser, t_ser / t_par, t_ser / t_par / W, match))
        f.write("par_iter_first: %.4f\n" % times_p[0])
        if len(times_p) > 1:
            f.write("par_iter_avg_rest: %.4f\n" % (sum(times_p[1:]) / (len(times_p) - 1)))
        if times_s:
            f.write("ser_iter_avg: %.4f\n" % (sum(times_s) / len(times_s)))
        f.write("cache_misses: %d\nchunk_chars_sent: %d\n" % (misses, sent_chars))
        f.write("inertia: %.6f\ncluster_sizes: %s\n" % (inertia, " ".join(str(c) for c in counts)))
        for j in range(K):
            top = sorted(range(V), key=lambda i: -cents[j][i])[:8]
            f.write("cluster %d (%d): %s\n" % (j, counts[j], ", ".join(vocab[i] for i in top)))
        f.write("LABELS: %s\n" % " ".join(str(x) for x in labels))
        f.close()
        print("output done")
