"""
Real trained models: time of ONE decode step and generation-state memory vs tokens already in context.
attn/hybrid: torch fp32 with a KV cache (filled with random K/V of the given length: timing only, not quality).
gdn: torch state, and the numpy+Numba stream engine (stream_lm.py).

    python bench_decode.py CKPT_DIR [--lens 256 4096 16384 65536 131072] [--json out.json]
"""
import argparse, json, os, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # numpy/BLAS for the stream engine; torch threads are set explicitly below
import numpy as np, torch
from models import Cfg, LM
from stream_lm import GDNStream

ap = argparse.ArgumentParser(); ap.add_argument("ckdir"); ap.add_argument("--lens", type=int, nargs="+", default=[256, 4096, 16384, 65536, 131072])
ap.add_argument("--json"); a = ap.parse_args()
torch.set_num_threads(4); torch.manual_seed(0)

def load(n):
    d = torch.load(f"{a.ckdir}/{n}.pt", map_location="cpu"); m = LM(Cfg(**d["cfg"])); m.load_state_dict(d["sd"]); return m.eval(), d["cfg"]

def med_ms(fn, reps):
    fn(); fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))

def nbytes(caches):
    return sum(t.numel() * t.element_size() for c in caches for t in c.values())

rows = []
models = {n: load(n) for n in ("attn", "hybrid", "gdn")}
stream = GDNStream(f"{a.ckdir}/gdn.pt")
tok = torch.tensor([[5]])
for N in a.lens:
    row = dict(N=N)
    for n, (m, c) in models.items():
        H, hd = c["heads"], c["d"] // c["heads"]
        caches = []
        for kind in c["layers"]:
            cc = {}
            if kind == "A":
                cc["k"] = torch.randn(1, H, N, hd); cc["v"] = torch.randn(1, H, N, hd)
            caches.append(cc)
        with torch.no_grad():
            m(tok, caches)                                           # create G states lazily (and append 1 to A caches)
            row[f"{n}_torch_ms"] = med_ms(lambda: m(tok, caches), 10 if N <= 16384 else 4)
        row[f"{n}_state_MB"] = nbytes(caches) / 2**20
        del caches
    row["gdn_numba_ms"] = med_ms(lambda: stream.step(5), 20); row["gdn_numba_state_MB"] = stream.state_bytes / 2**20
    rows.append(row)
    print(f"N={N:7d} | attn torch {row['attn_torch_ms']:7.2f} ms {row['attn_state_MB']:8.1f} MB | hybrid torch {row['hybrid_torch_ms']:7.2f} ms {row['hybrid_state_MB']:7.1f} MB"
          f" | gdn torch {row['gdn_torch_ms']:6.2f} ms {row['gdn_state_MB']:.2f} MB | gdn numpy+numba {row['gdn_numba_ms']:6.2f} ms", flush=True)
if a.json: json.dump(rows, open(a.json, "w"), indent=1)
