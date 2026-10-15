"""
Per-token decode cost and memory vs context length: KV-cache attention (TurboTensors'
own kernels) vs the constant-size delta state. One layer, H heads, head dim D, B=1.

    python bench_context.py [--heads 10] [--dim 64] [--json out.json]
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import fast_attention_decode, fast_attention_decode_numpy  # noqa: E402
from delta_state import DeltaState  # noqa: E402


def best_ms(fn, reps):
    fn(); fn()                                   # warm (JIT + caches)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lens", type=int, nargs="+",
                    default=[64, 256, 1024, 4096, 16384, 32768])
    ap.add_argument("--json")
    a = ap.parse_args()
    H, D = a.heads, a.dim
    rng = np.random.default_rng(0)
    scale = np.float32(1 / np.sqrt(D))

    st = DeltaState(H, D)
    q = rng.standard_normal((H, D)).astype(np.float32)
    k = rng.standard_normal((H, D)).astype(np.float32); k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((H, D)).astype(np.float32)
    g = np.full(H, 0.5, np.float32)
    delta_ms = best_ms(lambda: st.step(q, k, v, g), 200)

    rows = []
    for N in a.lens:
        kc = rng.standard_normal((1, H, N, D)).astype(np.float32)
        vc = rng.standard_normal((1, H, N, D)).astype(np.float32)
        q4 = q[None, :, None, :].copy()
        reps = 50 if N <= 4096 else 15
        np_ms = best_ms(lambda: fast_attention_decode_numpy(q4, kc, vc, scale, N), reps)
        nb_ms = best_ms(lambda: fast_attention_decode(q4, kc, vc, scale, N), reps)
        row = dict(N=N, kv_numpy_ms=np_ms, kv_numba_ms=nb_ms, delta_ms=delta_ms,
                   kv_cache_MB=(kc.nbytes + vc.nbytes) / 2**20, state_MB=st.nbytes / 2**20)
        rows.append(row)
        print(f"N={N:6d}  KV(numpy) {np_ms:8.3f} ms  KV(numba) {nb_ms:8.3f} ms  "
              f"delta {delta_ms:6.3f} ms  |  cache {row['kv_cache_MB']:8.2f} MB  state {row['state_MB']:.3f} MB")
        del kc, vc
    if a.json:
        json.dump(dict(heads=H, dim=D, cpus=os.cpu_count(), rows=rows), open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
