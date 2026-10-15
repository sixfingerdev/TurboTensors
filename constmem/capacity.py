"""
What the constant memory costs: recall accuracy of the delta state vs number of stored
bindings M, with random unit keys and a codebook of V value codes (no learning involved).
A KV cache is exact at any M (it stores every pair); the state is not.

    python capacity.py [--dim 64] [--json out.json]
"""
import argparse, json
import numpy as np
from delta_state import DeltaState


def recall(D, M, V, trials, rng, beta=1.0, decay=1.0):
    ok = 0
    for _ in range(trials):
        codes = rng.standard_normal((V, D)).astype(np.float32)
        codes /= np.linalg.norm(codes, axis=-1, keepdims=True)
        keys = rng.standard_normal((M, D)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
        ans = rng.integers(0, V, M)
        st = DeltaState(1, D, beta=beta, decay=decay)
        dummy_q = np.zeros((1, D), np.float32); g = np.ones(1, np.float32)
        for i in range(M):
            st.step(dummy_q, keys[i][None], codes[ans[i]][None], g)
        # read every stored key (q^T S) and decode by nearest code
        reads = np.einsum('md,de->me', keys, st.S[0])
        ok += int((np.argmax(reads @ codes.T, axis=-1) == ans).sum())
    return ok / (trials * M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--codes", type=int, default=256)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = []
    print(f"D={a.dim}, value codebook={a.codes} (chance {100/a.codes:.1f}%)")
    for load in (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        M = max(1, int(load * a.dim))
        acc = recall(a.dim, M, a.codes, a.trials, rng)
        out.append(dict(load=load, M=M, acc=acc))
        print(f"M={M:4d} (load {load:5.3f} x D)  recall {100*acc:6.2f}%")
    if a.json:
        json.dump(dict(dim=a.dim, codes=a.codes, rows=out), open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
