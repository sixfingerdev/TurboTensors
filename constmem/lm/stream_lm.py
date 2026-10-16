"""
Constant-memory generation for a trained all-"G" model (models.py), in numpy + Numba on CPU.
Per layer the whole past is two small tensors: the delta-rule state (H, d, d) and the conv tail (3D, K-1).
Nothing grows with the number of tokens generated or read.

    python stream_lm.py CKPT TOKENIZER --prompt "Türkiye'nin başkenti" --n 80 --temp 0.8 --topk 40
"""
import argparse, os, sys, time
# Single-token matvecs are tiny: multi-threaded BLAS (and a thread pool in the kernel) made one step ~20x
# slower from oversubscription (40 ms vs 2 ms measured). Must be set before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def gdn_step(S, q, k, v, beta, alpha, o):
    """S:(H,d,d) [key x value] in place; q,k,v:(H,d) f32; beta,alpha:(H,); o:(H,d).
    S <- alpha (S - beta k (k^T S)) + beta k v^T ;  o = S^T q   (read after write, same as models.delta_sequential)"""
    H, d, _ = S.shape
    for h in range(H):
        pred = np.zeros(d, dtype=np.float32)
        for i in range(d):
            ki = k[h, i]
            for e in range(d):
                pred[e] += ki * S[h, i, e]
        b = beta[h]; a = alpha[h]
        for e in range(d):
            o[h, e] = 0.0
        for i in range(d):
            bk = b * k[h, i]; qi = q[h, i]
            for e in range(d):
                s = a * (S[h, i, e] - bk * pred[e]) + bk * v[h, e]
                S[h, i, e] = s
                o[h, e] += qi * s


def _silu(x): return x / (1.0 + np.exp(-x))
def _sig(x): return 1.0 / (1.0 + np.exp(-x))
def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class GDNStream:
    def __init__(self, ckpt):
        import torch
        ck = torch.load(ckpt, map_location="cpu"); c = ck["cfg"]; sd = {k: v.float().numpy() for k, v in ck["sd"].items()}
        assert set(c["layers"]) == {"G"}, "stream_lm supports all-G models (a hybrid needs a KV cache for its A layers)"
        self.c, self.sd = c, sd
        self.D, self.H = c["d"], c["heads"]; self.hd = self.D // self.H; self.K = c["conv"]
        self.L = len(c["layers"]); self.emb = sd["emb.weight"]
        self.reset()

    def reset(self):
        self.S = [np.zeros((self.H, self.hd, self.hd), np.float32) for _ in range(self.L)]
        self.conv = [np.zeros((3 * self.D, self.K - 1), np.float32) for _ in range(self.L)]
        self._o = np.empty((self.H, self.hd), np.float32)

    @property
    def state_bytes(self):
        return sum(s.nbytes for s in self.S) + sum(c.nbytes for c in self.conv)

    @staticmethod
    def _rms(x, w, eps=1e-6):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def step(self, tok):
        sd, D, H, hd = self.sd, self.D, self.H, self.hd
        x = self.emb[tok].astype(np.float32)
        for i in range(self.L):
            p = f"blocks.{i}."
            h = self._rms(x, sd[p + "n1.w"])
            qkv = sd[p + "mix.qkv.weight"] @ h                                   # (3D,)
            win = np.concatenate([self.conv[i], qkv[:, None]], 1)               # (3D, K)
            self.conv[i] = win[:, 1:].copy()
            qkv = _silu((win * sd[p + "mix.conv.weight"][:, 0, :]).sum(1))
            q, k, v = qkv.reshape(3, H, hd)
            q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12) * hd ** -0.5
            k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
            ab = sd[p + "mix.ab.weight"] @ h
            beta = _sig(ab[:H]).astype(np.float32)
            alpha = np.exp(-np.exp(sd[p + "mix.A_log"]) * _softplus(ab[H:] + sd[p + "mix.dt_bias"])).astype(np.float32)
            gdn_step(self.S[i], q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), beta, alpha, self._o)
            o = self._rms(self._o, sd[p + "mix.onorm.w"]) * _silu(sd[p + "mix.g.weight"] @ h).reshape(H, hd)
            x = x + sd[p + "mix.o.weight"] @ o.reshape(D)
            h = self._rms(x, sd[p + "n2.w"])
            x = x + sd[p + "mlp.w2.weight"] @ (_silu(sd[p + "mlp.w1.weight"] @ h) * (sd[p + "mlp.w3.weight"] @ h))
        return self.emb @ self._rms(x, sd["norm.w"])

    def generate(self, ids, n, temp=0.8, topk=40, rng=None, eos=None):
        rng = rng or np.random.default_rng(0); self.reset(); logits = None
        for t in ids: logits = self.step(int(t))
        out = []
        for _ in range(n):
            if temp <= 0: nxt = int(logits.argmax())
            else:
                l = logits / temp; idx = np.argpartition(l, -topk)[-topk:]; pr = np.exp(l[idx] - l[idx].max()); pr /= pr.sum()
                nxt = int(rng.choice(idx, p=pr))
            if eos is not None and nxt == eos: break
            out.append(nxt); logits = self.step(nxt)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tokenizer"); ap.add_argument("--prompt", default="Türkiye")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer); m = GDNStream(a.ckpt)
    ids = tok.encode(a.prompt).ids; t0 = time.time()
    out = m.generate(ids, a.n, a.temp, a.topk, np.random.default_rng(a.seed), eos=tok.token_to_id("<eos>"))
    dt = time.time() - t0
    print(a.prompt + tok.decode(out)); print(f"\n[{len(out)} tokens, {(len(ids)+len(out))/dt:.0f} tok/s, state {m.state_bytes/1024:.0f} KB]", file=sys.stderr)
