"""
Streaming (token-by-token) inference for a trained Tessera COMPASS model, on numpy + the
Numba delta kernel. Memory per layer is constant: one (H, Ds, Ds) state plus a ring buffer of
the last `ctx_w` layer inputs. Nothing grows with context length.

Ports the configuration used by `compass.py facts*` (k_prod=True, q_pooled=True), faithfully
including two quirks of the torch code, found while porting (see README.md):
  * router(): the bias term is overwritten for valid offsets, so the pool is purely
    salience-based over the last ctx_w tokens;
  * prod_pools(): the per-token salience adds the same constant to every offset, which the
    softmax cancels, so the weights depend only on the fixed `prod_rb` recency bias.
"""
import numpy as np
from delta_state import DeltaState


def _ln(x, w, b, eps=1e-5):
    m = x.mean(); v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + eps) * w + b


def _gelu(x):  # exact (erf) GELU, same as nn.GELU()
    from math import erf, sqrt
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def _unit(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class StreamLayer:
    def __init__(self, sd, prefix, H, Ds, ctx_w):
        g = lambda n: sd[prefix + n].detach().cpu().numpy().astype(np.float64)
        self.H, self.Ds, self.w = H, Ds, ctx_w
        self.d = g("Wc.weight").shape[0]
        self.Wc, self.bc = g("Wc.weight"), g("Wc.bias")
        self.pool_u = g("pool_u")
        self.RA, self.RB = g("RA"), g("RB")
        self.prod_rb = g("prod_rb")[:ctx_w]
        self.Wv = g("Wv.weight")
        self.Wg, self.bg = g("Wg.weight"), g("Wg.bias")
        beta = 2 * _sig(g("beta")); decay = _sig(g("logit_decay"))
        self.Wo = g("Wo.weight")
        self.Wgo, self.bgo = g("gate_out.weight"), g("gate_out.bias")
        self.f1, self.fb1 = g("ffn.0.weight"), g("ffn.0.bias")
        self.f2, self.fb2 = g("ffn.2.weight"), g("ffn.2.bias")
        self.ln1 = (g("ln1.weight"), g("ln1.bias")); self.ln2 = (g("ln2.weight"), g("ln2.bias"))
        self.state = DeltaState(H, Ds, beta=beta.astype(np.float32), decay=decay.astype(np.float32))
        self.reset()

    def reset(self):
        self.state.reset()
        self.hist = np.zeros((self.w, self.d))      # hist[o] = input o tokens ago (o=0: current)
        self.nvalid = 0

    @property
    def nbytes(self):
        return self.state.nbytes + self.hist.nbytes

    def step(self, x):
        self.hist = np.roll(self.hist, 1, axis=0); self.hist[0] = x
        self.nvalid = min(self.nvalid + 1, self.w)
        w = self.w
        # router (salience pool, used for the write gate): invalid offsets sit at -1e4
        s = self.hist @ self.pool_u
        sc = np.full(w, -1e4); sc[:self.nvalid] = s[:self.nvalid]
        A = _softmax(sc)
        ctx = A @ self.hist
        r = x + self.Wc @ ctx + self.bc
        # product-binding pools: weights from the fixed recency bias only
        Ap = _softmax(self.prod_rb)                  # same for A and B pools (constant cancels)
        PA = PB = Ap @ self.hist
        key = _unit(_unit(PA @ self.RA) * _unit(PB @ self.RB))
        H, Ds = self.H, self.Ds
        k = np.tile(key, (H, 1)).astype(np.float32) if H == 1 else key.reshape(H, Ds).astype(np.float32)
        V = (self.Wv @ x).reshape(H, Ds).astype(np.float32)
        gate = _sig(self.Wg @ r + self.bg).astype(np.float32).reshape(H)
        y = self.state.step(k, k, V, gate)           # q == k (meet in the middle)
        y = self.Wo @ y.reshape(-1).astype(np.float64)
        x = _ln(x + _sig(self.Wgo @ x + self.bgo)[0] * y, *self.ln1)
        h = _gelu(self.f1 @ x + self.fb1)
        return _ln(x + self.f2 @ h + self.fb2, *self.ln2)


class StreamModel:
    def __init__(self, sd, layers, H, Ds, ctx_w=24):
        g = lambda n: sd[n].detach().cpu().numpy().astype(np.float64)
        self.emb = g("emb.weight"); self.d = self.emb.shape[1]
        self.Wout, self.bout = g("out.weight"), g("out.bias")
        self.layers = [StreamLayer(sd, f"layers.{i}.", H, Ds, ctx_w) for i in range(layers)]

    def reset(self):
        for l in self.layers: l.reset()

    @property
    def nbytes(self):
        return sum(l.nbytes for l in self.layers)

    def step(self, tok):
        x = self.emb[tok] * np.sqrt(self.d)
        for l in self.layers:
            x = l.step(x)
        return self.Wout @ x + self.bout
