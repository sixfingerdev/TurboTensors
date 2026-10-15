"""
Constant-memory decode kernels: the COMPASS delta-rule state, on TurboTensors' stack.

Source recurrence (tessera/compass.py, CompassLayer.forward), per head:

    y_t  = q_t^T S_{t-1}                              read BEFORE write
    pred = k_t^T S_{t-1}
    S_t  = decay * S_{t-1} + g_t * beta * k_t (v_t - pred)^T

S is a fixed (Ds x Dv) matrix per head: memory does not grow with context,
and one decode step costs O(H * Ds * Dv) regardless of how many tokens came before.
A KV-cache attention step costs O(H * N * Dh) and its cache grows with N.

This file is only the inference-side recurrence. The router/key/query projections that
produce q, k, v, g are not ported here (see README.md for what is and is not claimed).
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, nogil=True)
def delta_step(S, q, k, v, g, beta, decay, y):
    """One token, in place. S:(H,Ds,Dv) q,k:(H,Ds) v:(H,Dv) g,beta,decay:(H,) y:(H,Dv)."""
    H, Ds, Dv = S.shape
    for h in prange(H):
        # read and prediction both use S_{t-1}, one pass over S
        for e in range(Dv):
            y[h, e] = 0.0
        pred = np.zeros(Dv, dtype=np.float32)
        for d in range(Ds):
            qd = q[h, d]
            kd = k[h, d]
            for e in range(Dv):
                s = S[h, d, e]
                y[h, e] += qd * s
                pred[e] += kd * s
        coef = g[h] * beta[h]
        dec = decay[h]
        for d in range(Ds):
            ck = coef * k[h, d]
            for e in range(Dv):
                S[h, d, e] = dec * S[h, d, e] + ck * (v[h, e] - pred[e])


@njit(nogil=True, fastmath=True)
def _step_serial(S, q, k, v, g, beta, decay, y, pred):
    H, Ds, Dv = S.shape
    for h in range(H):
        for e in range(Dv):
            y[h, e] = 0.0
            pred[e] = 0.0
        for d in range(Ds):
            qd = q[h, d]
            kd = k[h, d]
            for e in range(Dv):
                s = S[h, d, e]
                y[h, e] += qd * s
                pred[e] += kd * s
        coef = g[h] * beta[h]
        dec = decay[h]
        for d in range(Ds):
            ck = coef * k[h, d]
            for e in range(Dv):
                S[h, d, e] = dec * S[h, d, e] + ck * (v[h, e] - pred[e])


@njit(nogil=True, fastmath=True)
def delta_scan(S, Q, K, V, G, beta, decay):
    """Prefill: Q,K:(N,H,Ds) V:(N,H,Dv) G:(N,H). Updates S in place, returns Y:(N,H,Dv)."""
    N, H, Dv = V.shape
    Y = np.empty((N, H, Dv), dtype=np.float32)
    pred = np.empty(Dv, dtype=np.float32)
    for t in range(N):
        _step_serial(S, Q[t], K[t], V[t], G[t], beta, decay, Y[t], pred)
    return Y


class DeltaState:
    """Per-layer recurrent state; the counterpart of Attention.k_cache / v_cache."""

    def __init__(self, n_heads: int, d_state: int, d_value: int = None,
                 beta=1.0, decay=0.9975):
        self.H, self.Ds = n_heads, d_state
        self.Dv = d_value or d_state
        self.S = np.zeros((self.H, self.Ds, self.Dv), dtype=np.float32)
        self.beta = np.full(self.H, beta, dtype=np.float32) if np.isscalar(beta) \
            else np.asarray(beta, dtype=np.float32)
        self.decay = np.full(self.H, decay, dtype=np.float32) if np.isscalar(decay) \
            else np.asarray(decay, dtype=np.float32)
        self._y = np.empty((self.H, self.Dv), dtype=np.float32)

    @property
    def nbytes(self) -> int:
        return self.S.nbytes

    def step(self, q, k, v, g):
        """Decode: returns y:(H,Dv) (a reused buffer, copy it if you keep it)."""
        delta_step(self.S, q, k, v, g, self.beta, self.decay, self._y)
        return self._y

    def prefill(self, Q, K, V, G):
        return delta_scan(self.S, Q, K, V, G, self.beta, self.decay)

    def reset(self):
        self.S.fill(0.0)
