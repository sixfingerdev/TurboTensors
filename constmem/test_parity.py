"""Numba recurrence vs the torch loop copied from tessera/compass.py (CompassLayer.forward)."""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from delta_state import DeltaState


def torch_reference(Q, K, V, g, beta, decay):
    """Verbatim recurrence from compass.py, B=1. Q,K,V:[1,N,H,Ds] g:[1,N,H]."""
    B, N, H, Ds = Q.shape
    S = torch.zeros(B, H, Ds, Ds)
    outs = []
    for t in range(N):
        q, k, v = Q[:, t], K[:, t], V[:, t]
        outs.append(torch.einsum('bhd,bhde->bhe', q, S))
        pred = torch.einsum('bhd,bhde->bhe', k, S)
        err = v - pred
        S = S * decay.view(1, -1, 1, 1)
        coef = g[:, t].unsqueeze(-1) * beta.view(1, -1, 1)
        S = S + coef.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', k, err)
    return torch.stack(outs, 1), S


def main():
    torch.manual_seed(0)
    N, H, Ds = 300, 10, 64
    Q = torch.randn(1, N, H, Ds)
    K = torch.nn.functional.normalize(torch.randn(1, N, H, Ds), dim=-1)
    V = torch.randn(1, N, H, Ds)
    g = torch.sigmoid(torch.randn(1, N, H))
    beta = 2 * torch.sigmoid(torch.full((H,), -1.1))
    decay = torch.sigmoid(torch.full((H,), 6.0))

    Y_ref, S_ref = torch_reference(Q, K, V, g, beta, decay)
    to = lambda x: np.ascontiguousarray(x.numpy(), dtype=np.float32)

    # path 1: token-by-token decode
    st = DeltaState(H, Ds, beta=to(beta), decay=to(decay))
    Y1 = np.stack([st.step(to(Q[0, t]), to(K[0, t]), to(V[0, t]), to(g[0, t])).copy()
                   for t in range(N)])
    # path 2: prefill scan
    st2 = DeltaState(H, Ds, beta=to(beta), decay=to(decay))
    Y2 = st2.prefill(to(Q[0]), to(K[0]), to(V[0]), to(g[0]))

    for name, Y, S in (("decode", Y1, st.S), ("prefill", Y2, st2.S)):
        ey = np.abs(Y - Y_ref[0].numpy()).max()
        es = np.abs(S - S_ref[0].numpy()).max()
        print(f"{name:8s} max|dY|={ey:.2e} max|dS|={es:.2e}")
        assert ey < 1e-3 and es < 1e-3, f"{name} diverges from torch reference"
    print("PARITY OK")


if __name__ == "__main__":
    main()
