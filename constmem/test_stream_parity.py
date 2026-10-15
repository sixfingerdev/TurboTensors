"""Streaming numpy/Numba port vs the torch CompassModel it was trained as (facts4 checkpoint)."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../../tessera")
from compass import CompassModel
from lodestar import TASKS
from compass_stream import StreamModel

CK = os.environ.get("CKPT", "checkpoints/facts4_L128_s0.pt")


def load():
    gen, vocab = TASKS["facts4"]
    m = CompassModel(vocab, d=64, layers=2, d_state=64, heads=1, ctx_w=24)
    for l in m.layers:
        l.q_pooled = True; l.k_prod = True; l.set_pool_balance(0.03)
    sd = torch.load(CK, map_location="cpu"); m.load_state_dict(sd); m.eval()
    return gen, m, StreamModel(sd, layers=2, H=1, Ds=64)


def stream_logits(sm, ids):
    sm.reset()
    return np.stack([sm.step(int(t)) for t in ids])


def main():
    torch.manual_seed(1)
    gen, m, sm = load()
    x, y = gen(8, 128)
    with torch.no_grad():
        ref = m(x).numpy()
    worst = 0.0; agree = 0; tot = 0
    for b in range(x.shape[0]):
        out = stream_logits(sm, x[b].tolist())
        worst = max(worst, np.abs(out - ref[b]).max())
        agree += int((out.argmax(-1) == ref[b].argmax(-1)).sum()); tot += out.shape[0]
    print(f"max|dlogit|={worst:.2e}  argmax agreement {agree}/{tot}")
    assert worst < 1e-3
    print("STREAM PARITY OK")


if __name__ == "__main__":
    main()
