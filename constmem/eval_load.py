"""Zero-shot fact load: facts8-trained model (no decay) streamed with M = 8..18 live facts (generator max: 6 names x 3 objects)."""
import functools, os, sys
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "../../tessera")
os.environ.setdefault("TASK", "facts8"); os.environ.setdefault("CKPT", "checkpoints/facts8_L128_s0_nodecay.pt")
from test_stream_parity import load, stream_logits
from lodestar import gen_facts

torch.manual_seed(0); import random; random.seed(0)
_, _, sm = load()
stream_logits(sm, [1, 2, 3])
for L in (1024, 16384):
    for M in (8, 12, 16, 18):
        ok = tot = 0
        for _ in range(6):
            x, y = gen_facts(1, L, M=M)
            out = stream_logits(sm, x[0].tolist()); mk = (y[0] != -100).numpy()
            ok += int((out.argmax(-1)[mk] == y[0].numpy()[mk]).sum()); tot += int(mk.sum())
        print(f"L={L:6d} M={M:2d}  acc {100*ok/tot:6.2f}% ({tot} queries)", flush=True)
