"""
Train / evaluate one LM variant. Same data order (seeded), tokens, and schedule for every variant.

  python train_lm.py --name gdn --layers GGGGGG --data DIR --tokens 20e6 --out OUT [--bf16]
Val/test are held-out articles. Reports val loss (nats/token), perplexity and bits-per-byte (tokenizer independent).
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn.functional as F
import sys
from models import Cfg, LM


class OriginalCompassLM(torch.nn.Module):
    """Tessera's CompassModel unchanged (python time loop, fixed router, own LayerNorm/FFN), as an LM."""
    def __init__(self, vocab, d, layers, heads):
        super().__init__()
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tessera"))
        from compass import CompassModel
        self.m = CompassModel(vocab, d=d, layers=layers, d_state=64, heads=heads, ctx_w=24)
        self.c = Cfg(vocab=vocab, d=d, layers="C" * layers, heads=heads)
    def forward(self, ids): return self.m(ids)
    def n_params(self): return sum(p.numel() for p in self.parameters())


def batches(path, B, T, seed):
    data = np.memmap(path, dtype=np.uint16, mode="r"); rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(data) - T - 1, B)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64))
        yield x, y


@torch.no_grad()
def evaluate(model, path, T, n_windows=256, bs=16, bf16=False):
    """Non-overlapping windows from the start of the held-out file; fixed for every variant."""
    data = np.memmap(path, dtype=np.uint16, mode="r"); model.eval(); tot = 0.0; n = 0
    for s in range(0, n_windows, bs):
        xs, ys = [], []
        for w in range(s, min(s + bs, n_windows)):
            i = w * T
            xs.append(data[i:i + T].astype(np.int64)); ys.append(data[i + 1:i + T + 1].astype(np.int64))
        x, y = torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            lg = model(x)
        tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum").item(); n += y.numel()
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--layers", default="GGGGGG")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=256); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--tokens", type=float, default=20e6); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--max-minutes", type=float, default=1e9)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model = OriginalCompassLM(8192, a.d, len(a.layers), a.heads) if a.layers.startswith("C") else LM(Cfg(d=a.d, layers=a.layers, heads=a.heads))
    P = model.n_params()
    steps = int(a.tokens // (a.B * a.T))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else 0.1 * a.lr + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, steps - a.warmup)))
    gen = batches(os.path.join(a.data, "train.bin"), a.B, a.T, a.seed)
    log = open(os.path.join(a.out, f"{a.name}.log"), "a")
    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    say(f"[{a.name}] layers={a.layers} d={a.d} params={P:,} steps={steps} tokens/step={a.B*a.T} bf16={a.bf16}")
    t0 = time.time(); hist = []
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, y = next(gen)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
            lg = model(x)
        loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if s % 50 == 0:
            say(f"[{a.name}] step {s}/{steps} loss {loss.item():.3f} lr {lr_at(s):.2e} {((s+1)*a.B*a.T)/(time.time()-t0):.0f} tok/s ({(time.time()-t0)/60:.1f} min)")
        last = s == steps - 1 or (time.time() - t0) / 60 > a.max_minutes
        if (s + 1) % a.eval_every == 0 or last:
            vl = evaluate(model, os.path.join(a.data, "val.bin"), a.T, bf16=a.bf16)
            hist.append(dict(step=s + 1, tokens=(s + 1) * a.B * a.T, val_loss=vl, minutes=(time.time() - t0) / 60))
            say(f"[{a.name}] VAL step {s+1} loss {vl:.4f} ppl {math.exp(vl):.1f}")
            torch.save(dict(cfg=model.c.__dict__, sd=model.state_dict()), os.path.join(a.out, f"{a.name}.pt"))
        if last: break
    tl = evaluate(model, os.path.join(a.data, "test.bin"), a.T, bf16=a.bf16)
    res = dict(name=a.name, layers=a.layers, params=P, tokens=(s + 1) * a.B * a.T, val=hist[-1]["val_loss"], test_loss=tl,
               test_ppl=math.exp(tl), minutes=(time.time() - t0) / 60, history=hist)
    json.dump(res, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
    say(f"[{a.name}] FINAL test loss {tl:.4f} ppl {math.exp(tl):.1f} in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
