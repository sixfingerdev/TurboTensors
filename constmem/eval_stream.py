"""
Streaming accuracy + speed of the trained facts4 COMPASS model with constant memory.
Sequences much longer than the 128 it was trained on are processed token by token; the only
state is `StreamModel.nbytes` bytes, whatever the length.

    python eval_stream.py [--lens 128 1024 4096 16384] [--seqs 4]
"""
import argparse, sys, time, json
import numpy as np, torch
sys.path.insert(0, ".")
from test_stream_parity import load, stream_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 1024, 4096, 16384])
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--decay", type=float, help="override the learned per-step state decay at inference (ablation)")
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.manual_seed(0)
    gen, m, sm = load()
    if a.decay is not None:
        for l in sm.layers:
            l.state.decay[:] = a.decay
    stream_logits(sm, [1, 2, 3])                      # JIT warmup
    print('decay:', 'learned' if a.decay is None else a.decay, flush=True)
    rows = []
    for L in a.lens:
        x, y = gen(a.seqs, L)
        ok = tot = 0; t0 = time.perf_counter()
        for b in range(a.seqs):
            out = stream_logits(sm, x[b].tolist())
            mask = (y[b] != -100) if y.dim() == 2 else None
            if mask is not None:
                ok += int((out.argmax(-1)[mask.numpy()] == y[b][mask].numpy()).sum()); tot += int(mask.sum())
            else:
                ok += int(out[-1].argmax() == int(y[b])); tot += 1
        dt = time.perf_counter() - t0
        row = dict(L=L, acc=ok / max(tot, 1), queries=tot, tok_per_s=a.seqs * L / dt, state_KB=sm.nbytes / 1024)
        rows.append(row)
        print(f"L={L:6d}  acc {100*row['acc']:6.2f}% ({tot} queries)  {row['tok_per_s']:7.0f} tok/s  state {row['state_KB']:.1f} KB", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
