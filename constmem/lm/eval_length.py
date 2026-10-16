"""
Held-out quality of each trained LM: test bits-per-byte (tokenizer independent) and loss vs window length.
All models were trained on 256-token windows. loss is measured on the same 256 target tokens while the preceding context is varied (0 / 256 / 768 / 3840),
i.e. what extra context buys (or costs) once it is longer than anything seen in training.

    python eval_length.py DATA_DIR CKPT_DIR [name ...]
"""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from models import Cfg, LM

data, ckdir = sys.argv[1], sys.argv[2]; names = sys.argv[3:] or ["attn", "gdn", "hybrid"]
tok = Tokenizer.from_file(f"{data}/tokenizer.json"); test = np.fromfile(f"{data}/test.bin", dtype=np.uint16).astype(np.int64)
torch.manual_seed(0); torch.set_num_threads(4)

def load(n):
    d = torch.load(f"{ckdir}/{n}.pt", map_location="cpu"); m = LM(Cfg(**d["cfg"])); m.load_state_dict(d["sd"]); return m.eval()

@torch.no_grad()
def loss_given_context(m, C, n_win=24, tgt=256, start=4096, stride=5000):
    """Loss on the SAME 256 target tokens for every C; C tokens of preceding text are given as context
    (it may cross an <eos> article boundary: the model sees the raw token stream). C=0 -> no context."""
    tot = 0.0
    for w in range(n_win):
        s = start + w * stride
        x = torch.tensor(test[s - C: s + tgt])[None]; y = torch.tensor(test[s - C + 1: s + tgt + 1])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lg = m(x)[0]
        tot += F.cross_entropy(lg[-tgt:].float(), y[-tgt:], reduction="sum").item()
    return tot / (n_win * tgt)

@torch.no_grad()
def loss_256(m, n_win=256):
    tot = 0.0
    for w in range(n_win):
        i = w * 257
        x = torch.tensor(test[i:i + 256])[None]; y = torch.tensor(test[i + 1:i + 257])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lg = m(x)[0]
        tot += F.cross_entropy(lg.float(), y, reduction="sum").item()
    return tot / (n_win * 256)

res = {}
# bits per byte on the first 256 windows x 256 tokens (same tokens for every model)
sub = test[: 256 * 257]; nbytes = len(tok.decode(sub.tolist(), skip_special_tokens=False).encode("utf-8"))
for n in names:
    m = load(n); r = {}
    tl = loss_256(m)
    r["ppl256"] = math.exp(tl)
    r["bpb"] = tl / math.log(2) * (256 * 256) / nbytes                  # first 256 windows of 256 tokens, as above
    r["ctx"] = {C: loss_given_context(m, C) for C in (0, 256, 768, 3840)}
    res[n] = r
    print(n, f"test ppl(256) {r['ppl256']:.1f} bpb {r['bpb']:.3f} | loss on same 256 targets by context:",
          " ".join(f"C={C}: {v:.3f}" for C, v in r["ctx"].items()), flush=True)
json.dump(res, open("results/long/eval_length.json", "w"), indent=1)
