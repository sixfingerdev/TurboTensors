"""Sample from any trained model (torch, with caches): python sample.py CKPT TOKENIZER "prompt" [--n 80]"""
import argparse, torch
from tokenizers import Tokenizer
from models import Cfg, LM
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("tokenizer"); ap.add_argument("prompt")
ap.add_argument("--n", type=int, default=80); ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--topk", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed)
d = torch.load(a.ckpt, map_location="cpu"); m = LM(Cfg(**d["cfg"])); m.load_state_dict(d["sd"]); m.eval()
tok = Tokenizer.from_file(a.tokenizer); ids = tok.encode(a.prompt).ids; caches = [dict() for _ in m.blocks]; out = []
with torch.no_grad():
    lg = m(torch.tensor([ids]), caches)[0, -1]
    for _ in range(a.n):
        v, ix = (lg / a.temp).topk(a.topk); nxt = ix[torch.multinomial(torch.softmax(v, -1), 1)].item()
        if nxt == tok.token_to_id("<eos>"): break
        out.append(nxt); lg = m(torch.tensor([[nxt]]), caches)[0, -1]
print(a.prompt + tok.decode(out))
