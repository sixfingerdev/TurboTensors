"""stream_lm (numpy/Numba) vs torch LM on real validation text."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from models import Cfg, LM
from stream_lm import GDNStream

ck, val = sys.argv[1], sys.argv[2]
d = torch.load(ck, map_location="cpu"); m = LM(Cfg(**d["cfg"])); m.load_state_dict(d["sd"]); m.eval()
ids = np.fromfile(val, dtype=np.uint16)[:300].astype(np.int64)
with torch.no_grad(): ref = m(torch.tensor(ids)[None])[0].numpy()
s = GDNStream(ck); out = np.stack([s.step(int(t)) for t in ids])
print("max|dlogit| = %.2e  argmax agreement %d/%d  state %.0f KB" % (np.abs(out - ref).max(), (out.argmax(-1) == ref.argmax(-1)).sum(), len(ids), s.state_bytes / 1024))
assert np.abs(out - ref).max() < 1e-2
print("STREAM-LM PARITY OK")
