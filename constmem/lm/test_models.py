import torch, sys
sys.path.insert(0, ".")
from models import *
torch.manual_seed(0)
B, H, T, d, C = 2, 3, 96, 16, 32
q = torch.nn.functional.normalize(torch.randn(B, H, T, d), dim=-1); k = torch.nn.functional.normalize(torch.randn(B, H, T, d), dim=-1)
v = torch.randn(B, H, T, d); beta = torch.sigmoid(torch.randn(B, H, T)); la = -torch.rand(B, H, T) * 0.5
S0 = torch.randn(B, H, d, d) * 0.1
o1, S1 = delta_chunkwise(q, k, v, beta, la, S0, C); o2, S2 = delta_sequential(q, k, v, beta, la, S0)
print("chunkwise vs sequential: max|do|=%.2e max|dS|=%.2e" % ((o1 - o2).abs().max(), (S1 - S2).abs().max()))
assert (o1 - o2).abs().max() < 1e-4 and (S1 - S2).abs().max() < 1e-4
# whole-model streaming vs full forward (hybrid), then padding path
c = Cfg(vocab=100, d=64, layers="GAGG", heads=2, chunk=16)
m = LM(c).eval(); ids = torch.randint(0, 100, (2, 50))
with torch.no_grad():
    full = m(ids)
    caches = [dict() for _ in c.layers]
    outs = [m(ids[:, :30], caches), *[m(ids[:, t:t+1], caches) for t in range(30, 50)]]
    st = torch.cat(outs, 1)
print("full vs prefill+decode: max|dlogit|=%.2e" % (full - st).abs().max())
assert (full - st).abs().max() < 1e-3
# grad flows
m.train(); m(ids).sum().backward(); print("params", m.n_params(), "OK")
