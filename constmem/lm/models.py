"""
Small causal LMs for a like-for-like comparison on CPU.

  "A" causal softmax attention (rotary)      -> Transformer baseline, KV cache grows with context
  "G" gated delta-rule layer                 -> COMPASS recurrence made LM-ready: learned q/k/v + short conv,
                                                data-dependent write strength beta_t and decay alpha_t,
                                                chunkwise-parallel training, constant-size state at inference.
                                                Prior art, stated plainly: this is Gated DeltaNet (Yang et al., 2024);
                                                COMPASS's own recurrence (Tessera) is its fixed-gate special case.
  layers = a string such as "AAAAAA" (Transformer), "GGGGGG" (all delta), "GGAGGA" (hybrid)

State recurrence per head (S in R^{d x d}):  S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T ,  o_t = S_t^T q_t
"""
import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 8192
    d: int = 256
    layers: str = "GGAGGA"
    heads: int = 4
    mlp_mult: float = 2.667        # SwiGLU, ~ 8/3
    conv: int = 4
    chunk: int = 32
    max_len: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w


class MLP(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        h = int(d * mult); h = (h + 63) // 64 * 64
        self.w1 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(d, h, bias=False); self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotary(x, base=10000.0, offset=0):
    B, H, T, d = x.shape
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = torch.outer(torch.arange(offset, offset + T, device=x.device, dtype=torch.float32), inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class Attn(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.H, self.hd = c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False); self.o = nn.Linear(c.d, c.d, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        off = 0 if cache is None else cache["k"].shape[2] if "k" in cache else 0
        q, k = rotary(q, offset=off), rotary(k, offset=off)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and off == 0))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


def delta_chunkwise(q, k, v, beta, logalpha, S0, C):
    """Exact chunkwise-parallel form of the gated delta rule (float32).
    q,k,v:(B,H,T,d)  beta,logalpha:(B,H,T)  S0:(B,H,d,d). T must be a multiple of C. Returns o, S_T."""
    B, H, T, d = q.shape; n = T // C
    q, k, v = (t.reshape(B, H, n, C, d) for t in (q, k, v))
    beta = beta.reshape(B, H, n, C); la = logalpha.reshape(B, H, n, C)
    cum = la.cumsum(-1)                                          # log gamma_i inside the chunk
    gam = cum.exp()
    le = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)                  # [i,j] = log(gamma_i / gamma_j)
    M = diff.masked_fill(~le, float("-inf")).exp()                # lower-triangular incl. diagonal
    Mstrict = M.tril(-1)
    KK = k @ k.transpose(-1, -2)
    A = torch.eye(C, device=q.device) + beta.unsqueeze(-1) * KK * Mstrict
    rhs = torch.cat([beta.unsqueeze(-1) * v, (beta * gam).unsqueeze(-1) * k], -1)
    X = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    Vt, Wp = X[..., :d], X[..., d:]
    QK = (q @ k.transpose(-1, -2)) * M
    gq = gam.unsqueeze(-1) * q
    dend = (cum[..., -1:] - cum).exp()
    gend = gam[..., -1]
    S = S0; outs = []
    for c in range(n):
        Dc = Vt[:, :, c] - Wp[:, :, c] @ S
        outs.append(gq[:, :, c] @ S + QK[:, :, c] @ Dc)
        S = gend[:, :, c, None, None] * S + (k[:, :, c] * dend[:, :, c].unsqueeze(-1)).transpose(-1, -2) @ Dc
    return torch.stack(outs, 2).reshape(B, H, T, d), S


def delta_sequential(q, k, v, beta, logalpha, S0):
    """Reference / decode form, token by token."""
    B, H, T, d = q.shape; S = S0; outs = []
    for t in range(T):
        a = logalpha[:, :, t].exp()[..., None, None]; b = beta[:, :, t][..., None, None]
        kt = k[:, :, t]
        S = a * (S - b * kt.unsqueeze(-1) * (kt.unsqueeze(-2) @ S)) + b * kt.unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, 2), S


class GDN(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.H, self.hd, self.C, self.K = c.heads, c.d // c.heads, c.chunk, c.conv
        D = c.d
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.ab = nn.Linear(D, 2 * self.H, bias=False)           # beta, decay inputs
        self.g = nn.Linear(D, D, bias=False)                      # output gate
        self.o = nn.Linear(D, D, bias=False)
        self.conv = nn.Conv1d(3 * D, 3 * D, c.conv, groups=3 * D, padding=c.conv - 1, bias=False)
        self.onorm = RMSNorm(self.hd)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus

    def _proj(self, x, conv_state=None):
        B, T, D = x.shape
        qkv = self.qkv(x).transpose(1, 2)                         # (B,3D,T)
        if conv_state is None:
            qkv = self.conv(qkv)[..., :T]
        else:                                                      # streaming: prepend last K-1 inputs
            full = torch.cat([conv_state, qkv], -1)
            conv_state_new = full[..., -(self.K - 1):]
            qkv = F.conv1d(full, self.conv.weight, groups=3 * D)
        qkv = F.silu(qkv).transpose(1, 2)
        q, k, v = qkv.reshape(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        q = F.normalize(q.float(), dim=-1) * self.hd ** -0.5; k = F.normalize(k.float(), dim=-1)
        ab = self.ab(x).float()
        beta = torch.sigmoid(ab[..., :self.H]).transpose(1, 2)
        la = (-torch.exp(self.A_log.float()) * F.softplus(ab[..., self.H:] + self.dt_bias.float())).transpose(1, 2)
        return q, k, v.float(), beta, la, (None if conv_state is None else conv_state_new)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        streaming = cache is not None
        if streaming and "S" not in cache:
            cache["S"] = x.new_zeros(B, self.H, self.hd, self.hd, dtype=torch.float32)
            cache["conv"] = x.new_zeros(B, 3 * D, self.K - 1)
        q, k, v, beta, la, cs = self._proj(x, cache["conv"] if streaming else None)
        S0 = cache["S"] if streaming else torch.zeros(B, self.H, self.hd, self.hd, device=x.device)
        if T == 1 or T % self.C:
            if T % self.C and T > 1 and not streaming:
                pad = self.C - T % self.C                          # right-pad: causal, so padding cannot leak back
                p = lambda t: F.pad(t, (0, 0, 0, pad)) if t.dim() == 4 else F.pad(t, (0, pad))
                o, S = delta_chunkwise(p(q), p(k), p(v), p(beta), p(la), S0, self.C); o = o[:, :, :T]
            else:
                o, S = delta_sequential(q, k, v, beta, la, S0)
        else:
            o, S = delta_chunkwise(q, k, v, beta, la, S0, self.C)
        if streaming:
            cache["S"], cache["conv"] = S, cs
        o = self.onorm(o.transpose(1, 2).to(x.dtype)) * F.silu(self.g(x)).view(B, T, self.H, self.hd)
        return self.o(o.reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, c: Cfg, kind: str):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.mix = Attn(c) if kind == "A" else GDN(c)
        self.mlp = MLP(c.d, c.mlp_mult)

    def forward(self, x, cache=None):
        x = x + self.mix(self.n1(x), cache)
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c, k) for k in c.layers)
        self.norm = RMSNorm(c.d)
        self.apply(self._init)
        for b in self.blocks:                                      # scaled residual init
            nn.init.normal_(b.mlp.w2.weight, std=0.02 / math.sqrt(2 * len(c.layers)))
            nn.init.normal_(b.mix.o.weight, std=0.02 / math.sqrt(2 * len(c.layers)))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, caches=None):
        x = self.emb(ids)
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i])
        return F.linear(self.norm(x), self.emb.weight)             # tied head

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
