"""
TurboTensors v4.0 - 
================================
Maximum Performance CPU Inference Engine
Built by a 14-year-old to beat HuggingFace

FINAL FIX: All dtypes handled correctly
"""

import numpy as np
import json
import struct
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from numba import njit, prange

__version__ = "4.0.0"
__author__ = "sixfingerdev"

# Thread optimization
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('MKL_NUM_THREADS', str(os.cpu_count()))

# ============================================================
# NUMBA JIT KERNELS - Maximum Performance
# ============================================================

@njit(fastmath=True, nogil=True)
def _fast_rms_norm_1d(x, weight, eps):
    """Single vector RMSNorm - decode path"""
    D = x.shape[0]
    sum_sq = 0.0
    for i in range(D):
        sum_sq += x[i] * x[i]
    rms_inv = 1.0 / np.sqrt(sum_sq / D + eps)
    out = np.empty(D, dtype=np.float32)
    for i in range(D):
        out[i] = x[i] * rms_inv * weight[i]
    return out

@njit(parallel=True, fastmath=True, nogil=True)
def fast_rms_norm(x, weight, eps):
    """Batched RMSNorm with parallel processing"""
    B, D = x.shape
    out = np.empty((B, D), dtype=np.float32)
    
    for b in prange(B):
        sum_sq = 0.0
        for i in range(D):
            sum_sq += x[b, i] * x[b, i]
        rms_inv = 1.0 / np.sqrt(sum_sq / D + eps)
        for i in range(D):
            out[b, i] = x[b, i] * rms_inv * weight[i]
    return out

@njit(parallel=True, fastmath=True, nogil=True)
def fast_silu_mul(h1, h3):
    """Fused SiLU(x) * gate"""
    B, D = h1.shape
    out = np.empty((B, D), dtype=np.float32)
    
    for b in prange(B):
        for i in range(D):
            val = h1[b, i]
            # Stable sigmoid
            if val >= 0:
                sigmoid = 1.0 / (1.0 + np.exp(-val))
            else:
                exp_val = np.exp(val)
                sigmoid = exp_val / (1.0 + exp_val)
            out[b, i] = val * sigmoid * h3[b, i]
    return out

@njit(fastmath=True, nogil=True)
def fast_embedding_lookup(table, indices):
    """Fast embedding lookup"""
    if indices.ndim == 1:
        B, T = 1, indices.shape[0]
        indices_2d = indices.reshape(1, -1)
    else:
        B, T = indices.shape
        indices_2d = indices
    
    D = table.shape[1]
    out = np.empty((B, T, D), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            idx = indices_2d[b, t]
            for d in range(D):
                out[b, t, d] = table[idx, d]
    return out

@njit(parallel=True, fastmath=True, nogil=True)
def fast_attention_decode(q, k_cache, v_cache, scale, current_pos):
    """
    Optimized single-token attention (decode phase)
    q: (B, H, 1, D)
    k_cache, v_cache: (B, H, MaxLen, D)
    """
    B, H, _, D = q.shape
    T_k = current_pos
    out = np.empty((B, H, 1, D), dtype=np.float32)
    
    for b in prange(B):
        for h in range(H):
            # Compute attention scores
            max_score = -1e30
            scores = np.empty(T_k, dtype=np.float32)
            
            for j in range(T_k):
                score = 0.0
                for d in range(D):
                    score += q[b, h, 0, d] * k_cache[b, h, j, d]
                scores[j] = score * scale
                if scores[j] > max_score:
                    max_score = scores[j]
            
            # Stable softmax
            sum_exp = 0.0
            for j in range(T_k):
                scores[j] = np.exp(scores[j] - max_score)
                sum_exp += scores[j]
            
            inv_sum = 1.0 / (sum_exp + 1e-10)
            
            # Weighted sum
            for d in range(D):
                acc = 0.0
                for j in range(T_k):
                    acc += scores[j] * v_cache[b, h, j, d]
                out[b, h, 0, d] = acc * inv_sum
    
    return out

@njit(parallel=True, fastmath=True, nogil=True)
def fast_attention_prefill(q, k, v, scale):
    """
    Optimized multi-token attention (prefill) with causal mask
    """
    B, H, T, D = q.shape
    out = np.empty((B, H, T, D), dtype=np.float32)
    
    for b in prange(B):
        for h in range(H):
            for i in range(T):
                max_score = -1e30
                scores = np.empty(i + 1, dtype=np.float32)
                
                # Scores (causal: only 0..i)
                for j in range(i + 1):
                    score = 0.0
                    for d in range(D):
                        score += q[b, h, i, d] * k[b, h, j, d]
                    scores[j] = score * scale
                    if scores[j] > max_score:
                        max_score = scores[j]
                
                # Softmax
                sum_exp = 0.0
                for j in range(i + 1):
                    scores[j] = np.exp(scores[j] - max_score)
                    sum_exp += scores[j]
                
                inv_sum = 1.0 / (sum_exp + 1e-10)
                
                # Output
                for d in range(D):
                    acc = 0.0
                    for j in range(i + 1):
                        acc += scores[j] * v[b, h, j, d]
                    out[b, h, i, d] = acc * inv_sum
    
    return out

@njit(fastmath=True, nogil=True)
def fast_qkv_split(qkv, n_heads, head_dim):
    """Split QKV projection: (B, T, 3*H*D) -> (B, H, T, D) each"""
    B, T, _ = qkv.shape
    q = np.empty((B, n_heads, T, head_dim), dtype=np.float32)
    k = np.empty((B, n_heads, T, head_dim), dtype=np.float32)
    v = np.empty((B, n_heads, T, head_dim), dtype=np.float32)
    
    for b in range(B):
        for t in range(T):
            for h in range(n_heads):
                for d in range(head_dim):
                    q[b, h, t, d] = qkv[b, t, h * head_dim + d]
                    k[b, h, t, d] = qkv[b, t, n_heads * head_dim + h * head_dim + d]
                    v[b, h, t, d] = qkv[b, t, 2 * n_heads * head_dim + h * head_dim + d]
    return q, k, v

@njit(fastmath=True, nogil=True)
def fast_attn_output_reshape(attn_out):
    """(B, H, T, D) -> (B, T, H*D)"""
    B, H, T, D = attn_out.shape
    out = np.empty((B, T, H * D), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                for d in range(D):
                    out[b, t, h * D + d] = attn_out[b, h, t, d]
    return out

@njit(fastmath=True, nogil=True)
def fast_top_k_sample(logits, k, temperature):
    """Fast top-k sampling"""
    n = len(logits)
    k = min(k, n)
    
    # Scale by temperature
    scaled = logits / temperature
    
    # Partial sort for top-k
    indices = np.arange(n, dtype=np.int64)
    for i in range(k):
        max_idx = i
        for j in range(i + 1, n):
            if scaled[indices[j]] > scaled[indices[max_idx]]:
                max_idx = j
        if max_idx != i:
            indices[i], indices[max_idx] = indices[max_idx], indices[i]
    
    top_k_indices = indices[:k].copy()
    top_k_logits = np.empty(k, dtype=np.float32)
    for i in range(k):
        top_k_logits[i] = scaled[top_k_indices[i]]
    
    # Stable softmax
    max_val = top_k_logits[0]
    for i in range(1, k):
        if top_k_logits[i] > max_val:
            max_val = top_k_logits[i]
    
    total = 0.0
    for i in range(k):
        top_k_logits[i] = np.exp(top_k_logits[i] - max_val)
        total += top_k_logits[i]
    
    for i in range(k):
        top_k_logits[i] /= total
    
    return top_k_indices, top_k_logits

@njit(fastmath=True, nogil=True)
def apply_repetition_penalty_fast(logits, token_ids, penalty):
    """Apply repetition penalty"""
    out = logits.copy()
    for token_id in token_ids:
        if 0 <= token_id < len(out):
            if out[token_id] > 0:
                out[token_id] /= penalty
            else:
                out[token_id] *= penalty
    return out

@njit(fastmath=True, nogil=True)
def get_unique_tokens(tokens, max_len=512):
    """Extract unique tokens"""
    seen = np.zeros(max_len, dtype=np.int64)
    count = 0
    for t in tokens:
        if count >= max_len:
            break
        found = False
        for i in range(count):
            if seen[i] == t:
                found = True
                break
        if not found:
            seen[count] = t
            count += 1
    return seen[:count]

# ============================================================
# WARMUP
# ============================================================

def warmup_jit(hidden_size=640, n_heads=10, head_dim=64, vocab_size=32000):
    """Pre-compile all JIT functions"""
    print("🔥 Warming up JIT kernels...", end=" ", flush=True)
    start = time.time()
    
    D = hidden_size
    H = n_heads
    HD = head_dim
    
    # Dummy tensors
    x1 = np.random.randn(1, D).astype(np.float32)
    x2 = np.random.randn(4, D).astype(np.float32)
    w = np.random.randn(D).astype(np.float32)
    h1 = np.random.randn(1, D * 4).astype(np.float32)
    
    # Warmup each kernel
    _ = _fast_rms_norm_1d(x1[0], w, 1e-6)
    _ = fast_rms_norm(x2, w, 1e-6)
    _ = fast_silu_mul(h1, h1)
    
    table = np.random.randn(100, D).astype(np.float32)
    ids = np.array([[0, 1, 2]], dtype=np.int64)
    _ = fast_embedding_lookup(table, ids)
    
    q = np.random.randn(1, H, 4, HD).astype(np.float32)
    k = np.random.randn(1, H, 4, HD).astype(np.float32)
    v = np.random.randn(1, H, 4, HD).astype(np.float32)
    _ = fast_attention_prefill(q, k, v, 0.1)
    
    q_dec = np.random.randn(1, H, 1, HD).astype(np.float32)
    k_cache = np.random.randn(1, H, 32, HD).astype(np.float32)
    v_cache = np.random.randn(1, H, 32, HD).astype(np.float32)
    _ = fast_attention_decode(q_dec, k_cache, v_cache, 0.1, 8)
    
    qkv = np.random.randn(1, 4, 3 * H * HD).astype(np.float32)
    _ = fast_qkv_split(qkv, H, HD)
    
    attn_out = np.random.randn(1, H, 4, HD).astype(np.float32)
    _ = fast_attn_output_reshape(attn_out)
    
    logits = np.random.randn(vocab_size).astype(np.float32)
    _ = fast_top_k_sample(logits, 50, 0.8)
    
    tokens = np.array([1, 2, 3, 1, 2], dtype=np.int64)
    unique = get_unique_tokens(tokens, 512)
    _ = apply_repetition_penalty_fast(logits, unique, 1.2)
    
    print(f"✓ {time.time()-start:.2f}s")

# ============================================================
# SAFETENSORS - COMPLETE FIX
# ============================================================

class SafeTensorsFile:
    """Complete safetensors reader with all dtypes"""
    
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'rb')
        
        header_size = struct.unpack('<Q', self.file.read(8))[0]
        header_bytes = self.file.read(header_size)
        self.header = json.loads(header_bytes.decode('utf-8'))
        self.data_offset = 8 + header_size
        self.tensor_info = {k: v for k, v in self.header.items() if k != '__metadata__'}
    
    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        if name not in self.tensor_info:
            return None
        
        info = self.tensor_info[name]
        dtype_str = info['dtype']
        shape = tuple(info['shape'])
        start, end = info['data_offsets']
        
        # Read raw bytes
        self.file.seek(self.data_offset + start)
        data = self.file.read(end - start)
        
        # Handle different dtypes
        if dtype_str == 'BOOL':
            # Boolean: 1 byte per element, skip these (not used in model)
            return None
        
        elif dtype_str == 'U8':
            # Unsigned int8
            return None
        
        elif dtype_str == 'I8':
            # Signed int8
            tensor = np.frombuffer(data, dtype=np.int8).reshape(shape)
            return np.ascontiguousarray(tensor.astype(np.float32))
        
        elif dtype_str == 'I16':
            tensor = np.frombuffer(data, dtype=np.int16).reshape(shape)
            return np.ascontiguousarray(tensor.astype(np.float32))
        
        elif dtype_str == 'I32':
            tensor = np.frombuffer(data, dtype=np.int32).reshape(shape)
            return np.ascontiguousarray(tensor.astype(np.float32))
        
        elif dtype_str == 'I64':
            tensor = np.frombuffer(data, dtype=np.int64).reshape(shape)
            return np.ascontiguousarray(tensor.astype(np.float32))
        
        elif dtype_str == 'F16':
            # Float16: 2 bytes per element
            f16 = np.frombuffer(data, dtype=np.float16)
            tensor = f16.astype(np.float32).reshape(shape)
            return np.ascontiguousarray(tensor)
        
        elif dtype_str == 'BF16':
            # BFloat16: 2 bytes per element
            bf16 = np.frombuffer(data, dtype=np.uint16)
            # Convert to F32: shift left by 16 bits
            f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
            tensor = f32.reshape(shape)
            return np.ascontiguousarray(tensor)
        
        elif dtype_str == 'F32':
            # Float32: 4 bytes per element
            tensor = np.frombuffer(data, dtype=np.float32).reshape(shape)
            return np.ascontiguousarray(tensor)
        
        elif dtype_str == 'F64':
            tensor = np.frombuffer(data, dtype=np.float64).reshape(shape)
            return np.ascontiguousarray(tensor.astype(np.float32))
        
        else:
            print(f"⚠️  Skipping unknown dtype {dtype_str} for {name}")
            return None
    
    def keys(self):
        return self.tensor_info.keys()
    
    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

# ============================================================
# LAYERS
# ============================================================

class RMSNorm:
    __slots__ = ['weight', 'eps']
    
    def __init__(self, weight: np.ndarray, eps: float = 1e-6):
        self.weight = np.ascontiguousarray(weight.astype(np.float32))
        self.eps = eps
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_2d = np.ascontiguousarray(x.reshape(-1, x.shape[-1]))
        
        if x_2d.shape[0] == 1:
            out = _fast_rms_norm_1d(x_2d[0], self.weight, self.eps).reshape(1, -1)
        else:
            out = fast_rms_norm(x_2d, self.weight, self.eps)
        
        return out.reshape(orig_shape)


class Linear:
    """Pre-transposed weights for efficient matmul"""
    __slots__ = ['weight_T', 'bias']
    
    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None):
        self.weight_T = np.ascontiguousarray(weight.T.astype(np.float32))
        self.bias = np.ascontiguousarray(bias.astype(np.float32)) if bias is not None else None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # NumPy BLAS
        out = x.astype(np.float32) @ self.weight_T
        if self.bias is not None:
            out = out + self.bias
        return out


class FeedForward:
    __slots__ = ['w1', 'w2', 'w3']
    
    def __init__(self, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray):
        self.w1 = Linear(w1)
        self.w2 = Linear(w2)
        self.w3 = Linear(w3)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        h1 = self.w1(x)
        h3 = self.w3(x)
        
        # Fused activation
        orig_shape = h1.shape
        h1_2d = np.ascontiguousarray(h1.reshape(-1, h1.shape[-1]))
        h3_2d = np.ascontiguousarray(h3.reshape(-1, h3.shape[-1]))
        fused = fast_silu_mul(h1_2d, h3_2d).reshape(orig_shape)
        
        return self.w2(fused)


class Attention:
    """Optimized attention with separate prefill/decode paths"""
    __slots__ = ['qkv', 'proj', 'n_heads', 'head_dim', 'scale',
                 'k_cache', 'v_cache', 'max_seq_len', 'current_pos', 'hidden_size']
    
    def __init__(self, qkv_weight: np.ndarray, proj_weight: np.ndarray,
                 n_heads: int, head_dim: int, max_seq_len: int = 2048):
        self.qkv = Linear(qkv_weight)
        self.proj = Linear(proj_weight)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_size = n_heads * head_dim
        self.scale = np.float32(1.0 / np.sqrt(head_dim))
        self.max_seq_len = max_seq_len
        
        self.k_cache = None
        self.v_cache = None
        self.current_pos = 0
    
    def _init_cache(self, B: int):
        self.k_cache = np.zeros((B, self.n_heads, self.max_seq_len, self.head_dim), dtype=np.float32)
        self.v_cache = np.zeros((B, self.n_heads, self.max_seq_len, self.head_dim), dtype=np.float32)
        self.current_pos = 0
    
    def __call__(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        B, T, C = x.shape
        
        if use_cache and self.k_cache is None:
            self._init_cache(B)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = np.ascontiguousarray(qkv)
        
        # Split
        q, k, v = fast_qkv_split(qkv, self.n_heads, self.head_dim)
        
        if use_cache:
            # Update cache
            self.k_cache[:, :, self.current_pos:self.current_pos+T, :] = k
            self.v_cache[:, :, self.current_pos:self.current_pos+T, :] = v
            self.current_pos += T
            
            # Choose path
            if T == 1:
                # Decode
                attn_out = fast_attention_decode(
                    np.ascontiguousarray(q),
                    self.k_cache,
                    self.v_cache,
                    self.scale,
                    self.current_pos
                )
            else:
                # Prefill
                attn_out = fast_attention_prefill(
                    np.ascontiguousarray(q),
                    np.ascontiguousarray(k),
                    np.ascontiguousarray(v),
                    self.scale
                )
        else:
            attn_out = fast_attention_prefill(
                np.ascontiguousarray(q),
                np.ascontiguousarray(k),
                np.ascontiguousarray(v),
                self.scale
            )
        
        # Reshape and project
        out = fast_attn_output_reshape(attn_out)
        return self.proj(out)
    
    def clear_cache(self):
        if self.k_cache is not None:
            self.k_cache.fill(0)
            self.v_cache.fill(0)
        self.current_pos = 0


class TransformerBlock:
    __slots__ = ['attn', 'ff', 'norm1', 'norm2']
    
    def __init__(self, attn: Attention, ff: FeedForward, norm1: RMSNorm, norm2: RMSNorm):
        self.attn = attn
        self.ff = ff
        self.norm1 = norm1
        self.norm2 = norm2
    
    def __call__(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ff(self.norm2(x))
        return x

# ============================================================
# MODEL
# ============================================================

class TurboLLM:
    """TurboTensors v4.0 - JET MODE 🚀"""
    
    def __init__(self, model_path: Union[str, Path], verbose: bool = True):
        self.verbose = verbose
        self.model_path = Path(model_path)
        
        self.blocks: List[TransformerBlock] = []
        self.tok_emb = None
        self.pos_emb = None
        self.final_norm = None
        self.lm_head = None
        
        self._load_model()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_model(self):
        start = time.time()
        self._log(f"📂 Loading: {self.model_path.name}")
        
        # Find files
        if self.model_path.is_file():
            st_files = [self.model_path]
            config_dir = self.model_path.parent
        else:
            st_files = sorted(self.model_path.glob("*.safetensors"))
            config_dir = self.model_path
        
        if not st_files:
            raise FileNotFoundError(f"No safetensors in {self.model_path}")
        
        # Config
        config_path = config_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        self.hidden_size = config.get("hidden_size", 640)
        self.n_heads = config.get("num_attention_heads", 10)
        self.n_layers = config.get("num_hidden_layers", 10)
        self.vocab_size = config.get("vocab_size", 32000)
        self.max_position = config.get("max_position_embeddings", 512)
        self.head_dim = self.hidden_size // self.n_heads
        
        self._log(f"   Config: {self.hidden_size}d, {self.n_heads}h, {self.n_layers}L, {self.vocab_size}v")
        
        # Load tensors
        tensors = {}
        for st_file in st_files:
            reader = SafeTensorsFile(str(st_file))
            for key in reader.keys():
                tensor = reader.get_tensor(key)
                if tensor is not None:  # Skip None (BOOL, etc.)
                    tensors[key] = tensor
        
        self._log(f"   Loaded {len(tensors)} tensors")
        
        # Build
        self._build_model(tensors)
        
        # Warmup
        warmup_jit(self.hidden_size, self.n_heads, self.head_dim, self.vocab_size)
        
        self._log(f"   ✅ Ready in {time.time()-start:.1f}s")
    
    def _build_model(self, tensors: Dict[str, np.ndarray]):
        # Embeddings
        self.tok_emb = np.ascontiguousarray(tensors["tok_emb.weight"])
        self.pos_emb = np.ascontiguousarray(tensors["pos_emb.weight"])
        
        # Blocks
        for i in range(self.n_layers):
            prefix = f"blocks.{i}."
            
            attn = Attention(
                tensors[f"{prefix}attn.qkv.weight"],
                tensors[f"{prefix}attn.proj.weight"],
                self.n_heads, self.head_dim, self.max_position
            )
            
            ff = FeedForward(
                tensors[f"{prefix}ff.w1.weight"],
                tensors[f"{prefix}ff.w2.weight"],
                tensors[f"{prefix}ff.w3.weight"]
            )
            
            norm1 = RMSNorm(tensors[f"{prefix}norm1.weight"])
            norm2 = RMSNorm(tensors[f"{prefix}norm2.weight"])
            
            self.blocks.append(TransformerBlock(attn, ff, norm1, norm2))
        
        # Final
        self.final_norm = RMSNorm(tensors["norm.weight"])
        
        if "lm_head.weight" in tensors:
            self.lm_head = Linear(tensors["lm_head.weight"])
        else:
            self.lm_head = Linear(self.tok_emb)
    
    def clear_cache(self):
        for block in self.blocks:
            block.attn.clear_cache()
    
    def forward(self, input_ids: np.ndarray, use_cache: bool = True) -> np.ndarray:
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = fast_embedding_lookup(self.tok_emb, input_ids)
        
        # Position
        past_len = 0
        if use_cache and self.blocks[0].attn.k_cache is not None:
            past_len = self.blocks[0].attn.current_pos
        
        pos = np.arange(T, dtype=np.int64) + past_len
        pos = np.clip(pos, 0, self.max_position - 1)
        pos_emb = self.pos_emb[pos]
        
        x = tok_emb + pos_emb
        
        # Transformer
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        prompt: Union[str, List[int], np.ndarray],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        tokenizer=None,
        stream: bool = False,
    ) -> Union[np.ndarray, str]:
        """Generate with maximum performance"""
        
        # Parse prompt
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("Tokenizer needed")
            input_ids = tokenizer.encode(prompt)
        elif isinstance(prompt, list):
            input_ids = prompt
        else:
            input_ids = prompt.tolist()
        
        generated = list(input_ids)
        self.clear_cache()
        
        if self.verbose:
            self._log(f"🔄 Generating {max_new_tokens} tokens (prompt: {len(input_ids)})")
        
        start_time = time.time()
        
        # PREFILL
        prefill_ids = np.array([generated], dtype=np.int64)
        logits = self.forward(prefill_ids, use_cache=True)
        
        prefill_time = time.time() - start_time
        
        if self.verbose:
            self._log(f"   Prefill: {len(input_ids)} toks @ {len(input_ids)/prefill_time:.1f} tok/s")
        
        decode_start = time.time()
        
        # DECODE
        for i in range(max_new_tokens):
            last_logits = logits[0, -1].astype(np.float32).copy()
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                gen_arr = np.array(generated[-512:], dtype=np.int64)
                unique_toks = get_unique_tokens(gen_arr, 512)
                last_logits = apply_repetition_penalty_fast(last_logits, unique_toks, repetition_penalty)
            
            # Sample
            if top_k > 0:
                top_k_idx, probs = fast_top_k_sample(last_logits, top_k, temperature)
                next_token = int(np.random.choice(top_k_idx, p=probs))
            else:
                scaled = last_logits / temperature
                scaled = scaled - scaled.max()
                probs = np.exp(scaled)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))
            
            generated.append(next_token)
            
            # Stream
            if stream and tokenizer:
                print(tokenizer.decode([next_token]), end='', flush=True)
            
            # Forward
            new_token = np.array([[next_token]], dtype=np.int64)
            logits = self.forward(new_token, use_cache=True)
            
            # Progress
            if self.verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - decode_start
                speed = (i + 1) / elapsed
                print(f"\r   Decode: {i+1}/{max_new_tokens} @ {speed:.1f} tok/s", end="", flush=True)
        
        if stream:
            print()
        
        decode_time = time.time() - decode_start
        
        if self.verbose:
            decode_speed = max_new_tokens / decode_time
            print(f"\n   ✅ Decode: {decode_speed:.1f} tok/s")
        
        if tokenizer is not None:
            return tokenizer.decode(generated)
        return np.array(generated)

# ============================================================
# UTILITIES
# ============================================================

def download_model(model_id: str, save_dir: str = "./models") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")
    
    save_path = Path(save_dir) / model_id.replace("/", "_")
    
    if save_path.exists():
        print(f"✅ Cached: {save_path}")
        return save_path
    
    print(f"📥 Downloading {model_id}...")
    snapshot_download(repo_id=model_id, local_dir=save_path, ignore_patterns=["*.bin", "*.pt"])
    return save_path


def benchmark(model: TurboLLM, prompt: str, tokenizer, max_new_tokens: int = 50, runs: int = 3):
    """Benchmark with multiple runs"""
    input_ids = tokenizer.encode(prompt)
    
    times = []
    for run in range(runs):
        model.clear_cache()
        start = time.time()
        output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                               temperature=0.8, top_k=50, repetition_penalty=1.2, tokenizer=None)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    best_time = min(times)
    
    return {
        "avg_speed": max_new_tokens / avg_time,
        "best_speed": max_new_tokens / best_time,
        "output": tokenizer.decode(output)
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" TURBOTENSORS v4.0 - JET MODE")
    print("=" * 70)
    
    # Setup
    model_path = download_model("sixfingerdev/kayra-1-exp")
    
    # Load
    print("\n🔧 LOADING")
    model = TurboLLM(model_path)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sixfingerdev/kayra-1-exp")
    
    # Benchmark
    print("\n BENCHMARK")
    print("-" * 70)
    
    result = benchmark(model, "Türkiye", tokenizer, max_new_tokens=50, runs=3)
    print(f"Speed: {result['avg_speed']:.1f} tok/s (best: {result['best_speed']:.1f})")
    print(f"Output: {result['output'][:100]}...")
    
    # HuggingFace comparison
    print("\nVS HUGGINGFACE")
    print("-" * 70)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            "sixfingerdev/kayra-1-exp", trust_remote_code=True, torch_dtype=torch.float32
        )
        hf_model.eval()
        
        input_torch = torch.tensor([tokenizer.encode("Türkiye")])
        
        # Warmup
        with torch.no_grad():
            _ = hf_model.generate(input_torch, max_new_tokens=5, do_sample=True)
        
        # Benchmark
        times = []
        for _ in range(3):
            start = time.time()
            with torch.no_grad():
                _ = hf_model.generate(input_torch, max_new_tokens=50, temperature=0.8,
                                     do_sample=True, top_k=50, repetition_penalty=1.3)
            times.append(time.time() - start)
        
        hf_speed = 50 / (sum(times) / len(times))
        
        print(f"HuggingFace: {hf_speed:.1f} tok/s")
        print(f"TurboTensors: {result['avg_speed']:.1f} tok/s")
        
        ratio = hf_speed / result['avg_speed']
        if ratio > 1:
            print(f"Gap: {ratio:.1f}x")
        else:
            print(f" FASTER by {1/ratio:.1f}x!")
    
    except Exception as e:
        print(f"HF comparison failed: {e}")
    
    # Demo
    print("\n GENERATION")
    print("-" * 70)
    
    for prompt in ["Türkiye", "Yapay zeka", "İstanbul"]:
        out = model.generate(prompt, max_new_tokens=30, tokenizer=tokenizer)
        print(f" {prompt} → {out[:80]}...")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
