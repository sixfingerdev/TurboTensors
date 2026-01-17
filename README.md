# ⚡ TurboTensors v4.0 — JET MODE (Experimental)

TurboTensors is an experimental, low-level CPU inference engine focused on minimizing framework overhead and maximizing real-time token generation on low-end and mid-range CPUs.

It is designed primarily for small to mid-sized language models (approximately 50M–300M parameters) and has been tested mainly with Turkish LLMs such as Kayra-1-exp.

IMPORTANT:
TurboTensors is NOT a general-purpose replacement for optimized BLAS-based frameworks on high-end CPUs.
It intentionally prioritizes low-overhead execution over heavy vectorized libraries.

---

## 🎯 Design Philosophy

Modern CPU inference stacks (PyTorch, oneDNN, MKL) perform exceptionally well on high-end CPUs,
but introduce significant overhead on resource-constrained systems.

TurboTensors targets:
- Older CPUs
- Limited cache sizes
- No AVX-512
- GPU-less environments
- Edge / experimental setups

The goal is predictable, low-latency inference, not peak FLOPS.

---

## ✨ Core Features

- Numba-JIT Kernels (LLVM)
  Critical execution paths are JIT-compiled to native machine code, avoiding Python interpreter overhead.

- Fused Operator Execution
  Operations such as RMSNorm and activation functions are fused to reduce memory traffic and cache misses.

- Prefill vs Decode Separation
  Distinct computational paths for context understanding and token generation, reducing redundant work.

- KV Cache Awareness
  Aggressive reuse of key/value tensors during autoregressive decoding.

- Zero-Copy Safetensors Loading
  Weights are accessed with minimal memory duplication (BF16 / F16 / F32 supported).

- Thread-Level Parallelism
  Uses nogil-style execution and explicit thread control to avoid Python GIL bottlenecks.

---

## 📊 Performance Snapshot

Test model: Kayra-1-exp (~85M parameters)
Hardware: consumer-grade laptop CPU (non-server, non-AVX512)

TurboTensors v4: ~45–60 tokens/s, very low first-token latency  
HuggingFace (CPU): ~12–18 tokens/s, high first-token latency

NOTE:
Results are hardware-dependent and primarily reflect performance on low to mid-tier CPUs.
On high-end CPUs, optimized BLAS-based engines may outperform TurboTensors.

---

## 🔬 Benchmark Results

Comprehensive benchmarks of core TurboTensors operations running on a GitHub Actions runner (Ubuntu, 4-core CPU):

### Core Operation Performance

| Operation | Time (ms) | Description |
|-----------|-----------|-------------|
| RMS Norm | 0.024 | Layer normalization |
| SiLU × Gate (Fused) | 0.060 | Fused activation function |
| Attention (Prefill, 32 tokens) | 0.436 | Multi-token attention |
| Attention (Decode, 1 token) | 0.093 | Single-token attention |
| Top-K Sampling | 2.410 | Token selection |

**Estimated Throughput:** ~386 tokens/second (based on core operations)

### Key Performance Characteristics

- **JIT Compilation:** ~5.2 seconds warmup time (one-time cost)
- **Memory Efficiency:** Zero-copy safetensors loading
- **Parallel Processing:** Numba parallel loops for multi-core utilization
- **Cache Optimization:** KV cache reuse during autoregressive decoding

### Example Output

```
======================================================================
TURBOTENSORS v4.0 - CORE OPERATIONS BENCHMARK
======================================================================

Configuration:
  Batch size: 1
  Sequence length: 128
  Hidden size: 640
  Attention heads: 10
  Head dimension: 64
  Vocabulary size: 32000

 Warming up JIT kernels...
🔥 Warming up JIT kernels... ✓ 5.25s
✓ Warmup completed in 5.25s

 [1/5] Benchmarking RMS Norm...
     Average time: 0.024 ms

 [2/5] Benchmarking SiLU * Gate (Fused)...
     Average time: 0.060 ms

 [3/5] Benchmarking Attention (Prefill)...
     Average time: 0.436 ms

 [4/5] Benchmarking Attention (Decode)...
     Average time: 0.093 ms

 [5/5] Benchmarking Top-K Sampling...
     Average time: 2.410 ms

======================================================================
BENCHMARK SUMMARY
======================================================================

Operation                    Time (ms)
----------------------------------------------------------------------
RMS Norm                        0.024
SiLU * Gate (Fused)             0.060
Attention (Prefill, 32 tok)     0.436
Attention (Decode, 1 tok)       0.093
Top-K Sampling                  2.410
----------------------------------------------------------------------

Estimated decode throughput: ~386.6 tokens/second

======================================================================
✓ BENCHMARK COMPLETE!
======================================================================
```

**Note:** Actual full-model performance will vary based on model architecture, hardware, and workload characteristics. These benchmarks represent individual operation performance on standard GitHub Actions infrastructure.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/sixfingerdev/TurboTensors.git
cd TurboTensors
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

The following packages will be installed:
- **numpy** (>=1.21.0) - Numerical computing
- **numba** (>=0.56.0) - JIT compilation for performance
- **transformers** (>=4.30.0) - Model tokenizer support
- **huggingface_hub** (>=0.16.0) - Model downloading
- **torch** (>=2.0.0) - PyTorch backend (optional, for comparisons)

### Step 3: Run the Code
```bash
python main.py
```

---

## 🚀 Usage

### Basic Usage

```python
from main import TurboLLM, download_model
from transformers import AutoTokenizer

# Download and load model
model_path = download_model("sixfingerdev/kayra-1-exp")
model = TurboLLM(model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sixfingerdev/kayra-1-exp")

# Generate text
output = model.generate(
    "Türkiye",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.2,
    tokenizer=tokenizer,
    stream=True
)

print(output)
```

### Advanced Options

```python
# Custom generation parameters
output = model.generate(
    prompt="Your prompt here",
    max_new_tokens=100,        # Number of tokens to generate
    temperature=0.8,            # Sampling temperature (0.0-2.0)
    top_k=50,                   # Top-K sampling
    repetition_penalty=1.2,     # Penalize repeated tokens
    tokenizer=tokenizer,
    stream=True                 # Stream output token by token
)
```

**Developer note:** Always enable `use_cache=True` to fully benefit from KV caching during generation.

---

## 🧠 Engineering Notes

Key ideas explored in this project:
- Cache-friendly memory layouts (L1/L2 aware)
- Partial Top-K sampling to avoid full logit sorting
- Manual control over compute granularity
- Avoidance of heavy framework abstractions

This project intentionally trades generality for clarity and control.

---

## 🚧 Limitations

- Not optimized for AVX-512 or large-core-count servers
- Not intended for very large models (1B+ parameters)
- Limited numerical precision experiments so far

TurboTensors is best viewed as a research and engineering exploration, not a production-ready engine.

---

## 👨‍💻 Author

Enes Altıparmak (sixfingerdev)  
Student — Kayseri Science High School

Interests:
- CPU inference optimization
- Turkish language models
- Tokenization and memory efficiency
- Low-level performance engineering

---

## 📌 Motivation

Understanding why systems are slow matters more than blindly using faster ones.
