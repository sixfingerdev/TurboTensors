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

## 🛠️ Installation

Dependencies:
- numpy
- numba
- safetensors
- transformers
- huggingface_hub

---

## 🚀 Usage (Conceptual)

1. Load model weights
2. Enable Jet Mode
3. Generate tokens with streaming support

Developer note:
Enable use_cache=True to fully benefit from KV caching.

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
