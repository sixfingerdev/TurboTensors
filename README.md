# ⚡ TurboTensors v4.0 - JET MODE 🚀

**TurboTensors** is a high-performance inference engine built from scratch to eliminate the overhead and latency of standard CPU libraries. Specifically optimized for Turkish Large Language Models like **Kayra-1-exp**, it enables a "real-time" experience even on hardware without a dedicated GPU.

> "Designed for those tired of the heavy footprint and high resource consumption of HuggingFace, focusing purely on raw performance." — **sixfingerdev**

---

## ✨ Key Features

* **Numba-JIT Kernels:** Compiles Python logic into machine code via LLVM at runtime, delivering C++ level execution speeds.
* **Fused Operator Logic:** Combines operations like SiLU activation and RMSNorm at the memory level (Kernel Fusion) to minimize bandwidth bottlenecks between CPU and RAM.
* **Advanced KV Caching:** Implements distinct mathematical paths for Prefill (understanding) and Decode (token generation) phases to avoid redundant computations.
* **Zero-Copy Safetensors:** Processes model weights (BF16, F16, or F32) with manual bit-shifting and optimized memory management.
* **Parallel Execution:** Fully utilizes every physical CPU core using OMP_NUM_THREADS and MKL optimizations.

---

## 📊 Performance (Benchmark)

CPU test results using the **Kayra-1-exp (85M Params)** model:

| Engine             | Speed (Token/s) | RAM Efficiency | Latency (First Token) |
| :----------------- | :-------------- | :------------- | :-------------------- |
| **TurboTensors v4**| **~45-60 tok/s**| **High** | **Ultra Low** |
| HuggingFace (CPU)  | ~12-18 tok/s    | Moderate       | High                  |

*Note: Benchmarks were conducted on a standard consumer-grade laptop CPU.*

---

## 🛠️ Installation

Install the necessary dependencies via your terminal:

pip install numpy numba safetensors transformers huggingface_hub

## 🚀 Quick Start

Example usage scenario:

1. Load the model and initialize Jet Mode.
2. Start lightning-fast generation with streaming support.

Developer Note: Ensure 'use_cache=True' is passed in the forward method to maximize KV-Cache efficiency.

---

## 🧠 Engineering Insights

This project bypasses Python's dynamic overhead using the following techniques:

1. **Memory Contiguity:** Keeps matrices contiguous in memory to maximize CPU L1/L2 Cache hit rates.
2. **Fast Sampling:** Implements Top-K sampling without sorting all logits (partial sort), significantly reducing computational cost.
3. **Thread Optimization:** Utilizes 'nogil=True' to bypass Python's Global Interpreter Lock (GIL) for true parallel processing.

---

## 👨‍💻 Developer

**Enes Altıparmak (sixfingerdev)**
Student at Kayseri Science High School. Specializing in Turkish LLM architectures, tokenizer optimization, and low-level software performance.

**Goal:** To become Turkey's top AI engineer.

---
## 📜 License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. 
**Commercial use is strictly prohibited.**
