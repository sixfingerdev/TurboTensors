# A Turkish language model on this stack (constmem/lm)

Earlier parts of `constmem/` used synthetic fact-recall. This directory trains actual language models on
**Turkish Wikipedia** (`wikimedia/wikipedia 20231101.tr`, first shard), 8k byte-level BPE, 60M training tokens.
Validation and test are **held-out articles** (every 100th / every 100th+50th article), not a split inside articles.
Everything ran on a 4-vCPU CPU container with bf16 autocast. Single seed, so differences of a few percent are not conclusive.

## What was built
| File | |
|---|---|
| `prepare_data.py` | parquet -> BPE tokenizer (trained on train only) -> `uint16` token files |
| `models.py` | `A` = causal attention (rotary), `G` = gated delta-rule layer, any layer string e.g. `GGAGGA` |
| `train_lm.py` | identical data order / schedule / eval for every variant; `--resume` |
| `eval_length.py`, `bench_decode.py`, `sample.py`, `stream_lm.py`, `test_*.py` | evaluation, decode benchmark, sampling, constant-memory engine, parity tests |
| `results/` | logs + JSON of every run, samples, benchmarks |
| `../checkpoints/lm/` | fp16 weights of the GDN and hybrid models + tokenizer (fp16 changes ppl by <0.03%) |

The `G` layer is COMPASS's delta rule made LM-ready: learned q/k/v with a short causal conv, data-dependent write
strength and decay, per-head output norm and gate, and an exact chunkwise-parallel form for training. **This is Gated
DeltaNet (Yang et al., 2024), prior art, not a new architecture.** Checks: chunkwise vs sequential 3.6e-7;
prefill+decode vs full forward 6e-7; numpy/Numba stream engine vs torch on real text 1.8e-5 (300/300 argmax).

## Step 1: does the original COMPASS work as an LM? (4.1M tokens, same everything)
| Model | Params | val loss @1M tokens | val ppl @4.1M | test ppl @4.1M |
|---|---:|---:|---:|---:|
| Transformer | 6.9M | 5.989 | 125.8 | 127.6 |
| Original COMPASS (`compass.py`, unchanged) | 9.3M | 5.928 | not run | not run |
| Gated delta (`GGGGGG`) | 7.3M | 5.905 | 104.4 | 106.3 |
| Hybrid (`GGAGGA`) | 7.1M | 5.987 | 113.1 | 115.2 |

Original COMPASS is competitive per token early on but its Python time loop and fact-binding router make it ~8x
slower (450 vs 3,500+ tok/s), so it could only be run to 1M tokens. That is why the delta layer was generalised.

## Step 2: 40M tokens (4,882 steps x 8,192 tokens)
| Model | Params | val ppl | **test ppl** | test bits/byte | train time |
|---|---:|---:|---:|---:|---:|
| Transformer (`AAAAAA`) | 6.9M | 33.4 | 36.5 | 1.331 | 83 min |
| Gated delta (`GGGGGG`) | 7.3M | 32.8 | 36.4 | 1.329 | 90 min |
| **Hybrid (`GGAGGA`)** | 7.1M | **31.8** | **34.9** | **1.314** | 131 min |

- The delta layer's 4M-token advantage (106 vs 128) **disappeared** with more training: at 40M it ties the Transformer.
  At this scale the delta layer is not a quality win; its win is memory and decode cost (below).
- The hybrid (2 attention layers of 6) is best, ~4% lower perplexity than either pure model. Single seed.
- The GDN run was interrupted by a container restart at step ~2150 and resumed from the step-2000 checkpoint
  (Adam moments restarted from zero, new data order); a small discontinuity is possible.

## Step 3: context longer than training (all trained on 256-token windows)
Loss on the **same** 256 target tokens, with C extra tokens of preceding text as context (24 targets):
| Model | C=0 | C=256 | C=768 | C=3,840 |
|---|---:|---:|---:|---:|
| Transformer | 3.757 | 4.752 | 5.403 | 5.789 |
| Gated delta | 3.723 | 3.644 | 3.643 | 3.644 |
| Hybrid | 3.697 | 3.618 | 3.705 | 3.934 |

Rotary attention trained at 256 does not extrapolate: more context makes it much *worse*. The delta layers use the
context and stay flat out to 3,840 tokens (no further gain either; they were never trained on longer). This tests
extrapolation, not what a Transformer trained at 4k would do.

## Step 4: decode cost per token and generation-state memory (real models, torch fp32 unless noted)
| Tokens in context | Transformer | Hybrid | Gated delta (torch) | Gated delta (numpy+Numba) |
|---:|---:|---:|---:|---:|
| 256 | 3.5 ms, 3.2 MB | 6.7 ms, 1.3 MB | 6.3 ms, 0.43 MB | 2.6 ms |
| 4,096 | 8.5 ms, 48 MB | 7.4 ms, 16 MB | 6.8 ms, 0.43 MB | 2.7 ms |
| 16,384 | 33 ms, 192 MB | 17 ms, 64 MB | 6.0 ms, 0.43 MB | 2.7 ms |
| 65,536 | 211 ms, 768 MB | 66 ms, 256 MB | 4.9 ms, 0.43 MB | 2.4 ms |
| 131,072 | 395 ms, 1.5 GB | 137 ms, 512 MB | 5.6 ms, 0.43 MB | 2.5 ms |

KV rows are filled with random K/V (timing only). The numpy+Numba engine (`stream_lm.py`, ~400 tok/s, 438 KB state)
is faster than torch for single-token steps only after a fix found while benchmarking: multi-threaded BLAS plus a
Numba thread pool oversubscribed 4 cores and made it 40 ms/token; one BLAS thread gives 2.5 ms. The hybrid keeps a KV cache for
its attention layers, so its memory still grows (at a third of the Transformer's rate).

## Samples (T=0.7, top-k 40, same seed; `results/long/samples.txt`)
> **gdn** İstanbul, Türkiye'nin en kalabalık şehridir. / Tarihçe / Mahallenin eski ismi Sapanca'dır. Daha sonra köye adını veren köyde; 2,750 kişi yaşamaktadır. ...

Fluent, Wikipedia-shaped Turkish (headings, "Nüfus", "Dış bağlantılar") but **factually unreliable**: it states
nonsense with confidence, and the hybrid sample degenerates into repeating "Bursa, İstanbul". The corpus is
dominated by village/district stubs and so are the samples. Perplexity ~35 on this data is a small-scale result;
this is not a usable assistant or fact source.

## Not done / next
- More tokens and parameters (the remaining ~85M tokens in the shard, a ~20M-parameter model): hours of CPU time.
- Multiple seeds; the hybrid-vs-pure gap (~1.5 ppl) is only one run.
- Training on longer windows to see whether the delta layers then gain from long context.
- Wiring the hybrid into `stream_lm.py` / `TurboLLM` (needs a KV cache for its attention layers).
