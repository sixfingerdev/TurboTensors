# constmem — constant-memory decode experiment (Tessera × TurboTensors)

**Question:** TurboTensors' decode step reads the whole KV cache, so cost and memory grow with context.
Tessera's COMPASS layer keeps a fixed `Ds × Dv` delta-rule state per head instead. What does moving that
recurrence onto TurboTensors' Numba/CPU stack buy, and what does it cost?

**Scope (honest):** this is the *inference recurrence only*. The COMPASS router / key / query projections
are not ported, no language model was trained or run, and nothing here is a quality claim about text generation.
Toy shapes: 1 layer, 10 heads, head dim 64, B=1, measured in a 4-vCPU cloud container (not consumer hardware).

## Files
| File | What |
|---|---|
| `delta_state.py` | `delta_step` (decode), `delta_scan` (prefill), `DeltaState` — in-place Numba kernels |
| `test_parity.py` | Matches the torch loop copied from `tessera/compass.py` (needs `torch`, test only). Max diff ≈ 6e-6 (decode and prefill) |
| `bench_context.py` | Per-token time and memory vs context length against TurboTensors' own attention-decode kernels |
| `capacity.py` | Recall accuracy of the state vs number of stored bindings (random unit keys, no learning) |

```bash
cd constmem
python test_parity.py
python bench_context.py
python capacity.py
```

## 1. Cost per decoded token (what you gain)
| Context N | KV decode, numpy path | KV decode, Numba kernel | Delta state | KV cache | State |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.093 ms | 0.039 ms | 0.010 ms | 0.3 MB | 0.16 MB |
| 1,024 | 1.06 ms | 1.11 ms | 0.010 ms | 5 MB | 0.16 MB |
| 4,096 | 3.8 ms | 4.5 ms | 0.010 ms | 20 MB | 0.16 MB |
| 16,384 | 28 ms | 74 ms | 0.010 ms | 80 MB | 0.16 MB |
| 32,768 | 56 ms | 156 ms | 0.010 ms | 160 MB | 0.16 MB |

Per layer. Multiply memory by layer count. Side finding: the repo's `fast_attention_decode` Numba kernel is
slower than the numpy path beyond ~16k tokens (the engine already uses numpy for decode, so no regression).

## 2. What it costs: recall capacity (D = 64, 256 value codes, chance 0.4%)
| Stored bindings M | 0.5×D | 0.75×D | 1×D | 1.5×D | 2×D | 4×D |
|---|---:|---:|---:|---:|---:|---:|
| Recall | 99.9% | 97.0% | 90.6% | 69.7% | 52.8% | 26.3% |

At D = 128 the same loads give 100 / 99.7 / 97.2 / 79.8 / 62.5 / 31.7 %. A KV cache is exact at every M.
So the trade is *exact + linear memory* vs *approximate + constant memory*, with the knee at M ≈ D.
This is the same wall Tessera's README documents; it is not removed here.

## Not shown / next steps
- No end-to-end tokens/s (no trained model uses this layer yet).
- Whether a model trained with COMPASS layers keeps useful quality at natural-language load is unmeasured.
- Candidate next step: export a trained `compass.py facts4` checkpoint and check that this recurrence reproduces its accuracy.
