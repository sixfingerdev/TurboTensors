# constmem — constant-memory decode experiment (Tessera × TurboTensors)

**Question:** TurboTensors' decode step reads the whole KV cache, so cost and memory grow with context.
Tessera's COMPASS layer keeps a fixed `Ds × Dv` delta-rule state per head instead. What does moving that
recurrence onto TurboTensors' Numba/CPU stack buy, and what does it cost?

**Scope (honest):** part 1-2 below are the bare recurrence at 1 layer / 10 heads / head dim 64. Part 3 is a *real
trained* COMPASS model (`facts4`, 2 layers, 111k params, 100% in-distribution) ported end to end and run
token by token. It is a synthetic fact-recall task, **not** a language model, and nothing here is a text-quality claim.
All numbers: 4-vCPU cloud container, not consumer hardware, small samples (query counts given).

## Files
| File | What |
|---|---|
| `delta_state.py` | `delta_step` (decode), `delta_scan` (prefill), `DeltaState` — in-place Numba kernels |
| `test_parity.py` | Matches the torch loop copied from `tessera/compass.py` (needs `torch`, test only). Max diff ≈ 6e-6 (decode and prefill) |
| `bench_context.py` | Per-token time and memory vs context length against TurboTensors' own attention-decode kernels |
| `capacity.py` | Recall accuracy of the state vs number of stored bindings (random unit keys, no learning) |
| `compass_stream.py` | Full trained COMPASS model (router, product-binding keys, FFN, LayerNorm) in numpy + the Numba kernel, one token at a time |
| `test_stream_parity.py` | Streaming port vs torch model on the checkpoint: max logit diff 2.1e-5, 1024/1024 argmax agree |
| `eval_stream.py` | Streaming accuracy / tokens/s at long lengths, optional `--decay` override |
| `checkpoints/facts4_L128_s0.pt` | The trained model (`python compass.py facts4 128 0 3000 64 1 8`, 901 s on this CPU) |

```bash
cd constmem
python test_parity.py
python bench_context.py
python capacity.py
CKPT=checkpoints/facts4_L128_s0_nodecay.pt python test_stream_parity.py   # CKPT selects the checkpoint; needs ../../tessera next to this repo, and torch
python eval_stream.py --decay 1.0 --lens 1024 4096 16384 65536
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

## 3. A trained model, streamed (facts4, trained at L=128)
Per-layer state = one 64x64 matrix + a 24-token input ring buffer; whole model **56 KB**, independent of length.
Speed is ~3,000 tokens/s, limited by numpy/Python overhead per token, not by the state update (0.005 ms).

| Length | Learned decay (0.998) | Decay forced to 1.0 at inference |
|---:|---:|---:|
| 1,024 | 91.7% (48 queries) | 100% (24 queries) |
| 4,096 | 39.6% (48 queries) | 100% (24 queries) |
| 16,384 | - | 100% (24 queries) |
| 65,536 | - | 100% (24 queries) |

Trained without decay from the start (`COMPASS_NO_DECAY=1 python compass.py facts4 128 0 3000 64 1 8`, new flag in the
Tessera repo, `checkpoints/facts4_L128_s0_nodecay.pt`, 100% in-distribution, 850 s): no train/test mismatch.

| Length | 128 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---:|---:|---:|---:|---:|
| No-decay model (24 queries each) | 100% | 100% | 100% | 100% | 95.8% |

Parity vs torch for this checkpoint: max logit diff 1.6e-5. The one miss at 65k (1 of 24) is the only error seen; with so few
queries it is not enough to say whether accumulated write-gate leakage starts to bite there.

Reading, for the first checkpoint: the model forgets because training taught it a per-step decay of ~0.998 (memory horizon of a few hundred
tokens: 0.998^4096 is ~3e-4). Removing the decay at inference, a train/test mismatch used here as an **ablation**,
keeps recall at 100% to 65k tokens (512x the training length) with 4 stored facts. That load is far below the capacity knee
of part 2, so this does **not** show that a decay-free model would survive heavy load; the no-decay run above is that follow-up for 4 facts; heavier load at length is still untested. Chance level on this task is 12.5-25%.

Cost of the same layer shape (1 head, D=64) as a KV-cache decode step, per layer:

| Context | KV numpy | KV Numba | Delta state | KV cache |
|---:|---:|---:|---:|---:|
| 16,384 | 1.8 ms | 7.9 ms | 0.005 ms | 8 MB |
| 65,536 | 7.6 ms | 28.7 ms | 0.005 ms | 32 MB |
| 131,072 | 20.2 ms | 59.9 ms | 0.005 ms | 64 MB |

### Found while porting (Tessera code, behaviour kept as-is)
- `router()`: for valid offsets the recency bias is overwritten by the assignment `Smat[:, o:, o] = ...`, so it only
  survives on invalid offsets.
- `prod_pools()`: the salience score is the current token's, repeated across all offsets, so the softmax cancels it.
  `uA`/`uB` receive no gradient and `PA == PB`; the "role seeking" is really a fixed recency average.
Both are worth fixing at the source; the port reproduces them so parity with the checkpoint holds.

## Not shown / next steps
- Natural-language quality of COMPASS layers is unmeasured; only a synthetic recall task was trained.
- No tokens/s against a trained KV-cache model on the same task (kernel-level costs only, above).
- Not integrated into `TurboLLM`/`TransformerBlock` (needs a trained hybrid model to be meaningful).
- Fix the two quirks above in Tessera and retrain; test more queries per length and higher fact load (`facts8`) at 16k-65k.
