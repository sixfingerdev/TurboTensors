"""
Turkish Wikipedia (wikimedia/wikipedia 20231101.tr) -> byte-level BPE (8k) -> uint16 token files.
Validation/test are held out BY ARTICLE (every 100th article -> val, every 100th offset by 50 -> test),
so no article is split across sets.

    python -I prepare_data.py <parquet> <outdir> [--max-train-tokens 60000000]
"""
import argparse, os, sys
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("outdir")
ap.add_argument("--vocab", type=int, default=8192)
ap.add_argument("--max-train-tokens", type=int, default=60_000_000)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

pf = pq.ParquetFile(a.parquet)
train_txt, val_txt, test_txt = [], [], []
i = 0
for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
    for t in batch.column("text").to_pylist():
        (val_txt if i % 100 == 0 else test_txt if i % 100 == 50 else train_txt).append(t)
        i += 1
print(f"articles {i}: train {len(train_txt)} val {len(val_txt)} test {len(test_txt)}", flush=True)

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<eos>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(train_txt[:40000], tr)          # tokenizer learned on train only
tok.save(os.path.join(a.outdir, "tokenizer.json"))
eos = tok.token_to_id("<eos>")

def encode(texts, limit=None):
    out, n = [], 0
    for k in range(0, len(texts), 500):
        for e in tok.encode_batch(texts[k:k + 500]):
            out.append(np.array(e.ids + [eos], dtype=np.uint16)); n += len(e.ids) + 1
        if limit and n >= limit: break
    return np.concatenate(out)

for name, texts, lim in (("val", val_txt, None), ("test", test_txt, None), ("train", train_txt, a.max_train_tokens)):
    ids = encode(texts, lim); ids.tofile(os.path.join(a.outdir, f"{name}.bin"))
    chars = sum(len(t) for t in texts[: None]) if name != "train" else None
    print(f"{name}: {len(ids):,} tokens", flush=True)
