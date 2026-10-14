#!/usr/bin/env python3
"""Compare every row of the eg2demo word table with the stock
EmbeddingGemma 2 text embedding of its word.

Usage: probe_embeddinggemma2_word_table.py MODEL_DIR ASSET_DIR [PREFIX]

MODEL_DIR is google/embeddinggemma-2 @ 914f7f89142e33e77833254d9c9b90c3cef7303b
(model.safetensors sha256 197a3296...5553d79). ASSET_DIR holds reader_words.json
and word_unit_f16.bin from the eg2demo v5 manifest. PREFIX defaults to no
prompt, the only one that matches. Needs torch, sentence-transformers >= 6.1,
pillow, and the pinned transformers commit. The model runs in float32: the
checkpoint's bfloat16 default adds about 0.4% norm error. CPU time is about
25 minutes on 4 cores.
"""

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    model_dir, asset_dir = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    words = json.load(open(f"{asset_dir}/reader_words.json"))
    table = np.memmap(f"{asset_dir}/word_unit_f16.bin", dtype="<f2", mode="r").reshape(len(words), 768)
    model = SentenceTransformer(model_dir, device="cpu", model_kwargs={"dtype": torch.float32})
    cos = np.zeros(len(words), np.float32)
    for start in range(0, len(words), 4096):
        ids = np.arange(start, min(start + 4096, len(words)))
        emb = model.encode([prefix + words[i] for i in ids], batch_size=256, normalize_embeddings=True)
        rows = np.asarray(table[ids], np.float32)
        cos[ids] = (emb * rows).sum(1) / np.linalg.norm(rows, axis=1)
    body = cos[:-1]
    print("quantiles 0/0.01%/0.1%/1%/50%:", np.round(np.quantile(body, [0, 1e-4, 1e-3, 1e-2, 0.5]), 6))
    off = np.where(body < 0.999)[0]
    print(f"rows below 0.999: {len(off)} ({(off < 320).sum()} in ids 0..319)")
    positions = defaultdict(list)
    for index, word in enumerate(words):
        positions[word].append(index)
    duplicates = [ids for ids in positions.values() if len(ids) > 1]
    print(f"duplicate strings: {len(duplicates)}")
    print(f"last entry {words[-1]!r}: cos {cos[-1]:.4f}")


if __name__ == "__main__":
    main()
