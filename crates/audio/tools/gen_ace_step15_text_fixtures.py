#!/usr/bin/env python3
"""Write a real-checkpoint reference for the ACE-Step 1.5 text encoder.

Normative reference: transformers 4.57.6 `AutoModel` on
`Qwen3-Embedding-0.6B/` from `ACE-Step/Ace-Step1.5` at revision
19671f406d603126926c1b7e2adc169acbcade22, as upstream
`conditioning_embed.py` calls it: the caption prompt through the full
model (`last_hidden_state`), lyrics through `embed_tokens` only. CPU,
float32.

    ../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_text_fixtures.py \
        <checkpoint>/Qwen3-Embedding-0.6B <out dir>

Writes caption_ids.npy / lyric_ids.npy (int32), caption_hidden.npy and
lyric_embeds.npy (f32) and a manifest to <out dir>, outside the
repository. The prompt follows upstream's SFT_GEN_PROMPT shape so the
special tokens are exercised; exact prompt assembly is ported later.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

PROMPT = (
    "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n"
    "# Caption\norchestral strings, solo cello, slow and warm, classical\n\n"
    "# Metas\n- bpm: 72\n- timesignature: 4\n- keyscale: D minor\n- duration: 30 seconds\n<|endoftext|>\n"
)
LYRICS = "# Languages\nen\n\n# Lyric\n[Instrumental]<|endoftext|>"


def main(model_dir: Path, out: Path) -> None:
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 58):
        sys.exit(f"transformers {transformers.__version__}: upstream requires <4.58")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32).eval()
    inv = model.rotary_emb.inv_freq
    if not bool((inv[1:] != 0).all()):
        sys.exit("rotary inv_freq has zero entries")
    cap = tok(PROMPT, return_tensors="pt", padding="longest", truncation=True, max_length=256)
    lyr = tok(LYRICS, return_tensors="pt", padding="longest", truncation=True, max_length=2048)
    with torch.inference_mode():
        hidden = model(input_ids=cap.input_ids).last_hidden_state[0]
        lyric = model.embed_tokens(lyr.input_ids)[0]
    arrays = {
        "caption_ids": cap.input_ids[0].numpy().astype(np.int32),
        "lyric_ids": lyr.input_ids[0].numpy().astype(np.int32),
        "caption_hidden": hidden.numpy().astype(np.float32),
        "lyric_embeds": lyric.numpy().astype(np.float32),
    }
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", np.ascontiguousarray(v))
    manifest = {
        "generator": "tools/gen_ace_step15_text_fixtures.py",
        "model_dir": str(model_dir),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation": model.config._attn_implementation,
        "files": {f"{k}.npy": hashlib.sha256((out / f"{k}.npy").read_bytes()).hexdigest() for k in arrays},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"caption {arrays['caption_ids'].size} tokens, lyrics {arrays['lyric_ids'].size} tokens -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gen_ace_step15_text_fixtures.py <Qwen3-Embedding dir> <out dir>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
