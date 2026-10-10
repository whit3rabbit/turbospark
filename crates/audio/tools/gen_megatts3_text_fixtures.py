#!/usr/bin/env python3
"""Regenerate the MegaTTS3 text-frontend and G2P Rust parity fixtures.

Normative reference: the unmodified MegaTTS3 code
(`tts/utils/text_utils/split_text.py`, `g2p()` in `tts/frontend_function.py`
from github.com/bytedance/MegaTTS3, Apache-2.0; the Hugging Face Space
mrfakename/MegaTTS3-Voice-Cloning carries the same files at revision
8c2b5ab2ac67affda57a649fe0637425e8e1de58).

Three kinds of fixture:
- chunks: a corpus pushed through the reference `chunk_text_english`;
- tiny G2P: a seeded tiny Qwen2 (HF transformers) with a small byte-level BPE
  tokenizer carrying the same added tokens as the real one, with `generate`
  results for the exact call the reference makes (`do_sample=True, top_k=1`);
- real G2P (optional, never committed): the real checkpoint through the
  reference's own `g2p()`.

The English normalizer has no fixture: WeTextProcessing needs pynini, which has
no macOS wheel, so its output could not be generated.

Needs torch, transformers, tokenizers, safetensors in a scratch venv:

    python gen_megatts3_text_fixtures.py --ref DIR --out crates/audio/testdata/megatts3_text
    python gen_megatts3_text_fixtures.py ... --real-g2p DIR --real-out DIR

Deterministic: fixed seeds, no network access.
"""

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch

CHUNK_CASES = [
    "Hello world.",
    "",
    "   ",
    "One sentence without terminal punctuation",
    "First. Second! Third? Fourth; fifth: sixth, seventh.",
    "A very long sentence that keeps going and going without any punctuation at all so that the chunker cannot split it anywhere and must emit it whole even though it is far longer than the limit",
    "Washington CNN When President Donald Trump declared in the House Chamber this week that executives at the nation\u2019s top automakers were \u201cso excited\u201d about their prospects amid his new tariff regime, it did not entirely reflect the conversation he\u2019d held with them earlier that day.",
    "Short one. " * 30,
    "Line one.\nLine two.\n\nLine three!   Line four?\tTabbed.",
    "Trailing space after stop. ",
    "Ends with a curly quote \u201cfinished.\u201d",
    "Ends with a wide stop\u3002 Then more text follows here, and more, and more, and more, and more, and more, and more, and more.",
    "Wide\uff0cmid\uff1bsentence\uff1atokens\uff01and\uff1fmore\u3002",
    "caf\u00e9, na\u00efve; r\u00e9sum\u00e9: \u00fcber! Mixed accents, again, and again, and again, and again, and again, and again, and again, and again.",
    "a." * 80,
    "x" * 129 + ". y",
    "x" * 130 + ". y",
    "Dr. Smith went to Washington. He arrived at 5 p.m. on Tuesday, and left the next day; nobody noticed, which was the point.",
    "Ellipsis... and more... and even more... until the limit is reached, finally, after a very long run of words that goes on.",
]
MAX_CHARS = [130, 20, 40, 300]

G2P_TEXTS = [
    "Hello world.",
    "Voice cloning from a short sample.",
    "<BOT>nested markers<BOS> inside",
    "\u00fcnicode caf\u00e9 \u201cquotes\u201d",
    "",
    "multiple   spaces\tand\nnewlines",
]
REAL_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Voice cloning needs a short, clean sample of the speaker.",
    "She sells sea shells by the sea shore, doesn't she?",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "Okay, so, um, this is a longer sentence; it has several clauses: one, two, three!",
]


def save_json(out, name, obj):
    with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=True, sort_keys=True)


def chunk_fixtures(out, ref):
    sys.path.insert(0, ref)
    from tts.utils.text_utils.split_text import chunk_text_english

    cases = []
    for text in CHUNK_CASES:
        for m in MAX_CHARS:
            cases.append({"text": text, "max_chars": m, "chunks": chunk_text_english(text, max_chars=m)})
    save_json(out, "chunks.json", cases)
    return len(cases)


def build_tiny_tokenizer(path):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "hello world voice cloning from a short sample the quick brown fox",
        "multiple spaces and newlines unicode cafe quotes nested markers inside",
    ] * 20
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tok.train_from_iterator(corpus, trainer)
    # The real tokenizer carries <BOT>, <BOS> and a block of <Reserved_TTS_n> as added tokens.
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in ["<|endoftext|>", "<BOT>", "<BOS>"]])
    tok.add_tokens([AddedToken(f"<Reserved_TTS_{i}>", special=False, normalized=False) for i in range(0, 4)])
    tok.save(path)
    return Tokenizer.from_file(path)


def tiny_g2p_fixtures(out):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    d = os.path.join(out, "tiny_g2p")
    os.makedirs(d, exist_ok=True)
    tok = build_tiny_tokenizer(os.path.join(d, "tokenizer.json"))
    ssi = tok.encode("<Reserved_TTS_0>").ids[0]
    vocab = ssi + 805  # room for speech tokens up to speech_start + 801
    torch.manual_seed(7)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, max_position_embeddings=512, use_sliding_window=False,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=gen))
    model.to(torch.float32).save_pretrained(d, safe_serialization=True)

    cases = []
    for text in G2P_TEXTS:
        txt_ids = tok.encode("<BOT>" + text + "<BOS>").ids
        inp = txt_ids + [145 + ssi]
        ids = torch.LongTensor([inp])
        # Greedy margin check so argmax cannot flip on rounding.
        cur, min_margin = ids, 1e9
        with torch.no_grad():
            for _ in range(10):
                logits = model(cur).logits[0, -1]
                top2 = torch.topk(logits, 2).values
                min_margin = min(min_margin, float(top2[0] - top2[1]))
                cur = torch.cat([cur, logits.argmax()[None, None]], dim=1)
        assert min_margin > 1e-3, (text, min_margin)
        first_new = int(cur[0, len(inp) + 2])  # third generated token: a real EOS to stop on
        for eos, max_new in ((800 + 1 + ssi, 10), (first_new, 10)):
            with torch.no_grad():
                out_ids = model.generate(ids, max_new_tokens=max_new, do_sample=True, top_k=1, eos_token_id=eos, pad_token_id=0)
            cases.append({"text": text, "txt_ids": txt_ids, "input_ids": inp, "eos": eos, "max_new": max_new,
                          "output": out_ids[0].tolist()})
    save_json(out, "tiny_g2p_cases.json", {"speech_start": ssi, "vocab": vocab, "cases": cases})
    return len(cases)


def real_g2p(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.modules.setdefault("whisper", types.ModuleType("whisper"))
    sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    from tts.frontend_function import g2p

    tok = AutoTokenizer.from_pretrained(args.real_g2p, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(args.real_g2p, dtype=torch.float32).eval()
    me = types.SimpleNamespace(g2p_tokenizer=tok, g2p_model=model, device="cpu", precision=torch.float32,
                               speech_start_idx=tok.encode("<Reserved_TTS_0>")[0])
    results = []
    for text in REAL_SENTENCES:
        with torch.no_grad():
            ph, tone = g2p(me, text)
        results.append({"text": text, "txt_ids": tok("<BOT>" + text + "<BOS>")["input_ids"],
                        "ph": ph[0].tolist(), "tone": tone[0].tolist()})
        print(len(results[-1]["txt_ids"]), "text tokens ->", len(results[-1]["ph"]), "phones")
    os.makedirs(args.real_out, exist_ok=True)
    save_json(args.real_out, "g2p_real.json", {"speech_start": me.speech_start_idx, "results": results})


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-g2p")
    ap.add_argument("--real-out")
    args = ap.parse_args()
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, *a, **k: x))
    os.makedirs(args.out, exist_ok=True)
    n_chunks = chunk_fixtures(args.out, args.ref)
    n_g2p = tiny_g2p_fixtures(args.out)
    files = {}
    for root, _, names in os.walk(args.out):
        for f in sorted(names):
            if f != "manifest.json":
                files[os.path.relpath(os.path.join(root, f), args.out)] = sha256(os.path.join(root, f))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"reference": "bytedance/MegaTTS3 split_text.py + frontend_function.g2p", "torch": torch.__version__,
                   "chunk_cases": n_chunks, "g2p_cases": n_g2p, "files": files}, fh, indent=2, sort_keys=True)
    print("wrote", args.out, n_chunks, "chunk cases", n_g2p, "g2p cases")
    if args.real_g2p:
        sys.path.insert(0, args.ref)
        real_g2p(args)


if __name__ == "__main__":
    main()
