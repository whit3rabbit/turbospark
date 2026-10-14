"""Reference vectors for the Core ML EmbeddingGemma 2 host.

Runs the SAME FluidInference packages through coremltools on the CPU, with
host preprocessing written independently of the Rust port:

- text: `tokenizers` encode, then <bos> ... <eos>, bf16 table rows x sqrt(512)
- image: transformers `Gemma4ImageProcessorPil` (resize, rescale, patchify,
  positions), pooled per `Gemma4VisionPooler`
- audio: 160-sample left pad, frame mask per the model card

The Rust gate (`crates/runtime/tests/coreml_embeddinggemma2.rs`) loads the
package with CPU-only compute and compares cosines. This checks the Rust
marshalling, tokenization and preprocessing, not the export's quality
against PyTorch (the card reports that).

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
        --with transformers==5.18.0 --with pillow==12.3.0 \
        python -I crates/runtime/tools/ref_embeddinggemma2_coreml.py ~/models/coreml/embeddinggemma-2-coreml

Writes crates/runtime/testdata/coreml_embeddinggemma2/. Tests never run it.
"""

import hashlib
import json
import math
import os
import sys
import wave

import coremltools as ct
import numpy as np
import tokenizers
import transformers
from PIL import Image
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

OUT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "coreml_embeddinggemma2")
)
CPU = ct.ComputeUnit.CPU_ONLY

TEXTS = [
    ("The quick brown fox jumps over the lazy dog.", None),
    ("What is the capital of France?", "search_query"),
    ("Paris is the capital and most populous city of France.", "document"),
    ("Bananas are a good source of potassium.", "document"),
    # Long enough to need embed_128.
    (" ".join(["Lighthouses guided ships along dangerous coasts for centuries."] * 8), "document"),
]


def compiled(model_dir, package, function=None):
    path = os.path.join(model_dir, package)
    if function is None:
        return ct.models.MLModel(path, compute_units=CPU)
    return ct.models.MLModel(path, compute_units=CPU, function_name=function)


def table_rows(model_dir, ids, scale):
    table = np.memmap(os.path.join(model_dir, "embeddings.bf16"), dtype=np.uint16, mode="r").reshape(-1, 512)
    rows = table[np.asarray(ids)].astype(np.uint32) << 16
    return rows.view(np.float32) * np.float32(scale)


def run_text(model_dir, cfg, rows):
    """rows: [n, 512] float32 already scaled. Returns the 768-d output."""
    lengths = sorted(int(k.split("_")[1]) for k in cfg["functions"] if k.startswith("embed_"))
    n = rows.shape[0]
    length = next(l for l in lengths if l >= n)
    pad = table_rows(model_dir, [cfg["pad_token_id"]] * (length - n), cfg["embed_scale"])
    embeds = np.concatenate([rows, pad], axis=0)[None].astype(np.float16)
    mask = (np.arange(length) < n).astype(np.float16)[None]
    m = compiled(model_dir, "EmbeddingGemma2Text.mlpackage", f"embed_{length}")
    out = m.predict({"inputs_embeds": embeds, "attention_mask": mask})
    return np.asarray(out["embedding"], dtype=np.float32).reshape(-1), length


def text_case(model_dir, cfg, tok, text, task):
    prefix = cfg["prompts"][task] if task else ""
    body = tok.encode(prefix + text, add_special_tokens=False).ids
    ids = [cfg["bos_token_id"]] + body[:510] + [cfg["eos_token_id"]]
    rows = table_rows(model_dir, ids, cfg["embed_scale"])
    v, length = run_text(model_dir, cfg, rows)
    return {"text": text, "task": task, "ids": ids, "function": length, "embedding": v.tolist()}


def soft_case(model_dir, cfg, start, end, soft):
    rows = np.concatenate(
        [
            table_rows(model_dir, [cfg["bos_token_id"], start], cfg["embed_scale"]),
            soft.astype(np.float32),
            table_rows(model_dir, [end, cfg["eos_token_id"]], cfg["embed_scale"]),
        ]
    )
    return run_text(model_dir, cfg, rows)[0]


def synthetic_image():
    """Deterministic 200x150 RGB test pattern: gradients plus a disc."""
    w, h = 200, 150
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // (w - 1)).astype(np.uint8)
    g = (y * 255 // (h - 1)).astype(np.uint8)
    disc = ((x - 120) ** 2 + (y - 60) ** 2) < 35**2
    b = np.where(disc, 230, 40).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def image_case(model_dir, cfg, rgb, budget):
    proc = Gemma4ImageProcessorPil(max_soft_tokens=budget)
    feat = proc(images=[Image.fromarray(rgb)], return_tensors="np")
    patches = feat["pixel_values"][0].astype(np.float32)  # [N, 768], RGB in [0, 1]
    positions = feat["image_position_ids"][0]  # [N, 2], -1 padding
    n_soft = int(feat["num_soft_tokens_per_image"][0])
    n = patches.shape[0]
    valid = (positions[:, 0] >= 0).astype(np.float16)
    table = np.fromfile(os.path.join(model_dir, "position_embeddings.f16"), dtype=np.float16).reshape(2, 1024, 768)
    pe = np.zeros((n, 768), dtype=np.float32)
    real = positions[:, 0] >= 0
    pe[real] = table[0][positions[real, 0]].astype(np.float32) + table[1][positions[real, 1]].astype(np.float32)
    # Gemma4VisionPooler._avg_pool_by_positions, padding weighted zero.
    k = 3
    max_x = positions[real, 0].max() + 1
    kernel = positions[:, 0] // k + (max_x // k) * (positions[:, 1] // k)
    pool = np.zeros((budget, n), dtype=np.float32)
    pool[kernel[real], np.nonzero(real)[0]] = 1.0 / (k * k)
    m = compiled(model_dir, "EmbeddingGemma2Vision.mlpackage", f"vision_{budget}")
    out = m.predict(
        {
            "patches": patches[None].astype(np.float16),
            "positions": positions[None].astype(np.float16),
            "position_embeddings": pe[None].astype(np.float16),
            "valid": valid[None],
            "pool": pool.astype(np.float16),
        }
    )
    soft = np.asarray(out["image_tokens"], dtype=np.float32).reshape(budget, 512)[:n_soft]
    v = soft_case(model_dir, cfg, cfg["vision"]["boi_token_id"], cfg["vision"]["eoi_token_id"], soft)
    return {"budget": budget, "soft_tokens": n_soft, "embedding": v.tolist()}


def synthetic_audio():
    """3.5 s of a deterministic chirp plus a tone burst, 16 kHz int16."""
    sr = 16000
    t = np.arange(int(3.5 * sr)) / sr
    chirp = 0.3 * np.sin(2 * np.pi * (200 * t + 300 * t * t))
    burst = 0.2 * np.sin(2 * np.pi * 880 * t) * ((t > 1.0) & (t < 1.6))
    return np.round((chirp + burst) * 32767).astype(np.int16)


def audio_case(model_dir, cfg, pcm16):
    samples = pcm16.astype(np.float32) / 32768.0
    a = cfg["audio"]
    window, pad, frames = 160000, a["left_pad_samples"], a["frames"]
    count = min(len(samples), window)
    waveform = np.zeros((1, pad + window), dtype=np.float32)
    waveform[0, pad : pad + count] = samples[:count]
    real = pad + count
    valid_frames = 0 if real < 321 else min(frames, (real - 321) // 160 + 1)
    mask = (np.arange(frames) < valid_frames).astype(np.float16)[None]
    m = compiled(model_dir, "EmbeddingGemma2Audio.mlpackage")
    out = m.predict({"waveform": waveform, "frame_mask": mask})
    tokens = (valid_frames + 3) // 4
    soft = np.asarray(out["audio_tokens"], dtype=np.float32).reshape(-1, 512)[:tokens]
    v = soft_case(model_dir, cfg, a["boa_token_id"], a["eoa_token_id"], soft)
    return {"tokens": tokens, "embedding": v.tolist()}


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    model_dir = os.path.expanduser(sys.argv[1])
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tok = tokenizers.Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    os.makedirs(OUT, exist_ok=True)

    texts = [text_case(model_dir, cfg, tok, t, task) for t, task in TEXTS]

    rgb = synthetic_image()
    Image.fromarray(rgb).save(os.path.join(OUT, "pattern.png"))
    images = [image_case(model_dir, cfg, rgb, b) for b in (70, 280)]

    pcm16 = synthetic_audio()
    wav_path = os.path.join(OUT, "chirp.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16.tobytes())
    audio = [audio_case(model_dir, cfg, pcm16)]

    with open(os.path.join(OUT, "reference.json"), "w") as fh:
        json.dump({"texts": texts, "images": images, "audio": audio}, fh)
    manifest = {
        "generator": "crates/runtime/tools/ref_embeddinggemma2_coreml.py",
        "model": "FluidInference/embeddinggemma-2-coreml@3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4",
        "compute_units": "CPU_ONLY",
        "python": sys.version.split()[0],
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "files": {f: sha(os.path.join(OUT, f)) for f in ("reference.json", "pattern.png", "chirp.wav")},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
