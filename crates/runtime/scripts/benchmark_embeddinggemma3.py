#!/usr/bin/env python3
"""EmbeddingGemma 300m (Gemma 3) on MLX vs Core ML vs the CPU port.

Every arm embeds the SAME token ids (tokenized once by the CPU port with the
`document` prompt), one text at a time, so the timing isolates the model.
Each sample is a fresh process: open, one discarded warmup pass over all
texts, one measured pass. Peak `phys_footprint` comes from `/usr/bin/time
-l`. Arm order rotates every run. Every sample records load and power;
`quiet_host` is false when any sample saw a 1-minute load of 4 or more or
battery power (docs/BENCHMARKING.md, Measurement hygiene).

Arms:
  cpu          crates/embeddings EmbeddingGemma, f32 on the CPU (bf16 MLX
               checkpoint dequantized at load)
  mlx          mlx-embeddings on the same bf16 checkpoint (GPU, bf16)
  coreml_ane / coreml_gpu / coreml_cpu / coreml_all
               a pinned community Core ML export through turbospark-coreml.
               Several exports run in one session with labeled dirs:
               `--coreml-dir a=<dir> --coreml-dir b=<dir>` and arms
               `coreml_ane@a coreml_ane@b ...`. A bare dir has no label.

Agreement: cosine of each arm's vectors to the `cpu` arm's, per text.

    cargo build --release -p turbospark-runtime --example embeddinggemma3_probe
    uv run --python 3.12 --with mlx-embeddings==0.1.0 --with mlx-lm==0.32.0 \\
        python crates/runtime/scripts/benchmark_embeddinggemma3.py \\
        --mlx-dir <mlx-community/embeddinggemma-300m-bf16> \\
        --coreml-dir <shirochenkov90/embeddinggemma-300m-coreml> --runs 3 --output out.json
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BINARY = Path("target/release/examples/embeddinggemma3_probe")
PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE)


def texts():
    """32 short sentences and 8 of about 100 tokens (same set as the
    EmbeddingGemma 2 probe)."""
    short = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "Bananas are a good source of potassium.",
    ]
    long = ("Lighthouses guided ships along dangerous coasts for centuries, "
            "and their keepers tended oil lamps through long winter nights. ")
    return [f"{short[i % 4]} ({i})" for i in range(32)] + [f"{long * 4}({i})" for i in range(8)]


def mlx_arm(model_dir, ids_path, out_path):
    """Runs inside a fresh process: open, warmup, measure."""
    import mlx.core as mx
    from mlx_embeddings.utils import load

    ids = json.load(open(ids_path))
    t = time.perf_counter()
    model, _ = load(model_dir)
    mx.eval(model.parameters())
    open_ms = (time.perf_counter() - t) * 1e3

    def run():
        out = []
        for row in ids:
            x = mx.array([row])
            e = model(x, mx.ones(x.shape)).text_embeds
            mx.eval(e)
            out.append(e[0].astype(mx.float32).tolist())
        return out

    t = time.perf_counter()
    run()
    warmup_ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    v = run()
    measured_ms = (time.perf_counter() - t) * 1e3
    json.dump({"arm": "mlx", "open_ms": open_ms, "warmup_ms": warmup_ms,
               "measured_ms": measured_ms, "texts": len(v), "embeddings": v}, open(out_path, "w"))


def power():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "ac" if "AC Power" in out else "battery"


def sample(arm, args, ids_path, tmp):
    out = Path(tmp) / f"{arm}.json"
    if arm == "mlx":
        cmd = [sys.executable, __file__, "--mlx-child", str(args.mlx_dir), ids_path, str(out)]
    elif arm == "cpu":
        cmd = [str(BINARY), "cpu", str(args.mlx_dir), ids_path, str(out)]
    else:
        units, _, label = arm.partition("@")
        if label not in args.coreml_dirs:
            raise RuntimeError(f"{arm}: no --coreml-dir labeled {label!r}")
        cmd = [str(BINARY), units, str(args.coreml_dirs[label]), ids_path, str(out)]
    load = os.getloadavg()[0]
    r = subprocess.run(["/usr/bin/time", "-l", *cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{arm} exited {r.returncode}:\n{r.stdout}{r.stderr}")
    s = json.load(open(out))
    s["arm"] = arm  # the probe reports the arm without its export label
    s["peak_phys_footprint_bytes"] = int(PEAK.search(r.stderr).group(1))
    s["load_avg_1m_before"] = round(load, 2)
    s["power"] = power()
    s["texts_per_s"] = s["texts"] / (s["measured_ms"] / 1e3)
    return s


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--mlx-child":
        mlx_arm(*sys.argv[2:])
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mlx-dir", type=Path, required=True)
    p.add_argument("--coreml-dir", action="append", required=True,
                   help="a Core ML export dir, or label=dir (repeatable)")
    p.add_argument("--arms", nargs="+", default=["cpu", "mlx", "coreml_ane", "coreml_gpu", "coreml_cpu"])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    a.coreml_dirs = {}
    for d in a.coreml_dir:
        label, sep, path = d.partition("=")
        a.coreml_dirs[label if sep else ""] = Path(path if sep else d)
    if platform.system() != "Darwin":
        p.error("macOS only")
    if not BINARY.is_file():
        p.error(f"{BINARY} is missing; build the example first")

    with tempfile.TemporaryDirectory() as tmp:
        texts_path, ids_path = str(Path(tmp) / "texts.json"), str(Path(tmp) / "ids.json")
        json.dump(texts(), open(texts_path, "w"))
        subprocess.run([str(BINARY), "ids", str(a.mlx_dir), texts_path, ids_path], check=True)
        lengths = [len(x) for x in json.load(open(ids_path))]
        samples = []
        arms = a.arms
        for run in range(a.runs):
            for arm in arms[run % len(arms):] + arms[:run % len(arms)]:
                s = sample(arm, a, ids_path, tmp)
                s["run"] = run + 1
                samples.append(s)
                print(f"run={run + 1} arm={arm} texts_per_s={s['texts_per_s']:.1f} "
                      f"open_ms={s['open_ms']:.0f} peak_mib={s['peak_phys_footprint_bytes'] / 1048576:.0f} "
                      f"load={s['load_avg_1m_before']} power={s['power']}", file=sys.stderr)

    ref = next((s["embeddings"] for s in samples if s["arm"] == "cpu"), None)
    summary = {}
    for arm in arms:
        ss = [s for s in samples if s["arm"] == arm]
        row = {}
        for f in ["texts_per_s", "measured_ms", "open_ms", "warmup_ms", "peak_phys_footprint_bytes"]:
            vals = [s[f] for s in ss]
            row[f] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        # A NaN or infinite element is written as JSON null. Count those
        # texts rather than score them: an export that overflows fp16 on one
        # placement must show up as broken, not crash the report.
        last = ss[-1]["embeddings"]
        row["nonfinite_texts"] = sum(any(x is None for x in v) for v in last)
        if ref and not row["nonfinite_texts"]:
            c = [cos(x, y) for x, y in zip(last, ref)]
            row["cosine_to_cpu"] = {"min": min(c), "mean": sum(c) / len(c)}
        summary[arm] = row
    for s in samples:
        del s["embeddings"]
    report = {
        "protocol": "embeddinggemma3-backends-warm-interleaved-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "versions": {k: md.version(k) for k in ("mlx", "mlx-embeddings", "mlx-lm")},
        "token_lengths": {"min": min(lengths), "max": max(lengths), "texts": len(lengths)},
        "quiet_host": all(s["load_avg_1m_before"] < 4.0 and s["power"] == "ac" for s in samples),
        "samples": samples,
        "summary": summary,
    }
    text = json.dumps(report, indent=2) + "\n"
    (a.output.write_text(text) if a.output else sys.stdout.write(text))


if __name__ == "__main__":
    main()
