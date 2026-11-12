#!/usr/bin/env python3
"""Single-stream decode benchmark over an OpenAI-compatible streaming endpoint.

Same harness for every engine. Tokens are counted with ONE tokenizer (the
Qwen3.8 tokenizer.json), never by SSE chunks, because a speculative round
commits several tokens in one event. Decode tok/s excludes the tokens that
arrived in the first chunk and the time to reach it (TTFT is reported apart).

  python3 -I bench_api.py --base-url http://127.0.0.1:8000 --model m \
      --tokenizer ~/.cache/kapitsa/models/qwen3.8-27b-4bit/tokenizer.json \
      --label kapitsa-dflash --cases short-explanation,copy-heavy --runs 3 --out x.json
"""
import argparse, json, pathlib, statistics, sys, time, urllib.request

PROMPTS = pathlib.Path(__file__).resolve().parent.parent / "crates/bench/prompts/real-generation-v1"
COPY = (
    "Rename the variable `buf` to `scratch` in the function below and return the COMPLETE "
    "function with nothing else changed.\n\n```rust\n"
    "fn merge_runs(left: &[u32], right: &[u32]) -> Vec<u32> {\n"
    "    let mut buf = Vec::with_capacity(left.len() + right.len());\n"
    "    let (mut i, mut j) = (0usize, 0usize);\n"
    "    while i < left.len() && j < right.len() {\n"
    "        if left[i] <= right[j] {\n            buf.push(left[i]);\n            i += 1;\n"
    "        } else {\n            buf.push(right[j]);\n            j += 1;\n        }\n    }\n"
    "    while i < left.len() {\n        buf.push(left[i]);\n        i += 1;\n    }\n"
    "    while j < right.len() {\n        buf.push(right[j]);\n        j += 1;\n    }\n"
    "    buf\n}\n```"
)


def prompt_for(case):
    if case == "copy-heavy":
        return COPY
    return (PROMPTS / f"{case}.txt").read_text()


def one_run(tok, base, model, prompt, max_tokens, temperature):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_t = last_t = None
    text, first_text = [], ""
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ch = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            piece = (ch.get("reasoning_content") or "") + (ch.get("content") or "")
            if not piece:
                continue
            now = time.perf_counter()
            if first_t is None:
                first_t, first_text = now, piece
            last_t = now
            text.append(piece)
    if first_t is None:
        raise RuntimeError("no generated text")
    full = "".join(text)
    n = len(tok.encode(full, add_special_tokens=False).ids)
    n_first = len(tok.encode(first_text, add_special_tokens=False).ids)
    dt = max(last_t - first_t, 1e-9)
    return {
        "tokens": n,
        "ttft_s": first_t - t0,
        "decode_s": dt,
        "decode_tok_s": (n - n_first) / dt,
        "text_hash": __import__("hashlib").sha256(full.encode()).hexdigest()[:12],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", default="short-explanation,medium-review,copy-heavy")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    result = {"label": a.label, "cases": {}}
    for case in a.cases.split(","):
        p = prompt_for(case)
        one_run(tok, a.base_url, a.model, p, 64, a.temperature)  # warmup, discarded
        runs = [one_run(tok, a.base_url, a.model, p, a.max_tokens, a.temperature) for _ in range(a.runs)]
        rates = [r["decode_tok_s"] for r in runs]
        result["cases"][case] = {
            "runs": runs,
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
            "ttft_median": statistics.median(r["ttft_s"] for r in runs),
        }
        c = result["cases"][case]
        print(
            f"{a.label:24} {case:18} {c['median']:6.2f} tok/s [{c['min']:.2f}-{c['max']:.2f}] "
            f"ttft {c['ttft_median']:.2f}s tokens {[r['tokens'] for r in runs]} hashes {sorted({r['text_hash'] for r in runs})}",
            flush=True,
        )
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
