"""Token-level KL divergence between this port and llama.cpp, ON THE SAME
GGUF BYTES (ROADMAP Phase G's last open gate clause).

Run `cargo test -p turbospark-bench --test logit_dump` first, pointed at a
GGUF-derived install; it writes the id sequence and this port's full-vocab
logits. This script replays the SAME IDS through llama.cpp, on the
published GGUF the install was streamed from, and reports how far apart the
two distributions are.

    TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/gguf-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture

    uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
      ~/models/gguf-ref/gemma-4-26B-A4B-it-Q8_0.gguf /tmp/kld/gguf-warm /tmp/kld/mlx-warm

WHAT QUESTION THIS ANSWERS, and it is not the one `kld.py` answers.
`kld.py` compares this port against mlx-lm on the MLX-quantized bytes, so
it audits the INT4 path. Nothing audited the GGUF path, and Gemma's GGUF
install scores 39.8808 perplexity against the MLX install's 37.4176 -- the
higher-precision side scoring WORSE, which is backwards. Two readings are
consistent with that: Q8_0 genuinely loses on this corpus, or the GGUF
repack carries residual error. A second engine on the same GGUF separates
them, which is the same reason `kld.py` exists one layer down.

DO NOT compare 0.62 nats (this port's two installs against each other) to
0.0264 (this port against mlx-lm). The first is INT4 against Q8_0, i.e.
DIFFERENT WEIGHTS; the second is one set of bytes through two engines. That
error is on the record in CLAUDE.local.md; do not make it again.

THE FLOOR ARMS ARE NOT OPTIONAL, for exactly the reason `kld.py` runs
mlx-lm twice: a KL between two engines has no natural scale. llama.cpp is
therefore run three times, and each extra run buys one axis of scale:

  cached vs batched, same backend   -- the SHAPE floor
  cached on CPU vs cached on Metal  -- the BACKEND floor
  (the headline, this port vs llama.cpp cached, same backend)

MATCH THE BACKEND, NOT JUST THE BYTES. This measurement was first taken
against llama.cpp on CPU, because a 26.9 GB model looks like it will not
fit under this machine's Metal wired limit (it does). That reading was
0.05838 mean nats at 95.1% top-1, which sits ABOVE the shape floor and
looks like a real gap in this port. It is not: llama.cpp's own CPU and
Metal paths disagree with EACH OTHER by 0.05510 nats and 4.9% of the
argmaxes on this model, and matching the backend collapses the headline to
0.00845 at 98.2%. A 7x error, entirely in the reference. Feeding both
engines the same bytes is not enough when one of them is running a
different arithmetic backend from the other.

BOTH ENGINES ARE STEPPED THROUGH A CACHE for the headline, because that is
the shape this port runs in.
"""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# `softcap_of` and `check_heads` were WRITTEN here (Gotcha 38's fix, after a
# hardcoded 30.0 aborted a correct `qwen3moe` run) and now live in `kld.py`
# beside `divergences` and `perplexity`, because the question they answer --
# what transform does this install's head apply -- is about the install and
# not about which engine it is being compared against. `kld.py` needed the
# same fix and importing it is one implementation rather than two.
from kld import check_heads, divergences, perplexity, softcap_of  # noqa: E402

HARNESS_SRC = pathlib.Path(__file__).resolve().parent / "llamacpp_logits.c"
HARNESS_BIN = pathlib.Path("/tmp/llamacpp_logits")


def build_harness() -> None:
    if HARNESS_BIN.exists() and HARNESS_BIN.stat().st_mtime > HARNESS_SRC.stat().st_mtime:
        return
    prefix = subprocess.run(
        ["brew", "--prefix"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(
        [
            "cc", "-O2", "-o", str(HARNESS_BIN), str(HARNESS_SRC),
            f"-I{prefix}/include", f"-L{prefix}/lib", "-lllama",
        ],
        check=True,
    )


def llamacpp_logits(
    model: pathlib.Path,
    ids_path: pathlib.Path,
    out: pathlib.Path,
    mode: str,
    rows: int,
    vocab: int,
    n_gpu_layers: int,
) -> np.ndarray:
    """llama.cpp's next-token logits for every position, as float32 [rows, vocab].

    Cached results are reused: each shape costs minutes and 577 MiB, and the
    inputs are fixed files, so a re-run of the analysis should not re-run the
    model. Delete the .f32 to force a fresh pass.
    """
    expected = rows * vocab * 4
    if out.exists() and out.stat().st_size == expected:
        print(f"  reusing {out} ({expected / 2**20:.0f} MiB)", file=sys.stderr)
    else:
        print(f"  running llama.cpp {mode} (ngl={n_gpu_layers})", file=sys.stderr)
        proc = subprocess.run(
            [str(HARNESS_BIN), str(model), str(ids_path), str(out), mode, str(n_gpu_layers)],
            capture_output=True, text=True, check=True,
        )
        report = dict(
            line.split(" ", 1) for line in proc.stdout.strip().splitlines() if " " in line
        )
        print(f"  {report}", file=sys.stderr)
        if int(report["n_vocab"]) != vocab:
            sys.exit(f"llama.cpp reports vocab {report['n_vocab']}, the dump says {vocab}")
    # The head check lives in `check_heads`, off the returned array rather
    # than off the harness's printed `max_abs_logit`, so that it runs on a
    # REUSED arm too. Keyed on the fresh-run path it was skipped exactly
    # when the analysis was being re-run, which is most of the time.
    data = np.fromfile(out, dtype=np.float32)
    if data.size != rows * vocab:
        sys.exit(f"{out} holds {data.size} values, expected {rows * vocab}")
    return data.reshape(rows, vocab)


def load_port_dump(dump: pathlib.Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((dump / "meta.json").read_text())
    rows, vocab = meta["rows"], meta["vocab_size"]
    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    return port.reshape(rows, vocab).astype(np.float32), meta


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <model.gguf> <port-dump-dir> [more-dump-dirs...]")
    model = pathlib.Path(sys.argv[1]).expanduser()
    dumps = [pathlib.Path(d).expanduser() for d in sys.argv[2:]]
    # Metal by default: this port is a Metal engine, and the backend has to
    # match or the headline measures ggml's CPU/Metal gap (see the module
    # doc). 0 is the other arm, kept as the backend floor.
    n_gpu_layers = int(os.environ.get("TURBOSPARK_LLAMACPP_NGL", "99"))
    other_ngl = 0 if n_gpu_layers else 99

    primary, meta = load_port_dump(dumps[0])
    rows, vocab, ids = meta["rows"], meta["vocab_size"], meta["token_ids"]
    first = meta["first_scored_position"]

    work = pathlib.Path(os.environ.get("TURBOSPARK_LLAMACPP_DIR", "/tmp/kld/llamacpp"))
    work.mkdir(parents=True, exist_ok=True)
    ids_path = work / "ids.i32"
    np.asarray(ids, dtype=np.int32).tofile(ids_path)

    build_harness()

    # The cache filename MUST carry the model stem. Every arm of every model
    # is the same `rows * vocab * 4` bytes, and the reuse check is a size
    # check, so a name keyed only on (mode, ngl) makes a second model
    # silently load the first one's logits and report itself as identical to
    # it. That is a wrong ANSWER, not a stale file, and it costs nothing to
    # rule out. Pre-existing arms written before this fix keep their old
    # names and are simply re-run once under the new ones.
    def arm(mode: str, ngl: int) -> np.ndarray:
        return llamacpp_logits(
            model, ids_path, work / f"{model.stem}-{mode}-ngl{ngl}.f32", mode, rows, vocab, ngl
        )

    def backend(ngl: int) -> str:
        return "CPU" if ngl == 0 else "Metal"

    cached = arm("cached", n_gpu_layers)
    batched = arm("batched", n_gpu_layers)
    cross = arm("cached", other_ngl)

    report = {
        "model": str(model),
        "rows": rows,
        "backend": backend(n_gpu_layers),
        # Read this first. It is the precondition for everything under it.
        # "llamacpp" keeps this report's key `max_abs_logit_llamacpp`, which
        # the shared implementation derives from the name rather than
        # hardcoding, so the frozen reports stay comparable.
        "heads": check_heads(cached, primary, softcap_of(meta["install"]), "llamacpp"),
        # llama.cpp against ITSELF, twice, so the headline has a scale.
        # Nothing below is readable without these two (see the module doc).
        "kl_shape_floor": divergences(batched, cached),
        "kl_backend_floor": divergences(cross, cached),
        "perplexity": {
            f"llamacpp_cached_{backend(n_gpu_layers)}": perplexity(cached, ids, first),
            f"llamacpp_batched_{backend(n_gpu_layers)}": perplexity(batched, ids, first),
            f"llamacpp_cached_{backend(other_ngl)}": perplexity(cross, ids, first),
        },
        "vs_llamacpp_cached": {},
    }
    for dump in dumps:
        port, dmeta = load_port_dump(dump)
        if port.shape != cached.shape:
            sys.exit(f"{dump} dumped {port.shape}, llama.cpp returned {cached.shape}")
        if dmeta["token_ids"] != ids:
            sys.exit(f"{dump} walked a different id sequence than {dumps[0]}")
        report["vs_llamacpp_cached"][dump.name] = {
            "install": dmeta["install"],
            "cache_state": dmeta["cache_state"],
            "kl": divergences(cached, port),
        }
        report["perplexity"][dump.name] = perplexity(port, ids, first)

    print(json.dumps(report, indent=2))
    (work / f"kld_llamacpp-{model.stem}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
