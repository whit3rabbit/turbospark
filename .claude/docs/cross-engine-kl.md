# Cross-engine KL: this port against mlx-lm and llama.cpp

Split out of the repository's `AGENTS.md`, which is loaded into every
session; this page is loaded when you follow the link. Nothing here is a
new fact, and `AGENTS.md` remains the map.

Does this port agree with another engine on the SAME bytes? Two steps every
time: dump this port's full-vocab logits plus the exact token ids it walked,
then replay those IDS (never the prose, or a tokenizer difference reads as a
numerics gap) through the reference.

**Read the floors, not just the number.** There are TWO: a shape floor
(batched versus cached) and a BACKEND floor, and the second was 41x the
first the one time it was skipped (`AGENTS.md` Gotcha 34). Run the reference
on Metal, not CPU. An MoE shape floor is four orders of magnitude above a
dense one, so the RATIO transfers between families and the absolute does not.

All commands run from the REPOSITORY ROOT, not from this directory.

```sh
# Cross-engine check (ROADMAP Phase Q, last item): does this port agree
# with mlx-lm on the SAME quantized bytes? Two steps. The first dumps this
# port's full-vocab logits for the quality corpus plus the exact token ids
# it walked (~275 MiB, ~30 s). The second replays those IDS -- never the
# prose, or a tokenizer difference would read as a numerics gap -- through
# mlx-lm and prints the KL. mlx-lm runs in a uv ephemeral env, so nothing
# is installed globally and nothing is added to this workspace. Needs
# `hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision
# 0d77464eeb233a2da68ebf9d7dc4edaac7db956d` first (14.6 GB). Read the
# floor, not just the number: docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark gemma4

# Same dump with NO warmup walk, which is the condition quality_gate takes
# its perplexity under. Reproduces the frozen row exactly; that is the
# cross-check that the dump measures what the gate measures.
TURBOSPARK_LOGIT_DUMP_COLD=1 TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/cold \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture

# The same question one layer down, for the GGUF path: does this port agree
# with llama.cpp on the SAME GGUF bytes? Closes Phase G's last gate clause
# and is the same-precision reference Phase S needs. Dump from a
# GGUF-derived install, then replay the ids through llama.cpp (brew's
# `llama.cpp`; a small harness against its own header, since no shipped
# binary teacher-forces an id list). Needs the 26.9 GB GGUF locally --
# llama.cpp cannot stream it the way the repack walk does. Metal by
# default, and that is not cosmetic: see Gotcha 34 before running it on CPU.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/gguf-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download ggml-org/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf \
  --local-dir ~/models/gguf-ref
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-Q8_0.gguf /tmp/kld/gguf-warm /tmp/kld/turbospark

# The check that says the IQ kernels are faithful rather than merely
# plausible: this port against llama.cpp on the IDENTICAL bytes. Needs the
# 12 GB file locally (llama.cpp cannot stream it) and MUST run on Metal --
# Gotcha 34. Reads 0.00440 mean nats at 97.5% top-1, against a 0.03741
# backend floor.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/iq3-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf /tmp/kld/iq3-warm

# The same check for the `qwen3moe` family (ROADMAP M3), and the ONLY
# instrument that can see the two things its fixture tests record themselves
# as blind to: the per-head q/k norms' ORDER relative to RoPE, and the RMS
# epsilon. Reads 0.00320 mean nats at 97.9% top-1 between a 0.00135 shape
# floor and a 0.00988 backend floor, so both are right. Metal, not CPU
# (Gotcha 34). Needs the 17.3 GB GGUF locally; the install was STREAMED from
# it, so it is not on disk by default. ~30 s per arm plus ~12 min to fetch.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/qwen3moe-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir ~/models/gguf-ref
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/Qwen3-30B-A3B-Q4_K_M.gguf /tmp/kld/qwen3moe-warm

# The family's FIRST external check, and until it ran its four gate rows were
# all self-referential. Aimed at the NEWEST code in it -- the `qwen35` dense
# GGUF path, whose name table, `SUPPORTED_GGUF` row and V-head de-interleave
# are all new and whose bring-up found three bugs of one kind (Gotcha 61). It
# is the only instrument that reaches the de-interleave on the DENSE half,
# the per-head q/k norms' ORDER relative to RoPE, and the RMS epsilon. Reads
# 0.000138 mean nats at 99.65% top-1, FIFTEEN TIMES BELOW its 0.00202 backend
# floor. Metal, not CPU (Gotcha 34). ~90 s for all three arms plus ~5 min to
# fetch the 9.5 GB GGUF, which is not on disk by default (the install was
# STREAMED from it). READ ITS BACKEND FLOOR AND NOT ITS SHAPE FLOOR: a dense
# model's batched and cached passes are nearly the same computation, so that
# floor collapses to 0.0000024 and a ratio against it means little
# (docs/BENCHMARKS.md).
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith9b-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download ornith-ai/Ornith-1.5-9B-GGUF Ornith-1.5-9B-Q8_0.gguf \
  --revision 0677a38f331a214c4e5e7bd07ecab04c14ac52f1 --local-dir ~/models/gguf-ref
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/Ornith-1.5-9B-Q8_0.gguf /tmp/kld/ornith9b-warm

# The SAME family's MoE half, against MLX rather than llama.cpp, because this
# install comes from an affine INT4 conversion where the 9B's comes from a
# GGUF -- so the pair covers both intake formats instead of one twice. Needs
# no fork: upstream mlx-lm carries `qwen3_5_moe`. Reads 0.02739 mean nats at
# 91.10% top-1 against a 0.02780 shape floor, i.e. BELOW it. **Do not compare
# that absolute to the 9B's 0.000138** -- an MoE shape floor is four orders of
# magnitude above a dense one because batched-vs-cached routing and reduce
# order is what it is made of, so the RATIO transfers and the absolute does
# not (`crates/bench/CLAUDE.md` Gotcha 8). No backend floor: mlx's CPU path
# runs this at 15.4 s/position (~2.5 h), measured, despite only 3B active.
TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith35b-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
  --revision 19504d912fa8fc7622bf6b1de3db5d5d890b1f02
uv run --python 3.12 --with mlx-lm --with numpy \
  scripts/kld_mlx_affine.py /tmp/kld/ornith35b-warm ornith-35b-4bit

# Its cross-engine KL, through the SAME driver the 1-bit one uses. The
# checkpoint name is REQUIRED and not defaulted: the two published
# checkpoints have the same module count and the same shapes, so pairing a
# dump with the wrong reference passes every check inside the script and
# reads as a kernel bug. Upstream mlx supports bits=2, so unlike the 1-bit
# arm this one needs no fork.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ternary-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with 'mlx-lm==0.31.2' --with numpy \
  scripts/kld_mlx_affine.py /tmp/kld/ternary-warm ternary-2bit
```
