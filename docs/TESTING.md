# Testing

What the suite covers, how it is gated, and how to run each part.

## The default suite

```sh
cargo test --workspace
```

1,435 tests as of 2026-08-29, plus 145 that are `#[ignore]`d, re-counted
2026-09-10 (see below). The two dates differ because only the second is
greppable: the first needs a build and was not re-derived on 2026-09-10. **Re-count before quoting either number.** Both were stale by more
than 2x when this line was last corrected (they read 458 and 18, unchanged
since 2026-08-08 while eleven families and phases landed), and nothing goes
red when they rot: a count is prose. The one-liners that produce them are
`cargo test --workspace` for the first and, for the second,
`rg '^\s*#\[ignore' crates --glob '**/tests/*.rs' | wc -l`. On macOS this includes every Metal test, which needs a real
Metal-capable device and Xcode's `metal` toolchain
(`xcrun -sdk macosx metal`). On Linux `crates/gpu` compiles to nothing and
the GPU-dependent test files compile away with it, so the same command
stays green.

The full pre-handoff gate:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

`make check` runs fmt-check + clippy + test-debug.

## What each crate's tests prove

| Crate | Covers |
| --- | --- |
| `core` | Runtime-config allowed sets and their panicking setters, chunk-size resolution. |
| `compute` | CPU reference kernels (RmsNorm, WHT, RoPE incl. YaRN frequency tables, causal attention with and without sinks, int4/int8 quant + GEMV, the GGUF block-quant and codebook references incl. MXFP4, embedding, MoE FFN, softcap-softmax). These are the numerical ground truth the GPU is checked against. |
| `invocation` | Argument parsing: outcomes, ordering, failures, defaults, usage text, exit status. Pure, no I/O. |
| `selection` | Sampling contract: shaping, truncation, penalty, choose. |
| `window-fit` | Conversation-window turn dropping. |
| `tokenizer` | Dialect resolution, chat templates (including the real vendored Qwen ChatML template through minijinja, and that a checkpoint's own template beats its dialect), streaming detokenizer, stop matching, tool-call parsing. |
| `model-io` | Manifest decode and field validation, packed-expert layout, resident index, SHA-256, install receipt. |
| `streaming` | Expert cache eviction policy against scripted access traces (no real install needed). |
| `gpu` | Per-kernel parity against the matching `compute` reference on real hardware, plus KV cache sizing and the resident zero-copy proof. macOS only. |
| `runtime` | The raw-completion loop against `ScriptedLogitProducer`, plus real end-to-end forward passes over synthetic installs. |
| `cli` | Black-box binary invocation, including real generation in all three modes. |
| `repack` | Safetensors parsing, quantization, `.gturbo` assembly round-tripped through every `model-io` loader, install verification. |
| `server` | OpenAI-compatible endpoint shapes, full-response and SSE, the `--model` argument parser, and (gated) the real `RealForwardRunner` backend end to end. |
| `bench` | Protocol constants and footer format, the memory sampler, and the binary's black-box output. Its gated targets carry the quality axis: per-install perplexity and golden digests, the damage-sensitivity proof, and the logit dump feeding the cross-engine KLD. |

### `gpt-oss`: kernels tested, flow not yet written (ROADMAP M5)

The family's four kernel-level differences each have a parity target, and
each is held against a reference **written out from ggml's own source
rather than called from this port** -- a formula compared against this
port's copy of it establishes only that the copy was self-consistent,
which is exactly the failure a new family risks.

| target | proves |
| --- | --- |
| `gpu/tests/moe_gguf_parity.rs` | the MXFP4 unpack against `dequantize_mxfp4`, and separately gpt-oss's expert MATH (clamped SwiGLU, per-expert biases) against `ggml_compute_forward_swiglu_oai_f32`. The two are separate cases so neither hides the other. |
| `gpu/tests/attention_sinks.rs` | the sink term against `causal_attention_with_sinks`, at sinks negligible, comparable and dominant relative to the real scores. |
| `gpu/tests/rope_yarn_parity.rs` | the YaRN frequency table against ggml's `rope_yarn` restated inline, plus the ramp boundaries and that `mscale` is not 1.0. |
| `compute/tests/quant_gguf_mxfp4.rs` | the MXFP4 codebook and E8M0 scale against a ggml oracle, by `==`, over all 256 exponent bytes. |
| `runtime/tests/gguf_install_refused.rs` | that an MXFP4 install opens and decodes, and that the SAME type is refused as a resident tensor -- the two executable-type gates disagreeing on purpose. |
| `repack/tests/gguf_checkpoint_network.rs` (ignored) | that all 459 real tensor names map and the derived `ArchConfig` equals the baseline, off the header and before any download. |

**Two mutations in this set are documented as unobservable rather than
claimed**, which is the honest form when a test cannot see something.
Dropping MXFP4's subnormal-exponent branch leaves everything green because
`e = 0` and `e = 1` stand for ~1e-39, which cannot survive a dot product
rounded to FP16 (the `compute` oracle is what pins those two). And omitting
the attention sink from the running maximum is a numerical-stability guard,
not a correctness one -- softmax is invariant to the choice of maximum, and
wherever the omission would overflow, the sink already dominates and both
answers are ~0.

There is no end-to-end test of the gpt-oss LAYER, because
`crates/runtime/src/families/gptoss/` does not exist yet. The fixture that
does exist is Gemma-shaped with gpt-oss's block types.

### The plain-GQA-plus-MoE flow: the `llama` and `qwen3moe` families

One decode flow, two families, so the tests come in pairs at the cheap end and
stay separate at the expensive one. Four levels, and the split between them is
what keeps the expensive ones rare:

- `crates/runtime/tests/real_forward_llama.rs` (default suite, macOS): builds a
  tiny Mixtral-shaped install through the real repack pipeline and decodes on
  real Metal. Covers determinism across expert-cache state, prefill/produce
  agreement, the no-allocation-per-token rule, the slot-count cap, and the
  dense-half refusal.
- `crates/runtime/tests/real_forward_qwen3moe.rs` (default suite, macOS): the
  same flow under the other family tag. Its reason to exist is one test:
  two installs of identical weights differing only in `ArchConfig.family`
  must decode differently, because one norms q and k per head and the other
  does not. Read at a context of 4 rather than at position 0, where a softmax
  over one key is 1.0 whatever the logit and no q/k transform is observable
  at all. Its module header records what it mutation-checked and what it
  provably cannot see (the norms' order relative to RoPE, and the epsilon).
- `crates/repack/tests/gguf_llama_rope_patch.rs` (unit part in the default
  suite): pins the rotary row permutation as the exact inverse of llama.cpp's
  converter transform. Its `#[ignore]`d half patches a real install in place,
  which is the diagnostic loop AGENTS.md Gotcha 33 prescribes -- seconds per
  hypothesis instead of a 35-minute repack.
- `crates/repack/tests/gguf_mixtral_install_network.rs` (`#[ignore]`d): the
  real published Mixtral Q4_K_M streamed into an install, plus a cheap dense
  walk (TinyLlama 1.1B Q6_K, 0.84 GiB, ~3 min) that exercises the half of the
  name table Mixtral never reaches. Prefer the dense one when the question is
  about names or the dense branch; it is 30x cheaper.
- `crates/repack/tests/gguf_qwen3moe_install_network.rs` (`#[ignore]`d): the
  real published Qwen3-30B-A3B Q4_K_M streamed into an install. This is the
  fine-grained checkpoint the Mixtral granularity finding asked for, so it is
  also the only one of the two whose memory oracle and quality gate mean
  anything (2.5 MiB per expert against 108.9). It asserts the granularity on
  the artifact's own layout rather than as arithmetic on a model card.

The streamed walk resumes: a layer file already on disk at the size the walk
would write is adopted without a network read, so a transport failure costs
the remaining layers rather than all of them.

## Gating conventions

Three gates are in use. Match the existing idiom when adding a test.

### macOS

Metal-dependent test files carry a crate-level attribute as line 1:

```rust
#![cfg(target_os = "macos")]
```

All of `crates/gpu/tests/*`, `crates/runtime/tests/{real_forward,
real_forward_gemma4,real_forward_qwen,golden_tokens}.rs`, `crates/cli/tests/real_generation.rs`,
and `crates/bench/tests/memory_oracle.rs` do this. Inside an otherwise
portable file, gate the single item instead
(`crates/cli/tests/mference_check.rs`).

Source-side, `crates/gpu/src/lib.rs` gates each `mod` and `pub use`
individually, and `crates/bench/src/main.rs` uses the cfg'd function-pair
idiom (real implementation plus a stub that exits 2).

### Ignored (expensive or needs external data)

92 targets carry `#[ignore]`d tests, 145 functions between them as of
2026-09-10 (`bench` 46/51, `repack` 34/60, `gpu` 4/15, `runtime` 2/6,
`selection` 2/4, `catalog` 1/2, `server` 1/4, `tokenizer` 2/3). Each has a reason string and
a module doc with the exact command. The commands below are the ones that are gates.

The previous substring-counting command also counted `#[ignore]` mentions
in comments. The command above now matches attribute lines only; the
2026-09-10 recount excludes those mentions rather than removing tests. **The one-liner counts `crates/*/tests` only**, so an `#[ignore]`
inside a crate's `src/` (there is one, in `crates/ffi`) is not in it.

**The BENCHMARK targets are not gates and report rather than assert**, so
they are listed here and documented where their numbers live rather than
being run in the handoff loop. `crates/selection`'s `rank_top_k` (a sampler
microbenchmark) and `crates/gpu`'s `attention_chunk_bench` (the split-KV
chunk sweep) are in `docs/BENCHMARKS.md`; `crates/gpu`'s
`gemv_bandwidth_bench` (weight-read headroom, `c(M)`, `c(R, M)`) and
`moe_prefill_batch_bench` are in `docs/BATCHED_PREFILL.md`, as is
`gdn_prefill_share_bench`. That last one exists because the obvious
instrument does not work: it prices `gdn_delta_step_prefill` against the
INT4 matrices a prefill micro-batch walks in **0.45 seconds**, where
`TURBOSPARK_DISPATCH_PROFILE=1` -- the only surface that names kernels -- waits
on every command buffer at commit and did not finish a 150-token prefill in
12 minutes.

```sh
# Real ~14.6 GB Gemma 4 checkpoint download plus full repack.
cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture

# Real ~20.4 GB Qwen 3.6 checkpoint download plus full repack. The only
# test that covers the multi-shard walk on that family: its synthetic
# fixture is a single in-memory shard, so a companion tensor living in a
# different shard than its weight is unexercised in the default suite.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture

# Real ~4.28 GB Qwen2.5 MLX checkpoint download plus full repack. The target
# also contains a cheap MLX header gate and a pinned Q3_K_M GGUF header gate.
# The MLX install is the real Qwen2 artifact currently exercised end to end;
# there is no Qwen2 memory or quality baseline yet.
TURBOSPARK_QWEN2_INSTALL_DIR=~/models/qwen25-7b-4bit.gturbo \
  cargo test -p turbospark-repack --test qwen2_checkpoint_network --release -- --ignored --nocapture

# Real ~270 MB HF checkpoint download through the Llama-family mapping.
cargo test -p turbospark-repack --test hf_checkpoint_network --release -- --ignored --nocapture

# The four GGUF checks (ROADMAP Phase G). All are `*_network` but NONE of
# them downloads a checkpoint: each reads a few KB to a few MB off a
# 20-27 GB remote file over range requests, in seconds. They are grouped
# here rather than above for that reason -- do not budget a download for
# them, and prefer a ranged read when adding the next one.
#
# 1. The header, three ways: this port's parser against llama.cpp's
#    converter, every tensor name maps, and the ArchConfig derived from
#    GGUF metadata equals the one the .gturbo install declares.
#    The same file's three `scopes_phase_s_*` cases survey candidate
#    sub-4-bit checkpoints (ROADMAP Phase S) at the same cost, and print a
#    ggml type histogram in BYTES. Read that histogram's UNSIZED rows first:
#    a type this port cannot size is one it cannot ingest, which on a mixed
#    file is usually the routed experts, and printing it as 0 bytes once
#    made an imatrix file look 76% Q8_0. Check the sized total against the
#    published file size before believing any row.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# 2. Which half of Gemma's fused ffn_gate_up_exps is the gate. Correlates a
#    dequantized layer 0 expert 0 against the MLX install; doubles as a
#    real-data check on the Q8_0 dequant reference.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# 3. The evidence behind the repack-time transcode, which is now LANDED
#    (`gguf_checkpoint/transcode.rs::transcode_f32`): GGUF's F32 norms are upcast
#    BF16 and narrow back bit-exactly, and INT8-transcoding its F32 router
#    does not move the routing decision. Needs no install. The transcode's
#    own behaviour is covered by the unit tests in
#    `crates/repack/tests/gguf_checkpoint.rs`, on the synthetic fixture;
#    this one is why those tests are allowed to assume what they assume.
cargo test -p turbospark-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# 4. The Q4_K reference against real published bytes. Correlates a
#    dequantized layer 0 expert 0 gate row of the real Qwen 3.6 Q4_K_M
#    against the same row in the install (+0.9930), with the unrelated `up`
#    matrix as the control (+0.12). This is the only check that can catch a
#    decoder and the fixture quantizer that feeds it being wrong TOGETHER,
#    which is the failure mode `crates/compute`'s unit tests structurally
#    cannot see. It skips all-zero rows and asserts the two sides agree on
#    which those are: see AGENTS.md Gotcha 30 before reading a low number
#    out of it.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# 4b. The same, for the three IQ-codebook types (ROADMAP Phase S), all in
#     one run because the candidate puts a different one on each side of a
#     routed expert and a THIRD on layer 29 alone. IQ3_XXS gate/up reads
#     +0.9751/+0.9760 against the MLX install, IQ4_NL down +0.9927, IQ4_XS
#     +0.9928, every crossed control under 0.05. Also re-settles
#     FUSED_GATE_FIRST for this converter, deriving the answer from the
#     correlations rather than asserting the constant.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_iq_network --release -- --ignored --nocapture

# 5. Qwen's SOURCE CONVENTIONS, which are not a format question: llama.cpp
#    orders V heads differently and stores -exp(A_log). Checks every tensor
#    on that axis on every layer, and with TURBOSPARK_QWEN_PATCH=1 rewrites
#    them in place, which is how a whole-model coherence test costs seconds
#    instead of a ~21-minute repack. Numbers in crates/repack/CLAUDE.md
#    Gotcha 7; the trap that made this eight tensors rather than three is
#    AGENTS.md Gotcha 33.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_convention_patch --release -- --ignored --nocapture

# 6. The diagnostic that found the five QUANTIZED tensors on that axis,
#    which a BF16-only probe structurally cannot see. Recovers the
#    permutation outright where a tensor has one row per head.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_quant_probe --release -- --ignored --nocapture

# The memory oracle (see docs/BENCHMARKING.md). One target per model
# family: the footprint assertion is a whole-session peak, so two
# families in one process cannot each have a ceiling.
#
# THE TWO BLOCKS BELOW ARE EXAMPLES, NOT THE FULL LIST, and the list has
# grown past what is worth duplicating here: FIFTEEN oracle targets (gemma4,
# qwen36, qwen3moe, mistral, gptoss, museglimmer, ornith9b, ornith35b,
# qwen38, qwen4exp, ternary, vision, spark, minimax, qwen3_dense) and FOURTEEN quality
# gates (the same list without mistral and vision, plus iq3's).
# MiniMax has no accepted baseline yet; see docs/MINIMAX_M2_PHASE0.md.
#
# Those two numbers read SEVEN and SEVEN until 2026-09-06, which is the
# rot this very paragraph tells you to check for -- run the `ls` below
# rather than believing them. AGENTS.md's command block carries the full set
# with their env vars and their per-family quirks -- the two windows that
# are not 4,096, the one budget that is not 1,024, and which installs each
# needs. Run `ls crates/bench/tests/*_{quality_gate,memory_oracle}.rs` if
# that block is ever out of date too.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture

# Prefix KV reuse against the real install. NOT a memory or numerics gate --
# it is a BEHAVIOURAL one, and the only thing that can make its claim: a turn
# that continues from the previous turn's KV must generate the tokens a full
# re-prefill generates. A scripted producer has no KV to be wrong about, so
# the sliding-window ring and the cursor rewind are unreachable without an
# install. Run it on any change to the generation loops, `kv_prefix`, or the
# KV cursor. Seconds, not minutes.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-runtime --test prefix_reuse_real --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q; numbers in docs/BENCHMARKS.md). Split
# per family for the same one-model-per-process reason as the oracle.
# Takes about 80 seconds each (four arms: perplexity, greedy, sampled, and
# a constrained-working-set repeat at 8 expert-cache slots).
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture

# The 3-bit install's gate (ROADMAP Phase S). A SEPARATE TARGET with its own
# chip row, not the Gemma gate with an env var moved: the existing rows are
# keyed on the chip and freeze the MLX INT4 goldens, so pointing an install
# var at a different artifact asserts the wrong digests and fails for a
# reason that is not a regression. The same trap is why neither GGUF install
# ever got a Phase Q run. ~2 min.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-bench --test iq3_quality_gate --release -- --ignored --nocapture

# Proof the gate above can see quantization damage: clone the install
# (APFS clonefile, original untouched), shift one quantization level in a
# strided subset of INT4 weight ranges selected from packed_experts/layout.json
# (never the BF16 scales or biases), and re-measure. About 30 seconds.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture

# THE VISION GATES (docs/VISION.md's stages 2 and 4). Listed because they
# were in none of these sections until 2026-09-06 -- this file did not
# contain the word "vision" at all, through nine landed milestones.
#
# `~/models/qwen38-27b-vision.gturbo` is a SEPARATE install from
# `qwen38-27b.gturbo` on purpose: that one backs the frozen qwen38 rows and
# ~0.9 GiB of tower would force a re-freeze for a component neither gate
# exercises. Do not merge them.
#
# The multi-page memory oracle. Four rounds over ONE open runner (large ->
# medium -> small -> large again), asserting the peak does not grow past
# what the largest page establishes AND that each round's transcription
# carries a marker unique to that page -- the second half is not decoration,
# an oracle that asserts only a memory SHAPE cannot see whether the pages
# were read at all, which is the M-V5 bug's exact shape.
TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
  cargo test -p turbospark-bench --test vision_memory_oracle --release -- --ignored --nocapture

# The tower against mlx-vlm, per stage. Needs a dump from
# `scripts/vision_tower_probe.py --mode dump` as well as the install, and
# the probe tower must be at the revision the install was STREAMED from --
# the two published towers have identical shapes, so pairing the wrong one
# compares two different models and fails nothing loudly.
TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
TURBOSPARK_VISION_DUMP_DIR=/tmp/vision-dump \
  cargo test -p turbospark-runtime --test vision_tower_parity --release -- --ignored --nocapture

# The full model against mlx-vlm on a text+image prompt, which is the only
# instrument that reaches which rows land at which positions and whether
# the mRoPE selector agrees. Its working dump is ~2 GB and regenerable in
# about four minutes, so /tmp is correct for it. See docs/VISION.md.
TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
TURBOSPARK_VISION_KLD_DIR=/tmp/vision-kld \
  cargo test -p turbospark-bench --test vision_logit_dump --release -- --ignored --nocapture

# The cross-engine half of Phase Q, and the one test here whose second
# step is NOT cargo: it dumps this port's full-vocab logits plus the exact
# token ids it walked (~275 MiB, ~30 s), then replays those IDS through
# mlx-lm and prints the KL. Needs the 14.6 GB reference checkpoint
# (`hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision
# 0d77464eeb233a2da68ebf9d7dc4edaac7db956d`). mlx-lm runs in a uv
# ephemeral env, so it is never installed globally and never enters this
# workspace's dependency graph. TURBOSPARK_LOGIT_DUMP_COLD=1 skips the
# warmup walk and reproduces quality_gate's frozen perplexity exactly,
# which is the cross-check that the dump measures what the gate measures.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark gemma4

# The same shape against llama.cpp, on a GGUF-derived install and the GGUF
# bytes it was streamed from. This is what closed Phase G's last gate
# clause. Needs brew's llama.cpp (the driver compiles
# scripts/llamacpp_logits.c against its header) and a LOCAL copy of the
# 26.9 GB GGUF: llama.cpp cannot stream it the way the repack walk can, so
# unlike every other target here it is gated on disk rather than on time.
# Runs llama.cpp three times, and all three are needed to read the result:
# two shapes for the shape floor, two backends for the backend floor
# (AGENTS.md Gotcha 34 -- CPU is NOT interchangeable with Metal here).
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/gguf-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-Q8_0.gguf /tmp/kld/gguf-warm

# The same again for the DENSE `qwen35` half (Ornith-1.5-9B), which is worth
# running as its own example for two reasons. It was the family's FIRST
# external check of any kind -- its four frozen gate rows are all
# self-referential, and the per-tensor probe beside them is a STATIC check
# that cannot see how the runtime USES the tensors it validates. And READING
# it needs the backend floor rather than the shape floor: a dense model's
# batched and cached passes are nearly the same computation, so that floor
# collapses (0.0000024 here against the MoE families' ~0.00135) and a ratio
# against it means little, while the headline still sits 15x BELOW the
# backend floor. All three arms finish in ~90 s at this size, so unlike the
# dense 27B -- where `kld_mlx_affine.py` reports a string instead of the
# backend floor -- there is no reason to skip one.
TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith9b-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/Ornith-1.5-9B-Q8_0.gguf /tmp/kld/ornith9b-warm

# The same driver also answers questions about checkpoints this port CANNOT
# RUN, which is how ROADMAP Phase S was quality-gated before a kernel
# existed: point it at a candidate GGUF and pass this port's existing dumps
# as the comparison arms. No ingest, no install, one download.
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  /tmp/kld/gguf-warm /tmp/kld/mlx-warm
#
# The driver caches each arm's 577 MiB of logits under the MODEL STEM, and
# that is load-bearing rather than tidy: every arm of every model is the
# same rows * vocab * 4 bytes and the reuse check is a size check, so a
# name without the stem makes a second model silently load the first one's
# logits and report the two as identical. A wrong answer, not an error.

# Split-KV chunk-count sweep on the decode attention kernel. Needs no
# model install: it is the kernel alone at the real Gemma 4 shapes, and
# it reports speedup ratios rather than absolute times so it stays
# readable on a throttled machine. Takes seconds.
cargo test -p turbospark-gpu --test attention_chunk_bench --release -- --ignored --nocapture

# THE THREE STREAMED INSTALLS. Each reads a published GGUF over HTTP a layer
# at a time -- the 12-27 GB checkpoint is never written to disk -- and writes
# only the install. 18-24 min each. What they buy over the ranged probes is
# everything a fixture cannot show: every hole the real files exposed was in
# the SHAPE of the model rather than the block format. Each refuses to
# overwrite an install the gates are measured against.
TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_install_network --release -- --ignored --nocapture
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_install_network --release -- --ignored --nocapture
# The third is the MIXED one, and the only install whose layers differ from
# each other. It asserts the per-layer stride saves over 30% on disk, which
# is the whole premise of ingesting it: padded to the model-wide maximum the
# same experts would take 16.23 GB instead of 10.33.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-repack --test gguf_iq_install_network --release -- --ignored --nocapture

# The server's real backend end to end: one model open, one non-streaming
# and one streaming request through the bound loopback server.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-server --test real_backend --release -- --ignored --nocapture
```

Run all ignored tests at once with `cargo test --workspace -- --ignored`,
but note that three of them download many gigabytes.

A performance measurement is `#[ignore]`d rather than left out because it
is the evidence behind a constant in `src/` (`MAX_CHUNKS`), and a constant
whose justification cannot be re-run is a constant nobody will ever dare
change. It asserts correctness (every chunk count computes the same
attention) so it cannot rot silently; only the timings are advisory.

### Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `TURBOSPARK_GEMMA4_INSTALL_DIR` | `gemma4_checkpoint_network`, `memory_oracle`, `quality_gate`, `quality_sensitivity`, `logit_dump`, `real_backend`, `gguf_checkpoint_network`, `gguf_fused_gate_network`, `prefix_reuse_real` | Where the real `.gturbo` install lives. The oracle, the quality gate, the logit dump, and the server test skip (with a note) when unset; the repack test falls back to a temp dir. The two GGUF tests use the install as an independent REFERENCE rather than as a subject: `gguf_checkpoint_network` cross-checks names and `ArchConfig` against it and skips those two checks when unset, and `gguf_fused_gate_network` correlates against its expert weights and skips entirely. Both accept a leading `~/`. |
| `TURBOSPARK_QWEN36_INSTALL_DIR` | `qwen36_checkpoint_network`, `qwen36_memory_oracle`, `qwen36_quality_gate`, `logit_dump` | Where the repacked Qwen 3.6 install lives. The oracle and the quality gate skip (with a note) when unset; the repack test falls back to a temp dir. Deliberately a second variable rather than a generalized one, so both installs can coexist and each target asserts its own family's row. `logit_dump` takes it as a fallback when the Gemma variable is unset; its cross-engine reference is `scripts/kld_mlx_affine.py`'s `qwen36` row, not `scripts/kld.py` (that driver has no guard that an MoE reference loaded quantized, and this checkpoint is one). |
| `TURBOSPARK_QWEN3MOE_INSTALL_DIR` | `gguf_qwen3moe_install_network`, `qwen3moe_memory_oracle`, `qwen3moe_quality_gate`, `logit_dump` | Where the real Qwen3-30B-A3B Q4_K_M GGUF install lives (ROADMAP M3). A FOURTH variable for the same reason as the two below it: each family's oracle asserts its own ceiling (2,900 MiB here against Gemma's 2,300, because 48 layers of slot cache is not 30) and its own frozen digests. The install test refuses to write to any of the pinned variables. `logit_dump` takes it as the third fallback, which is how the cross-engine KL against llama.cpp is run on this family. |
| `TURBOSPARK_QWEN2_INSTALL_DIR` | `qwen2_checkpoint_network` | Where the real Qwen2/Qwen2.5 `.gturbo` install lives. The full target streams the pinned 4-bit MLX checkpoint; it falls back to a process-specific temporary directory when unset. There is no Qwen2 memory oracle, quality gate, or frozen baseline yet. |
| `TURBOSPARK_DSV2_INSTALL_DIR` | `dsv2_memory_oracle`, `dsv2_quality_gate`, `logit_dump` | Where the real DeepSeek-V2-Lite-Chat Q8_0 `.gturbo` install lives (`~/.turbospark/models/dsv2lite-16b.gturbo`). The two gates skip (with a note) when unset; the oracle runs the protocol at the family's own 8,192/1,024 row (`real_model_params`), and the quality gate needs NO assistant prefix (V2's answer slot is plain prose after `Assistant:`). `logit_dump` takes it as a fallback for the llama.cpp cross-engine arm that closed the numerics gap. |
| `TURBOSPARK_GEMMA4_IQ_INSTALL_DIR` | `gguf_iq_install_network`, `iq3_quality_gate` | Where the 3-bit (IQ3_XXS/IQ4_NL) Gemma 4 install lives (ROADMAP Phase S). A THIRD variable rather than a reuse of the Gemma one, for the reason `iq3_quality_gate` documents: the quality rows are keyed on the chip and freeze the MLX INT4 goldens, so pointing an existing variable at a different artifact asserts the wrong digests. The install test refuses to write to either of the two variables above, and both readers skip or fall back to a temp dir when unset. |
| `TURBOSPARK_LOGIT_DUMP_DIR` | `logit_dump` | Where to write `logits.f16` and `meta.json` (~275 MiB on either family). Required: the target skips when unset, since a few hundred MB is not something to write to a default path. |
| `TURBOSPARK_LOGIT_DUMP_COLD=1` | `logit_dump` | Skips the warmup walk, so the dump comes off a COLD expert cache. That is the condition `quality_gate` takes its perplexity under. It EXISTS to measure the warm/cold difference rather than assume it, and the answer is now zero: AGENTS.md Gotcha 27's fix made the routed dispatch order a function of the route alone, so cold and warm dumps are byte-identical and a warm dump reproduces the frozen cold row exactly (`qwen3moe` warm reads perplexity 14.757589 against a frozen 14.7576). The ~0.5% gap this row used to describe was measured BEFORE that fix; keep the flag, because "the difference is zero" is a claim that has to stay checkable. |
| `TURBOSPARK_PHASES=1` | `turbospark-check` | Prints the per-phase decode breakdown (GPU wait, router readback, expert pread, routed bind, cache hit rate). |
| `TURBOSPARK_DISPATCH_PROFILE=1` | `turbospark-check`, `gpu::PassEncoder` | Ranks the individual dispatches inside each command buffer. Encodes one compute encoder per dispatch (Apple GPUs sample counters only at encoder boundaries) and waits on every buffer, so it perturbs the run it measures: a ranking aid, not a throughput number. Covered by `crates/gpu/tests/dispatch_profile.rs`. |
| `TURBOSPARK_SHARED_CB=0` | `RealForwardRunner` | Reverts the shared-expert branch to encoding after the expert pread instead of on its own overlapping command buffer. The A/B seam for any throughput claim. |
| `TURBOSPARK_ROUTED_PIPELINE=0` | `RealForwardRunner` | Reverts a layer's routed-expert command buffer to rolling uncommitted into the next layer's first buffer instead of committing at the end of the layer (the one-layer pipeline). |

## Test-writing notes

- **Never hardcode a token id from fixture JSON.** The vendored tokenizer
  fixtures carry high placeholder ids (e.g. `248044`) that the `tokenizers`
  crate renumbers at load time. Resolve ids from a loaded `MfTokenizer`
  (`token_to_id`, `end_of_turn_id`).
- **Never assert specific generated text.** Synthetic installs have
  deterministic but untrained weights, so output is structurally real and
  semantically meaningless, and a short generation can decode to the empty
  string. Assert that generation ran: token counts, stop reason, log lines.
- **Build test installs with the helpers**, not a hand-written
  `ArchConfig`: `repack::build_synthetic_gemma4_install` (dense) or its
  `_swa` / `_moe` / `_moe_streamed` variants, or
  `build_synthetic_gemma4_real_install` (verbatim checkpoint naming, which
  is what selects the runner's real Gemma 4 decode flow). Their non-shape
  fields are pinned to `gemma4_26b_a4b()`'s values on purpose.
- **Prefer exact assertions over thresholds.** The steady-state memory
  guarantee is asserted as "zero Metal buffers allocated per token"
  (`dense_decode_allocates_no_gpu_buffers_per_token`), not as an RSS
  threshold, because the exact form cannot go flaky. The one place a
  threshold is unavoidable is the memory oracle, which is why it carries a
  documented headroom policy.
- **Temp dirs** follow the repeated idiom: a `static AtomicU64` counter
  plus `std::env::temp_dir().join(format!("<label>-{pid}-{n}"))`.
- **Metal tests run serially by shared device state** in practice; the
  Swift original runs `swift test --no-parallel` for the same reason. If a
  GPU test starts flaking under parallelism, that is the first thing to
  suspect.
- **Integration tests in one file share a process and its environment.**
  Each `tests/*.rs` is its own binary, but its `#[test]`s run as threads
  inside it, so `std::env::set_var` is global to the file and there is no
  ordering guarantee between cases. A file covering an env-gated feature has
  every test ask for the same setting and reaches the off path another way:
  `real_forward_qwen35_mtp.rs` covers "drafting off" through the open-time
  refusal rather than by unsetting `TURBOSPARK_MTP_DRAFT` mid-run. Never toggle
  a `TURBOSPARK_*` var between tests in one file.
- **A probe that can return a degenerate value has to assert against it, not
  print it.** `crates/bench/tests/mtp_accept_length_probe.rs` measures the
  MTP head's accept length, and on its first run READ ZERO -- because the head
  was broken rather than weak (centered norms read plainly; fixed, and it now
  pays 1.44-1.66x). A version that merely printed its table would have handed
  a reader a quotable "loses" verdict on a question that was not settled. It
  asserts a functional drafter (first proposal accepted above 2%) and fails. The bar separates
  "drafting" from "not drafting", never "pays" from "loses" -- a gate that
  encoded the interesting threshold would be asserting the answer.
  `crates/bench/tests/mtp_head_probe.rs` is the paired instrument that says
  WHY, and `crates/repack/tests/mtp_install_fidelity_network.rs` is the one
  that cleared the weights (see AGENTS.md Gotcha 57).

## Benchmarks

Throughput and memory measurement, including the memory oracle and the
Swift baseline table it asserts against, are documented separately in
[BENCHMARKING.md](BENCHMARKING.md).
